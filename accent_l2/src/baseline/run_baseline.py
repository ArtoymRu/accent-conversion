"""
Baseline эксперименты (пункт 4 курсовой работы).

Запускает три вида baseline:
1. Классификация акцентов по HuBERT признакам (logistic regression, MLP)
2. EnCodec реконструкция + MCD по ширине полосы (1.5 / 3.0 / 6.0 kbps)
3. WER оригинальной речи (Whisper small ASR) — опционально (--run_wer)

Аргументы:
  --data_root  путь к data/l2_arctic (по умолчанию: project/data/l2_arctic)
  --features_dir  путь к extracted HuBERT features (outputs/features/hubert)
  --out_dir    папка для сохранения результатов (outputs/baseline)
  --device     cuda или cpu
  --max_samples  макс. файлов на акцент для MCD/WER (default: 30)
  --run_wer    если флаг выставлен — запускает WER (требует загрузки Whisper)
  --skip_mcd   пропустить EnCodec baseline
  --skip_cls   пропустить классификацию

Пример:
  python src/baseline/run_baseline.py --device cuda --max_samples 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def run_classification_baseline(features_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Logistic Regression / SVM / MLP на mean-pooled HuBERT признаках, слои 6/9/12.

    Два протокола оценки:
      random         — случайный split по высказываниям (с утечкой спикеров)
      speaker_indep  — held-out 1 спикер на акцент (честная оценка)
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    from src.utils.splits import (
        speaker_independent_split, random_utterance_split,
        speaker_from_pt_filename, print_split_info,
    )

    rows = []
    ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]

    for layer_dir in sorted(features_dir.glob("layer*")):
        layer_name = layer_dir.name
        print(f"\n  Layer: {layer_name}")

        # Load features + speaker codes
        X_list, y_list, spk_list = [], [], []
        for pt_file in sorted(layer_dir.glob("*.pt")):
            try:
                spk = speaker_from_pt_filename(pt_file.stem)
                if spk is None:
                    continue
                data = torch.load(pt_file, weights_only=True)
                feat = data["features"]
                if isinstance(feat, torch.Tensor):
                    feat = feat.numpy()
                feat = feat.squeeze(0) if feat.ndim == 3 else feat
                X_list.append(feat.mean(axis=0))
                y_list.append(data.get("accent", ""))
                spk_list.append(spk)
            except Exception:
                continue

        if not X_list:
            print(f"  No features in {layer_dir}")
            continue

        X = np.stack(X_list)
        speakers = np.array(spk_list)
        le = LabelEncoder()
        le.fit(ACCENTS)
        y = le.transform(y_list)

        classifiers = {
            "LogReg": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "SVM_RBF": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                                 early_stopping=True, random_state=42),
        }

        for protocol in ("random", "speaker_indep"):
            if protocol == "speaker_indep":
                idx_tr, idx_val, idx_te = speaker_independent_split(speakers, y)
            else:
                idx_tr, idx_val, idx_te = random_utterance_split(y)

            print_split_info(idx_tr, idx_val, idx_te, speakers, y,
                             protocol="speaker_independent" if protocol == "speaker_indep" else "random")

            # combine train+val for sklearn (no early stopping needed)
            idx_tv = np.concatenate([idx_tr, idx_val])
            X_tv, y_tv = X[idx_tv], y[idx_tv]
            X_te, y_te = X[idx_te], y[idx_te]

            scaler = StandardScaler()
            X_tv_sc = scaler.fit_transform(X_tv)
            X_te_sc = scaler.transform(X_te)

            for clf_name, clf in classifiers.items():
                import sklearn
                clf_fresh = sklearn.clone(clf)
                clf_fresh.fit(X_tv_sc, y_tv)
                y_pred = clf_fresh.predict(X_te_sc)
                acc = accuracy_score(y_te, y_pred)
                f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
                print(f"  [{protocol}] {clf_name}: Accuracy={acc:.3f}, F1={f1:.3f}")
                rows.append({
                    "experiment": f"HuBERT_{layer_name}_{clf_name}_{protocol}",
                    "model": "HuBERT",
                    "layer": layer_name,
                    "classifier": clf_name,
                    "protocol": protocol,
                    "accuracy": round(acc, 4),
                    "f1_macro": round(f1, 4),
                    "metric_type": "classification",
                })

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "classification_baseline.csv", index=False)

    # Print comparison table
    if len(df) > 0:
        pivot = df.pivot_table(
            index=["layer", "classifier"],
            columns="protocol",
            values="f1_macro",
        ).round(3)
        print("\n--- F1-macro: random vs speaker_indep ---")
        print(pivot.to_string())
    return df


def run_mcd_baseline(data_root: Path, out_dir: Path, max_per_accent: int, device: str) -> pd.DataFrame:
    """EnCodec encode-decode MCD baseline."""
    from src.baseline.encodec_baseline import run_encodec_baseline
    result = run_encodec_baseline(
        data_root=data_root,
        out_dir=out_dir,
        max_per_accent=max_per_accent,
        bandwidths=[1.5, 3.0, 6.0],
        run_asr=False,
        device=device,
    )
    df = result["df"]
    # Build rows for experiment table
    rows = []
    for bw in [1.5, 3.0, 6.0]:
        col = f"mcd_bw{bw}"
        if col not in df.columns:
            continue
        mcd_mean = df[col].mean()
        mcd_std = df[col].std()
        rows.append({
            "experiment": f"EnCodec_{bw}kbps",
            "model": "EnCodec",
            "bandwidth_kbps": bw,
            "mcd_mean": round(mcd_mean, 3),
            "mcd_std": round(mcd_std, 3),
            "metric_type": "reconstruction",
        })
    return pd.DataFrame(rows)


def build_experiment_table(cls_df: pd.DataFrame, mcd_df: pd.DataFrame,
                            wer_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Merge all results into a single experiment comparison table."""
    parts = []

    if cls_df is not None and len(cls_df) > 0:
        # Classification part: accuracy as main metric
        cls_pivot = cls_df[["experiment", "model", "layer", "classifier", "accuracy", "f1_macro"]].copy()
        cls_pivot["mcd_mean"] = "-"
        cls_pivot["wer_mean"] = "-"
        parts.append(cls_pivot)

    if mcd_df is not None and len(mcd_df) > 0:
        mcd_pivot = mcd_df[["experiment", "model", "bandwidth_kbps", "mcd_mean", "mcd_std"]].copy()
        mcd_pivot["accuracy"] = "-"
        mcd_pivot["f1_macro"] = "-"
        mcd_pivot["wer_mean"] = "-"
        parts.append(mcd_pivot)

    if wer_df is not None and len(wer_df) > 0:
        parts.append(wer_df)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True, sort=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "experiment_table.csv", index=False)

    print("\n" + "=" * 70)
    print("ТАБЛИЦА ЭКСПЕРИМЕНТОВ (BASELINE)")
    print("=" * 70)
    print(combined.to_string(index=False))
    print(f"\nСохранено: {out_dir / 'experiment_table.csv'}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--run_wer", action="store_true")
    parser.add_argument("--skip_mcd", action="store_true")
    parser.add_argument("--skip_cls", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else _ROOT / "data" / "l2_arctic"
    features_dir = Path(args.features_dir) if args.features_dir else _ROOT / "outputs" / "features" / "hubert"
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "outputs" / "baseline"

    print(f"Data root:    {data_root}")
    print(f"Features dir: {features_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Device:       {args.device}")

    cls_df, mcd_df, wer_df = None, None, None

    # --- 1. Classification baseline ---
    if not args.skip_cls and features_dir.exists():
        print("\n" + "=" * 60)
        print("1. КЛАССИФИКАЦИЯ АКЦЕНТОВ (HuBERT features)")
        print("=" * 60)
        cls_df = run_classification_baseline(features_dir, out_dir)

    # --- 2. EnCodec MCD baseline ---
    if not args.skip_mcd and data_root.exists():
        print("\n" + "=" * 60)
        print("2. ENCODEC РЕКОНСТРУКЦИЯ + MCD")
        print("=" * 60)
        mcd_df = run_mcd_baseline(data_root, out_dir / "encodec",
                                   max_per_accent=args.max_samples,
                                   device=args.device)

    # --- 3. WER baseline (optional) ---
    if args.run_wer and data_root.exists():
        print("\n" + "=" * 60)
        print("3. WER ОРИГИНАЛЬНОЙ РЕЧИ (Whisper)")
        print("=" * 60)
        from src.baseline.wer_baseline import run_wer_baseline
        wer_result = run_wer_baseline(
            data_root=data_root,
            out_dir=out_dir / "wer",
            max_per_accent=args.max_samples,
            device=args.device,
        )
        wer_summary = wer_result["summary"].reset_index()
        wer_summary.columns = ["accent", "wer_mean", "wer_std", "count"]
        wer_summary["experiment"] = wer_summary["accent"].apply(lambda a: f"WER_orig_{a}")
        wer_summary["model"] = "Whisper-small"
        wer_df = wer_summary

    # --- Build combined experiment table ---
    build_experiment_table(cls_df, mcd_df, wer_df, out_dir)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
