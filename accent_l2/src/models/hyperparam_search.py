"""
Подбор гиперпараметров для классификатора акцентов (пункт 5 курсовой).
Использует Optuna для поиска оптимальных гиперпараметров LSTM-классификатора.

Пространство поиска:
  hidden_size:  [128, 256, 512]
  num_layers:   [1, 2, 3]
  dropout:      [0.1, 0.2, 0.3, 0.5]
  lr:           [1e-4, 5e-4, 1e-3, 3e-3]
  bidirectional: [True, False]
  batch_size:   [32, 64, 128]

Аргументы:
  --features_dir  путь к outputs/features/hubert
  --out_dir       outputs/hparam_search
  --layer         12 (по умолчанию)
  --model         lstm, transformer, cnn
  --n_trials      число испытаний Optuna (default: 20)
  --epochs        эпох на trial (default: 15)
  --device        cuda или cpu

Пример:
  python src/models/hyperparam_search.py --layer 12 --model lstm --n_trials 20
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


def load_features(features_dir: Path, layer: str = "layer12"):
    from src.utils.splits import speaker_from_pt_filename
    layer_dir = features_dir / layer
    X_list, y_list, spk_list = [], [], []
    for pt_file in sorted(layer_dir.glob("*.pt")):
        try:
            spk = speaker_from_pt_filename(pt_file.stem)
            if spk is None:
                continue
            data = torch.load(pt_file, weights_only=True)
            feat = data["features"].squeeze(0)
            if isinstance(feat, torch.Tensor):
                feat = feat.float()
            X_list.append(feat)
            y_list.append(data.get("accent", ""))
            spk_list.append(spk)
        except Exception:
            continue
    return X_list, y_list, spk_list


class SequenceDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset: stores sequences as list, pads per-batch."""

    def __init__(self, sequences: List[torch.Tensor], labels: np.ndarray, max_len: int = 512):
        self.seqs = [s[:max_len].float() if s.size(0) > max_len else s.float()
                     for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate_sequences(batch):
    seqs, labels = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return padded, torch.stack(labels)


def quick_eval(model, ds: SequenceDataset, device: str) -> float:
    """Evaluate F1-macro on validation set via DataLoader."""
    loader = DataLoader(ds, batch_size=128, collate_fn=collate_sequences, num_workers=0)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            logits = model(Xb.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(yb.numpy())
    return float(f1_score(all_labels, all_preds, average="macro", zero_division=0))


def train_quick(model, tr_ds: SequenceDataset, val_ds: SequenceDataset,
                lr: float, epochs: int, batch_size: int, device: str) -> float:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_sequences, num_workers=0)

    best_f1 = 0.0
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        f1 = quick_eval(model, val_ds, device)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def build_model(model_type: str, trial) -> nn.Module:
    """Build model with hyperparameters suggested by Optuna trial."""
    if model_type == "lstm":
        from src.models.architectures import LSTMAccentClassifier
        return LSTMAccentClassifier(
            hidden_size=trial.suggest_categorical("hidden_size", [128, 256, 512]),
            num_layers=trial.suggest_categorical("num_layers", [1, 2, 3]),
            dropout=trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5]),
            bidirectional=trial.suggest_categorical("bidirectional", [True, False]),
        )
    elif model_type == "transformer":
        from src.models.architectures import TransformerAccentClassifier
        d_model = trial.suggest_categorical("d_model", [128, 256, 512])
        nhead_options = [h for h in [2, 4, 8] if d_model % h == 0]
        return TransformerAccentClassifier(
            d_model=d_model,
            nhead=trial.suggest_categorical("nhead", nhead_options),
            num_layers=trial.suggest_categorical("num_layers", [1, 2, 3]),
            dropout=trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5]),
            dim_feedforward=trial.suggest_categorical("dim_feedforward", [256, 512, 1024]),
        )
    elif model_type == "cnn":
        from src.models.architectures import CNNAccentClassifier
        return CNNAccentClassifier(
            n_filters=trial.suggest_categorical("n_filters", [128, 256, 512]),
            dropout=trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5]),
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--layer", default="12")
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer", "cnn"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    features_dir = Path(args.features_dir) if args.features_dir else _ROOT / "outputs" / "features" / "hubert"
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "outputs" / "hparam_search"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    layer = f"layer{args.layer}"

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("Install optuna: pip install optuna")

    print(f"Подбор гиперпараметров: model={args.model}, layer={layer}, trials={args.n_trials}")
    print(f"Device: {device}")

    # Load and prepare data
    X_seqs, y_labels, speakers = load_features(features_dir, layer)
    le = LabelEncoder(); le.fit(ACCENTS)
    y_enc = le.transform(y_labels)
    speakers_arr = np.array(speakers)

    # Speaker-independent split
    from src.utils.splits import speaker_independent_split, print_split_info
    idx_tr, idx_val, idx_te = speaker_independent_split(speakers_arr, y_enc)
    print_split_info(idx_tr, idx_val, idx_te, speakers_arr, y_enc, "speaker_independent")
    y_tr, y_val, y_te = y_enc[idx_tr], y_enc[idx_val], y_enc[idx_te]

    max_len = 128  # Сокращено с 512: типичное высказывание ~100-150 фреймов, 4x ускорение LSTM
    tr_ds = SequenceDataset([X_seqs[i] for i in idx_tr], y_tr, max_len)
    val_ds = SequenceDataset([X_seqs[i] for i in idx_val], y_val, max_len)
    te_ds = SequenceDataset([X_seqs[i] for i in idx_te], y_te, max_len)

    print(f"Train: {len(idx_tr)}, Val: {len(idx_val)}, Test: {len(idx_te)}", flush=True)

    def objective(trial):
        model = build_model(args.model, trial)
        lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3, 3e-3])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        val_f1 = train_quick(model, tr_ds, val_ds,
                             lr=lr, epochs=args.epochs,
                             batch_size=batch_size, device=device)
        params_str = ", ".join(f"{k}={v}" for k, v in trial.params.items())
        print(f"  Trial {trial.number:2d}: val_f1={val_f1:.4f} | {params_str}", flush=True)
        return val_f1

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials,
                   show_progress_bar=False, gc_after_trial=True)

    print(f"\nЛучшие гиперпараметры (val F1={study.best_value:.4f}):", flush=True)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}", flush=True)

    # Retrain best model on train+val, eval on test
    best_trial = study.best_trial
    best_model = build_model(args.model, best_trial)
    best_lr = best_trial.params["lr"]
    best_bs = best_trial.params["batch_size"]

    idx_trval = np.concatenate([idx_tr, idx_val])
    y_trval = np.concatenate([y_tr, y_val])
    trval_ds = SequenceDataset([X_seqs[i] for i in idx_trval], y_trval, max_len)

    print(f"\nПереобучение лучшей модели на train+val ({len(y_trval)} samples)...", flush=True)
    test_f1 = train_quick(best_model, trval_ds, te_ds,
                          lr=best_lr, epochs=args.epochs * 2,
                          batch_size=best_bs, device=device)
    print(f"Test F1 (лучшая модель): {test_f1:.4f}", flush=True)

    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)

    # Trials dataframe
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / f"{args.model}_{layer}_trials.csv", index=False)

    # Summary
    summary = {
        "model": args.model,
        "layer": layer,
        "n_trials": args.n_trials,
        "best_val_f1": round(study.best_value, 4),
        "test_f1": round(test_f1, 4),
        **{k: v for k, v in study.best_params.items()},
    }
    pd.DataFrame([summary]).to_csv(out_dir / f"{args.model}_{layer}_best.csv", index=False)
    print(f"Результаты: {out_dir}")

    # Plot: optimization history
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(values)
    axes[0].plot(values, "o-", alpha=0.5, label="Значение trial")
    axes[0].plot(best_so_far, "r-", linewidth=2, label="Лучшее значение")
    axes[0].set_title(f"История оптимизации ({args.model}, {layer})")
    axes[0].set_xlabel("Номер trial")
    axes[0].set_ylabel("Val F1-macro")
    axes[0].legend()

    # Parameter importance plot
    try:
        importances = optuna.importance.get_param_importances(study)
        params_sorted = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        names = [p[0] for p in params_sorted]
        imps = [p[1] for p in params_sorted]
        axes[1].barh(names[:10], imps[:10], color="steelblue")
        axes[1].set_title("Важность гиперпараметров")
        axes[1].set_xlabel("Важность (fANOVA)")
    except Exception:
        axes[1].text(0.5, 0.5, "Недостаточно trials\nдля анализа важности",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig(out_dir / f"{args.model}_{layer}_optuna.png", dpi=150)
    plt.close()
    print(f"Сохранён график: {args.model}_{layer}_optuna.png")


if __name__ == "__main__":
    main()
