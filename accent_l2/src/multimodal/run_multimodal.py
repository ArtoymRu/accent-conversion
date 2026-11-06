"""
Мультимодальный эксперимент: текст + аудио (пункт 6 курсовой).

Запускает и сравнивает три стратегии fusion:
  - early_fusion:  конкатенация признаков
  - late_fusion:   независимые ветки + взвешенная сумма
  - cross_attention: cross-attention между модальностями

Аргументы:
  --data_root      путь к data/l2_arctic
  --features_dir   путь к outputs/features/hubert
  --out_dir        outputs/multimodal
  --layer          12 (по умолчанию)
  --epochs         30
  --device         cuda или cpu

Пример:
  python src/multimodal/run_multimodal.py --layer 12 --epochs 30
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.multimodal.text_features import load_transcripts, TFIDFTextEncoder
from src.multimodal.fusion_model import (
    EarlyFusionClassifier, LateFusionClassifier, CrossAttentionFusionClassifier
)

ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


def load_paired_data(features_dir: Path, data_root: Path, layer: str = "layer12"):
    """
    Load paired (audio_feat, text, accent, speaker) for samples that have
    both an extracted HuBERT feature file AND a transcript.
    Returns: audio_feats, texts, accents, speakers
    """
    from src.data.accent_map import SPEAKER_TO_ACCENT

    print("Loading transcripts...")
    transcripts = load_transcripts(data_root)
    print(f"  Loaded {len(transcripts)} transcripts")

    layer_dir = features_dir / layer
    audio_feats, texts, accents, speakers = [], [], [], []

    for pt_file in sorted(layer_dir.glob("*.pt")):
        stem = pt_file.stem
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        speaker, utterance = parts[0].upper(), parts[1]
        if speaker not in SPEAKER_TO_ACCENT:
            continue

        txt = transcripts.get((speaker, utterance))
        if txt is None:
            continue

        try:
            data = torch.load(pt_file, weights_only=True)
            feat = data["features"].squeeze(0)
            audio_mean = feat.mean(0).numpy()
        except Exception:
            continue

        audio_feats.append(audio_mean)
        texts.append(txt)
        accents.append(data.get("accent", ""))
        speakers.append(speaker)

    print(f"  Paired samples: {len(audio_feats)}")
    return audio_feats, texts, accents, speakers


def train_fusion_model(
    model, model_name: str,
    X_audio_tr, X_text_tr, y_tr,
    X_audio_val, X_text_val, y_val,
    X_audio_te, X_text_te, y_te,
    epochs: int, batch_size: int, device: str, out_dir: Path,
) -> Dict:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def make_loader(Xa, Xt, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(Xt, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_loader = make_loader(X_audio_tr, X_text_tr, y_tr, shuffle=True)
    val_loader = make_loader(X_audio_val, X_text_val, y_val)
    te_loader = make_loader(X_audio_te, X_text_te, y_te)

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1, best_state, patience, no_improve = -1.0, None, 8, 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xa, Xt, yb in tr_loader:
            Xa, Xt, yb = Xa.to(device), Xt.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xa, Xt)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        scheduler.step()
        tr_loss /= len(X_audio_tr)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xa, Xt, yb in val_loader:
                logits = model(Xa.to(device), Xt.to(device))
                preds = logits.argmax(1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(yb.numpy())
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop @ epoch {epoch}")
                break

        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: loss={tr_loss:.3f}, val_acc={val_acc:.3f}, val_f1={val_f1:.3f}")

    # Test evaluation
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xa, Xt, yb in te_loader:
            logits = model(Xa.to(device), Xt.to(device))
            preds = logits.argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(yb.numpy())
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"    Test: Accuracy={test_acc:.4f}, F1={test_f1:.4f}")

    # Save plots
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_fusion_curves(history, model_name, out_dir)
    _save_confusion_matrix(all_labels, all_preds, model_name, out_dir)

    return {
        "model": model_name,
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "best_val_f1": round(best_val_f1, 4),
    }


def _save_fusion_curves(history, model_name, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title(f"{model_name} — потери")
    axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("Loss")
    axes[1].plot(history["val_acc"], label="Accuracy")
    axes[1].plot(history["val_f1"], label="F1-macro")
    axes[1].set_title(f"{model_name} — метрики (val)")
    axes[1].set_xlabel("Эпоха"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"curves_{model_name}.png", dpi=120)
    plt.close()


def _save_confusion_matrix(labels, preds, model_name, out_dir: Path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=ACCENTS, yticklabels=ACCENTS, ax=ax)
    ax.set_title(f"Матрица ошибок — {model_name}")
    ax.set_xlabel("Предсказание"); ax.set_ylabel("Истина")
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_{model_name}.png", dpi=120)
    plt.close()


def ablation_audio_only(X_audio_tr, y_tr, X_audio_val, y_val, X_audio_te, y_te):
    """Quick sklearn ablation: audio-only baseline for comparison."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_audio_tr, y_tr)
    preds = clf.predict(X_audio_te)
    return {
        "model": "AudioOnly_LogReg",
        "test_accuracy": round(accuracy_score(y_te, preds), 4),
        "test_f1": round(f1_score(y_te, preds, average="macro", zero_division=0), 4),
        "best_val_f1": "-",
    }


def ablation_text_only(X_text_tr, y_tr, X_text_te, y_te):
    """Quick sklearn ablation: text-only baseline."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_text_tr, y_tr)
    preds = clf.predict(X_text_te)
    return {
        "model": "TextOnly_LogReg",
        "test_accuracy": round(accuracy_score(y_te, preds), 4),
        "test_f1": round(f1_score(y_te, preds, average="macro", zero_division=0), 4),
        "best_val_f1": "-",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--layer", default="12")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text_dim", type=int, default=512)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else _ROOT / "data" / "l2_arctic"
    features_dir = Path(args.features_dir) if args.features_dir else _ROOT / "outputs" / "features" / "hubert"
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "outputs" / "multimodal"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    layer = f"layer{args.layer}"

    print(f"Device: {device}, Layer: {layer}, Epochs: {args.epochs}")

    # Load paired audio + text data
    audio_feats, texts, accents, speakers = load_paired_data(features_dir, data_root, layer)
    if len(audio_feats) == 0:
        print("No paired data found. Check features_dir and data_root.")
        return

    # Encode text
    print("Fitting TF-IDF on transcripts...")
    encoder = TFIDFTextEncoder(mode="char", ngram_range=(2, 4), max_features=args.text_dim)
    X_text_all = encoder.fit_transform(texts)
    X_audio_all = np.stack(audio_feats)

    le = LabelEncoder(); le.fit(ACCENTS)
    y_all = le.transform(accents)
    speakers_arr = np.array(speakers)

    # Speaker-independent split: held-out 1 speaker per accent for test
    from src.utils.splits import speaker_independent_split, print_split_info
    idx_tr, idx_val, idx_te = speaker_independent_split(speakers_arr, y_all)
    print_split_info(idx_tr, idx_val, idx_te, speakers_arr, y_all, "speaker_independent")
    y_tr, y_val, y_te = y_all[idx_tr], y_all[idx_val], y_all[idx_te]

    Xa_tr, Xa_val, Xa_te = X_audio_all[idx_tr], X_audio_all[idx_val], X_audio_all[idx_te]
    Xt_tr, Xt_val, Xt_te = X_text_all[idx_tr], X_text_all[idx_val], X_text_all[idx_te]
    print(f"Train: {len(idx_tr)}, Val: {len(idx_val)}, Test: {len(idx_te)}")

    audio_dim = Xa_tr.shape[1]
    text_dim = Xt_tr.shape[1]
    print(f"Audio dim: {audio_dim}, Text dim: {text_dim}")

    results = []

    # Ablation: audio only and text only
    print("\n--- Ablation: Audio only ---")
    results.append(ablation_audio_only(Xa_tr, y_tr, Xa_val, y_val, Xa_te, y_te))
    print("\n--- Ablation: Text only ---")
    results.append(ablation_text_only(Xt_tr, y_tr, Xt_te, y_te))

    # Multimodal fusion models
    fusion_models = {
        "early_fusion": EarlyFusionClassifier(audio_dim=audio_dim, text_dim=text_dim),
        "late_fusion": LateFusionClassifier(audio_dim=audio_dim, text_dim=text_dim),
        "cross_attention": CrossAttentionFusionClassifier(audio_dim=audio_dim, text_dim=text_dim),
    }

    for mname, model in fusion_models.items():
        print(f"\n--- {mname} ---")
        res = train_fusion_model(
            model=model, model_name=mname,
            X_audio_tr=Xa_tr, X_text_tr=Xt_tr, y_tr=y_tr,
            X_audio_val=Xa_val, X_text_val=Xt_val, y_val=y_val,
            X_audio_te=Xa_te, X_text_te=Xt_te, y_te=y_te,
            epochs=args.epochs, batch_size=args.batch_size,
            device=device, out_dir=out_dir,
        )
        results.append(res)

    # Summary table
    df = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "multimodal_experiment_table.csv", index=False)

    print("\n" + "=" * 60)
    print("ТАБЛИЦА ЭКСПЕРИМЕНТОВ (Объединение модальностей)")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nСохранено: {out_dir / 'multimodal_experiment_table.csv'}")

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["test_accuracy"], width=0.4, label="Accuracy", color="steelblue")
    ax.bar(x + 0.2, df["test_f1"].astype(float), width=0.4, label="F1-macro", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Сравнение методов объединения модальностей")
    ax.set_ylabel("Значение метрики")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fusion_comparison.png", dpi=150)
    plt.close()
    print("Saved: fusion_comparison.png")


if __name__ == "__main__":
    main()
