"""
Обучение классификаторов акцентов на MFCC признаках (пункт 4/5 курсовой).

Два уровня моделей:
  Baseline (NN прямого распространения):
    mlp        — mean-pool MFCC → 2-layer MLP

  Enhanced (усложнённые модели):
    mlp_deep   — mean-pool MFCC → 4-layer MLP с BatchNorm и Dropout
    lstm       — MFCC-последовательность → BiLSTM с attention
    cnn        — MFCC-последовательность → Multi-scale 1D CNN

Признаки: MFCC + delta + delta2 (39-dim), outputs/features/mfcc/*.pt

Аргументы:
  --features_dir  outputs/features/mfcc
  --out_dir       outputs/models/mfcc
  --epochs        50
  --batch_size    128
  --device        cuda или cpu
  --models        mlp mlp_deep lstm cnn

Пример:
  python src/models/train_mfcc_classifier.py
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.models.architectures import (
    MLPBaseline, LSTMAccentClassifier, CNNAccentClassifier
)
from src.utils.splits import (
    speaker_independent_split, print_split_info, speaker_from_pt_filename
)

ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]
INPUT_DIM = 39  # MFCC(13) + delta(13) + delta2(13)


# ─────────────────────────────── data loading ─────────────────────────────── #

def load_mfcc_features(features_dir: Path):
    """
    Загружает MFCC признаки из .pt файлов.
    Returns: X_mean (mean-pooled), X_seq (sequences), y_labels, speakers
    """
    X_mean, X_seq, y_list, spk_list = [], [], [], []
    for pt_file in sorted(features_dir.glob("*.pt")):
        try:
            spk = speaker_from_pt_filename(pt_file.stem)
            if spk is None:
                continue
            data = torch.load(pt_file, weights_only=True)
            feat = data["features"].float()   # (T, 39)
            mean_feat = data["mean"].float()   # (39,)
            if feat.shape[0] == 0:
                continue
            X_seq.append(feat)
            X_mean.append(mean_feat)
            y_list.append(data["accent"])
            spk_list.append(spk)
        except Exception:
            continue
    print(f"  Loaded {len(X_mean)} MFCC feature files", flush=True)
    return X_mean, X_seq, y_list, spk_list


# ──────────────────────────── sequence dataset ────────────────────────────── #

class SequenceDataset(Dataset):
    """Хранит последовательности как список, padding — в collate_fn."""

    def __init__(self, sequences: List[torch.Tensor], labels: np.ndarray, max_len: int = 512):
        self.seqs = [s[:max_len] if s.size(0) > max_len else s for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate_sequences(batch):
    seqs, labels = zip(*batch)
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return padded, torch.stack(labels)


# ───────────────────────────── training loop ──────────────────────────────── #

def train_model(
    model: nn.Module,
    model_name: str,
    tr_loader: DataLoader,
    val_loader: DataLoader,
    te_loader: DataLoader,
    n_train: int,
    epochs: int,
    device: str,
    out_dir: Path,
    patience: int = 10,
) -> Dict:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1, best_state, no_improve = -1.0, None, 0
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        scheduler.step()
        tr_loss /= n_train

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                preds = model(Xb.to(device)).argmax(1).cpu()
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
                print(f"    Early stopping at epoch {epoch}", flush=True)
                break

        if epoch % 5 == 0:
            print(f"    Epoch {epoch:3d}/{epochs}: tr_loss={tr_loss:.3f}, "
                  f"val_acc={val_acc:.3f}, val_f1={val_f1:.3f}", flush=True)

    # Test evaluation with best weights
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in te_loader:
            preds = model(Xb.to(device)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(yb.numpy())
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epochs_trained = len(history["train_loss"])
    print(f"    Test: Accuracy={test_acc:.4f}, F1={test_f1:.4f}", flush=True)

    # Save curves
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_curves(history, model_name, out_dir)
    _save_confusion_matrix(all_labels, all_preds, model_name, out_dir)

    return {
        "model": model_name,
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "best_val_f1": round(best_val_f1, 4),
        "epochs_trained": epochs_trained,
    }


def _save_curves(history, model_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title(f"{model_name} — потери (MFCC)")
    axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("Loss")
    axes[1].plot(history["val_acc"], label="Accuracy")
    axes[1].plot(history["val_f1"], label="F1-macro")
    axes[1].set_title(f"{model_name} — метрики (val)")
    axes[1].set_xlabel("Эпоха"); axes[1].legend()
    plt.tight_layout()
    lc_dir = out_dir / "learning_curves"
    lc_dir.mkdir(exist_ok=True)
    plt.savefig(lc_dir / f"curves_{model_name}.png", dpi=120)
    plt.close()


def _save_confusion_matrix(labels, preds, model_name, out_dir):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=ACCENTS, yticklabels=ACCENTS, ax=ax)
    ax.set_title(f"Матрица ошибок — {model_name} (MFCC)")
    ax.set_xlabel("Предсказание"); ax.set_ylabel("Истина")
    plt.tight_layout()
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    plt.savefig(cm_dir / f"cm_{model_name}.png", dpi=120)
    plt.close()


# ──────────────────────────────── main ────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", nargs="+",
                        default=["mlp", "mlp_deep", "lstm", "cnn"])
    args = parser.parse_args()

    features_dir = (Path(args.features_dir) if args.features_dir
                    else _ROOT / "outputs" / "features" / "mfcc")
    out_dir = (Path(args.out_dir) if args.out_dir
               else _ROOT / "outputs" / "models" / "mfcc")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"MFCC Classifier Training")
    print(f"  features_dir: {features_dir}")
    print(f"  out_dir:      {out_dir}")
    print(f"  device:       {device}")
    print(f"  models:       {args.models}", flush=True)

    if not features_dir.exists():
        print(f"ERROR: features_dir not found: {features_dir}")
        print("Run: python src/features/extract_mfcc.py")
        return

    # Load features
    X_mean_all, X_seq_all, y_labels, speakers = load_mfcc_features(features_dir)
    if len(X_mean_all) == 0:
        print("No MFCC features found."); return

    le = LabelEncoder(); le.fit(ACCENTS)
    y_enc = le.transform(y_labels)
    speakers_arr = np.array(speakers)

    # Speaker-independent split
    idx_tr, idx_val, idx_te = speaker_independent_split(speakers_arr, y_enc)
    print_split_info(idx_tr, idx_val, idx_te, speakers_arr, y_enc, "speaker_independent")
    y_tr, y_val, y_te = y_enc[idx_tr], y_enc[idx_val], y_enc[idx_te]

    # Mean-pooled tensors (for MLP models)
    X_mean_tensor = torch.stack(X_mean_all)               # (N, 39)
    Xm_tr = X_mean_tensor[idx_tr]; Xm_val = X_mean_tensor[idx_val]; Xm_te = X_mean_tensor[idx_te]

    def make_mean_loader(X, y, shuffle=False):
        ds = TensorDataset(X, torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    # Sequence datasets (for LSTM/CNN)
    max_len = 512
    seq_tr = SequenceDataset([X_seq_all[i] for i in idx_tr], y_tr, max_len)
    seq_val = SequenceDataset([X_seq_all[i] for i in idx_val], y_val, max_len)
    seq_te = SequenceDataset([X_seq_all[i] for i in idx_te], y_te, max_len)

    def make_seq_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate_sequences, num_workers=0)

    # Model definitions
    model_configs = {
        "mlp": (
            MLPBaseline(input_dim=INPUT_DIM, hidden_dims=(128, 64), dropout=0.3),
            False,  # use mean-pooled
        ),
        "mlp_deep": (
            MLPBaseline(input_dim=INPUT_DIM, hidden_dims=(256, 128, 64), dropout=0.4),
            False,
        ),
        "lstm": (
            LSTMAccentClassifier(input_dim=INPUT_DIM, hidden_size=128,
                                 num_layers=2, dropout=0.3, bidirectional=True),
            True,   # use sequences
        ),
        "cnn": (
            CNNAccentClassifier(input_dim=INPUT_DIM, n_filters=64, dropout=0.3),
            True,
        ),
    }

    results = []
    print(f"\nTrain: {len(idx_tr)}, Val: {len(idx_val)}, Test: {len(idx_te)}", flush=True)

    for mname in args.models:
        if mname not in model_configs:
            print(f"  Unknown model: {mname}, skipping"); continue

        model, use_seq = model_configs[mname]
        print(f"\n  Training {mname} ({'sequence' if use_seq else 'mean-pooled'})...", flush=True)

        if use_seq:
            tr_loader = make_seq_loader(seq_tr, shuffle=True)
            val_loader = make_seq_loader(seq_val)
            te_loader = make_seq_loader(seq_te)
        else:
            tr_loader = make_mean_loader(Xm_tr, y_tr, shuffle=True)
            val_loader = make_mean_loader(Xm_val, y_val)
            te_loader = make_mean_loader(Xm_te, y_te)

        res = train_model(
            model=model, model_name=mname,
            tr_loader=tr_loader, val_loader=val_loader, te_loader=te_loader,
            n_train=len(idx_tr), epochs=args.epochs, device=device, out_dir=out_dir,
        )
        results.append(res)

    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    out_path = out_dir / "experiment_table.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 55)
    print("РЕЗУЛЬТАТЫ (Audio + MFCC)")
    print("=" * 55)
    print(df.to_string(index=False))
    print(f"\nСохранено: {out_path}", flush=True)


if __name__ == "__main__":
    main()
