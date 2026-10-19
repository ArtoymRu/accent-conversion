"""
Обучение и сравнение нейросетевых классификаторов акцентов (пункт 5 курсовой).

Запускает обучение CNN, LSTM, Transformer на HuBERT признаках из outputs/features/hubert.
Сохраняет таблицу экспериментов, кривые обучения, матрицы ошибок.

Аргументы:
  --features_dir  путь к outputs/features/hubert
  --out_dir       папка для результатов (outputs/models)
  --device        cuda или cpu
  --layer         номер слоя HuBERT (6, 9, 12) или all
  --model         mlp, cnn, lstm, transformer, all
  --epochs        число эпох (default: 30)
  --batch_size    размер батча (default: 64)

Пример:
  python src/models/train_classifier.py --layer 12 --model all --epochs 40
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

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.models.architectures import (
    MLPBaseline, CNNAccentClassifier, LSTMAccentClassifier, TransformerAccentClassifier
)

ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]
ACCENT_NAMES_RU = ["арабский", "мандарин", "хинди", "корейский", "испанский", "вьетнамский"]


def load_features(features_dir: Path, layer: str = "layer12"):
    """Load features from .pt files, return (X_list, y_labels, speakers, paths)."""
    from src.utils.splits import speaker_from_pt_filename
    layer_dir = features_dir / layer
    X_list, y_list, spk_list, paths = [], [], [], []
    for pt_file in sorted(layer_dir.glob("*.pt")):
        try:
            spk = speaker_from_pt_filename(pt_file.stem)
            if spk is None:
                continue
            data = torch.load(pt_file, weights_only=True)
            feat = data["features"]
            if isinstance(feat, torch.Tensor):
                feat = feat.float()
            feat = feat.squeeze(0) if feat.dim() == 3 else feat  # (T, 768)
            X_list.append(feat)
            y_list.append(data.get("accent", ""))
            spk_list.append(spk)
            paths.append(str(pt_file))
        except Exception:
            continue
    return X_list, y_list, spk_list, paths


def pad_sequence_batch(sequences: List[torch.Tensor], max_len: int = None) -> torch.Tensor:
    """Pad list of (T_i, D) tensors to (N, max_T, D).
    NOTE: Only use for small datasets. For large datasets, use SequenceDataset + collate_sequences."""
    if max_len is None:
        max_len = max(s.size(0) for s in sequences)
    D = sequences[0].size(1)
    out = torch.zeros(len(sequences), max_len, D)
    for i, s in enumerate(sequences):
        l = min(s.size(0), max_len)
        out[i, :l, :] = s[:l]
    return out


class SequenceDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset for variable-length sequences.
    Stores sequences as a list and pads per-batch via collate_sequences."""

    def __init__(self, sequences: List[torch.Tensor], labels: np.ndarray, max_len: int = 512):
        # Only truncate to max_len — no padding yet (done in collate_fn)
        self.seqs = [s[:max_len].float() if s.size(0) > max_len else s.float() for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate_sequences(batch):
    """Pad a batch of variable-length sequences to the longest in the batch."""
    seqs, labels = zip(*batch)
    # pad_sequence expects list of (T_i, D), pads to max T in batch
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return padded, torch.stack(labels)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, acc, f1, all_preds, all_labels


def train_model(
    model: nn.Module, model_name: str, layer: str,
    X_train, y_train, X_val, y_val, X_test, y_test,
    epochs: int, batch_size: int, device: str, out_dir: Path,
    use_sequences: bool = True, suffix: str = "",
) -> Dict:
    """Full training loop with early stopping and curve saving."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if use_sequences:
        max_len = 512
        tr_ds = SequenceDataset(X_train, y_train, max_len)
        val_ds = SequenceDataset(X_val, y_val, max_len)
        te_ds = SequenceDataset(X_test, y_test, max_len)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_sequences, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                collate_fn=collate_sequences, num_workers=0)
        te_loader = DataLoader(te_ds, batch_size=batch_size,
                               collate_fn=collate_sequences, num_workers=0)
    else:
        X_tr = torch.stack([x.mean(0) for x in X_train])
        X_v = torch.stack([x.mean(0) for x in X_val])
        X_te = torch.stack([x.mean(0) for x in X_test])
        tr_ds = TensorDataset(X_tr, torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(X_v, torch.tensor(y_val, dtype=torch.long))
        te_ds = TensorDataset(X_te, torch.tensor(y_test, dtype=torch.long))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        te_loader = DataLoader(te_ds, batch_size=batch_size)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = -1.0
    best_state = None
    patience = 8
    no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        if epoch % 5 == 0:
            print(f"    Epoch {epoch}/{epochs}: tr_loss={tr_loss:.3f}, "
                  f"val_acc={val_acc:.3f}, val_f1={val_f1:.3f}")

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_acc, test_f1, test_preds, test_labels = eval_epoch(model, te_loader, criterion, device)
    print(f"    Test: Accuracy={test_acc:.4f}, F1={test_f1:.4f}")

    tag = f"{model_name}{suffix}_{layer}"

    # Save model
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / f"{tag}.pt")

    # Learning curves
    _plot_learning_curves(history, f"{model_name}{suffix}", layer, out_dir)

    # Confusion matrix
    _plot_confusion_matrix(test_labels, test_preds, f"{model_name}{suffix}", layer, out_dir)

    return {
        "model": f"{model_name}{suffix}", "layer": layer,
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "best_val_f1": round(best_val_f1, 4),
        "epochs_trained": len(history["train_loss"]),
    }


def _plot_learning_curves(history, model_name, layer, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Val loss")
    axes[0].set_title(f"{model_name} ({layer}) — кривые потерь")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["val_acc"], label="Val Accuracy")
    axes[1].plot(history["val_f1"], label="Val F1-macro")
    axes[1].set_title(f"{model_name} ({layer}) — метрики качества")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    curves_dir = out_dir / "learning_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(curves_dir / f"curves_{model_name}_{layer}.png", dpi=120)
    plt.close()


def _plot_confusion_matrix(labels, preds, model_name, layer, out_dir: Path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=ACCENTS, yticklabels=ACCENTS, ax=ax)
    ax.set_title(f"Матрица ошибок — {model_name} ({layer})")
    ax.set_xlabel("Предсказанный акцент")
    ax.set_ylabel("Истинный акцент")
    plt.tight_layout()
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_dir / f"cm_{model_name}_{layer}.png", dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train accent classifiers")
    parser.add_argument("--features_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--layer", default="12", help="6, 9, 12, or all")
    parser.add_argument("--model", default="all", help="mlp, cnn, lstm, transformer, or all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for all models")
    parser.add_argument("--suffix", default="", help="Suffix appended to model/plot names (e.g. '_d05')")
    args = parser.parse_args()

    features_dir = Path(args.features_dir) if args.features_dir else _ROOT / "outputs" / "features" / "hubert"
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "outputs" / "models"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    layers = ["layer6", "layer9", "layer12"] if args.layer == "all" else [f"layer{args.layer}"]
    model_names = ["mlp", "cnn", "lstm", "transformer"] if args.model == "all" else [args.model]

    print(f"Device: {device}, Layers: {layers}, Models: {model_names}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    import pandas as pd
    all_results = []

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer: {layer}")
        print("="*60)

        X_seqs, y_labels, speakers, _ = load_features(features_dir, layer)
        if not X_seqs:
            print(f"  No features found for {layer}")
            continue

        le = LabelEncoder()
        le.fit(ACCENTS)
        y_enc = le.transform(y_labels)
        speakers_arr = np.array(speakers)

        # Speaker-independent split: held-out 1 speaker per accent for test
        from src.utils.splits import speaker_independent_split, print_split_info
        idx_tr, idx_val, idx_te = speaker_independent_split(speakers_arr, y_enc)
        print_split_info(idx_tr, idx_val, idx_te, speakers_arr, y_enc,
                         protocol="speaker_independent")

        X_tr = [X_seqs[i] for i in idx_tr]
        X_val = [X_seqs[i] for i in idx_val]
        X_te = [X_seqs[i] for i in idx_te]
        y_tr, y_val, y_te = y_enc[idx_tr], y_enc[idx_val], y_enc[idx_te]

        print(f"  Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")

        d = args.dropout
        print(f"  Dropout: {d}")
        model_configs = {
            "mlp": (MLPBaseline(hidden_dims=(512, 256), dropout=d), False),
            "cnn": (CNNAccentClassifier(n_filters=256, dropout=d), True),
            "lstm": (LSTMAccentClassifier(hidden_size=256, num_layers=2, dropout=d), True),
            "transformer": (TransformerAccentClassifier(d_model=256, nhead=4, num_layers=2,
                                                         dropout=d), True),
        }

        for mname in model_names:
            if mname not in model_configs:
                print(f"  Unknown model: {mname}")
                continue
            model, use_seq = model_configs[mname]
            print(f"\n  Training {mname}...")
            result = train_model(
                model=model, model_name=mname, layer=layer,
                X_train=X_tr, y_train=y_tr,
                X_val=X_val, y_val=y_val,
                X_test=X_te, y_test=y_te,
                epochs=args.epochs, batch_size=args.batch_size,
                device=device, out_dir=out_dir,
                use_sequences=use_seq, suffix=args.suffix,
            )
            all_results.append(result)

    # Save experiment table
    if all_results:
        df = pd.DataFrame(all_results)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "experiment_table.csv", index=False)
        print("\n" + "="*60)
        print("ТАБЛИЦА ЭКСПЕРИМЕНТОВ (Нейросетевые классификаторы)")
        print("="*60)
        print(df.to_string(index=False))
        print(f"\nСохранено: {out_dir / 'experiment_table.csv'}")


if __name__ == "__main__":
    main()
