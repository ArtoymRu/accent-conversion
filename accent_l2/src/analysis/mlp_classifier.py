"""
MLP baseline for accent classification on HuBERT features.

Usage:
    python src/analysis/mlp_classifier.py [--features_dir outputs/features] [--layer 12]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.accent_map import ACCENTS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(features_dir: Path, layer: int):
    """Load mean-pooled HuBERT features + labels for a given layer."""
    root = features_dir / "hubert" / f"layer{layer}"
    if not root.exists():
        raise FileNotFoundError(f"Feature dir not found: {root}")

    # Only load files with the speaker-prefixed format: {SPEAKER}_{basename}.pt
    # (filters out legacy files that lack the speaker prefix)
    from src.data.accent_map import SPEAKER_TO_ACCENT
    valid_speakers = set(SPEAKER_TO_ACCENT.keys())

    X_list, y_list, paths = [], [], []
    for pt_path in sorted(root.glob("*.pt")):
        stem = pt_path.stem
        speaker_part = stem.split("_")[0].upper()
        if speaker_part not in valid_speakers:
            continue  # skip legacy un-prefixed files
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        feat = data["features"]
        if feat.dim() == 3:
            feat = feat.squeeze(0)  # (T, D)
        x = feat.mean(dim=0).numpy()  # mean-pool over time -> (D,)
        accent_id = int(data["accent_id"])
        if accent_id < 0:
            continue
        X_list.append(x)
        y_list.append(accent_id)
        paths.append(str(pt_path))

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AccentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp(
    X_train, y_train, X_val, y_val,
    hidden_dims=(512, 256),
    num_classes=6,
    epochs=100,
    lr=1e-3,
    batch_size=256,
    weight_decay=1e-4,
    dropout=0.3,
    device="cpu",
    patience=15,
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_vl = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val, dtype=torch.long).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = AccentMLP(X_train.shape[1], list(hidden_dims), num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_vl).argmax(dim=1)
            val_acc = (preds == y_vl).float().mean().item()

        history["train_loss"].append(total_loss / len(X_train))
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, scaler, history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model, scaler, X_test, y_test, device="cpu"):
    X_scaled = scaler.transform(X_test)
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, target_names=ACCENTS, digits=4)
    cm = confusion_matrix(y_test, preds)
    return acc, f1, preds, report, cm


def save_confusion_matrix(cm, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=ACCENTS, yticklabels=ACCENTS, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("MLP Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_learning_curve(history, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Loss")
    ax2.plot(history["val_acc"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Validation Accuracy")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLP baseline for accent classification")
    parser.add_argument("--features_dir", default="outputs/features")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--hidden_dims", default="512,256", help="Comma-separated hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--output_dir", default="outputs/classification/mlp")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = base / features_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    num_classes = len(ACCENTS)

    print(f"Loading features from layer {args.layer}...")
    X, y = load_features(features_dir, args.layer)
    print(f"  {X.shape[0]} samples, {X.shape[1]}-dim features, {num_classes} classes")
    unique, counts = np.unique(y, return_counts=True)
    for uid, cnt in zip(unique, counts):
        print(f"  {ACCENTS[uid]}: {cnt}")

    # Train/val/test split (stratified)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(X, y))
    X_trainval, y_trainval = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    val_frac = args.val_size / (1.0 - args.test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    train_idx, val_idx = next(sss_val.split(X_trainval, y_trainval))
    X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
    X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

    print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Architecture: {X_train.shape[1]} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
    print(f"Device: {args.device}\n")

    model, scaler, history = train_mlp(
        X_train, y_train, X_val, y_val,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        device=args.device,
        patience=20,
    )

    acc, f1, preds, report, cm = evaluate(model, scaler, X_test, y_test, device=args.device)
    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Test F1 (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(report)

    # Save outputs
    torch.save({"model_state": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_std": scaler.scale_}, out_dir / "mlp_model.pt")
    save_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    save_learning_curve(history, out_dir / "learning_curve.png")

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "accuracy", "f1_macro"])
        w.writeheader()
        w.writerow({"layer": args.layer, "accuracy": f"{acc:.4f}", "f1_macro": f"{f1:.4f}"})

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(f"Layer {args.layer} | Accuracy={acc:.4f} | F1-macro={f1:.4f}\n\n")
        f.write(report)

    print(f"\nOutputs saved to {out_dir}")


if __name__ == "__main__":
    main()
