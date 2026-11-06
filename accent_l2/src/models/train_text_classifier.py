"""
Обучение классификаторов акцентов на текстовых признаках (пункт 4/5 курсовой).

Текстовая модальность: транскрипты из data/l2_arctic/ACCENT/SPEAKER/transcript/

Два вида TF-IDF признаков (512-dim каждый):
  char  — char n-gram TF-IDF, ngram_range=(2,4)
  word  — word n-gram TF-IDF, ngram_range=(1,2)

Два уровня моделей для каждого признака:
  Baseline:  MLP (NN прямого распространения) — 2 скрытых слоя
  Enhanced:  Глубокий MLP с BatchNorm — 4 скрытых слоя

Примечание: Все спикеры читают одни и те же CMU-ARCTIC предложения,
поэтому текстовые признаки не несут информацию об акценте в задаче
speaker-independent — это подтверждается результатами экспериментов.

Аргументы:
  --data_root   путь к data/l2_arctic
  --out_dir     outputs/models/text
  --epochs      50
  --batch_size  128
  --device      cuda или cpu

Пример:
  python src/models/train_text_classifier.py
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

from src.utils.splits import speaker_independent_split, print_split_info
from src.multimodal.text_features import load_transcripts, TFIDFTextEncoder

ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


# ─────────────────────────── MLP архитектуры ──────────────────────────────── #

class MLPBaseline(nn.Module):
    """MLP baseline: TF-IDF → 2-layer MLP."""

    def __init__(self, input_dim: int = 512, hidden_dims=(256, 128),
                 n_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPDeep(nn.Module):
    """Глубокий MLP: TF-IDF → 4-layer MLP с BatchNorm (enhanced model)."""

    def __init__(self, input_dim: int = 512, hidden_dims=(512, 256, 128, 64),
                 n_classes: int = 6, dropout: float = 0.4):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────── training utilities ───────────────────────────── #

def train_mlp(
    model: nn.Module,
    model_name: str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    epochs: int,
    batch_size: int,
    device: str,
    out_dir: Path,
    patience: int = 10,
) -> Dict:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def to_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    tr_loader = to_loader(X_tr, y_tr, shuffle=True)
    val_loader = to_loader(X_val, y_val)
    te_loader = to_loader(X_te, y_te)

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        scheduler.step()
        tr_loss /= len(y_tr)

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

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}/{epochs}: loss={tr_loss:.4f}, "
                  f"val_acc={val_acc:.3f}, val_f1={val_f1:.3f}", flush=True)

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
    print(f"    Test: Accuracy={test_acc:.4f}, F1={test_f1:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_curves(history, model_name, out_dir)
    _save_cm(all_labels, all_preds, model_name, out_dir)

    return {
        "model": model_name,
        "test_accuracy": round(test_acc, 4),
        "test_f1": round(test_f1, 4),
        "best_val_f1": round(best_val_f1, 4),
        "epochs_trained": len(history["train_loss"]),
    }


def _save_curves(history, name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title(f"{name} — потери (Text)")
    axes[0].set_xlabel("Эпоха"); axes[0].set_ylabel("Loss")
    axes[1].plot(history["val_acc"], label="Accuracy")
    axes[1].plot(history["val_f1"], label="F1-macro")
    axes[1].set_title(f"{name} — метрики (val)")
    axes[1].set_xlabel("Эпоха"); axes[1].legend()
    plt.tight_layout()
    lc_dir = out_dir / "learning_curves"
    lc_dir.mkdir(exist_ok=True)
    plt.savefig(lc_dir / f"curves_{name}.png", dpi=120)
    plt.close()


def _save_cm(labels, preds, name, out_dir):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=ACCENTS, yticklabels=ACCENTS, ax=ax)
    ax.set_title(f"Матрица ошибок — {name}")
    ax.set_xlabel("Предсказание"); ax.set_ylabel("Истина")
    plt.tight_layout()
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    plt.savefig(cm_dir / f"cm_{name}.png", dpi=120)
    plt.close()


# ─────────────────────────────── main ─────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--text_dim", type=int, default=512)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else _ROOT / "data" / "l2_arctic"
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "outputs" / "models" / "text"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Text Classifier Training")
    print(f"  data_root: {data_root}")
    print(f"  out_dir:   {out_dir}")
    print(f"  device:    {device}", flush=True)

    # Load transcripts
    print("\nLoading transcripts...", flush=True)
    transcripts = load_transcripts(data_root)
    if not transcripts:
        print("No transcripts found. Check data_root.")
        return
    print(f"  Loaded {len(transcripts)} transcripts", flush=True)

    # Build dataset: (speaker, utterance, text, accent)
    from src.data.accent_map import SPEAKER_TO_ACCENT
    items = []
    for (speaker, utt), text in sorted(transcripts.items()):
        accent = SPEAKER_TO_ACCENT.get(speaker)
        if accent:
            items.append((speaker, utt, text, accent))
    print(f"  Items with known speaker: {len(items)}", flush=True)

    speakers_arr = np.array([x[0] for x in items])
    texts_all = [x[2] for x in items]
    accents_all = [x[3] for x in items]

    le = LabelEncoder(); le.fit(ACCENTS)
    y_enc = le.transform(accents_all)

    # Speaker-independent split (indices into items)
    idx_tr, idx_val, idx_te = speaker_independent_split(speakers_arr, y_enc)
    print_split_info(idx_tr, idx_val, idx_te, speakers_arr, y_enc, "speaker_independent")
    y_tr, y_val, y_te = y_enc[idx_tr], y_enc[idx_val], y_enc[idx_te]

    texts_tr = [texts_all[i] for i in idx_tr]
    texts_val = [texts_all[i] for i in idx_val]
    texts_te = [texts_all[i] for i in idx_te]

    print(f"\nTrain: {len(idx_tr)}, Val: {len(idx_val)}, Test: {len(idx_te)}", flush=True)

    # ── Feature set 1: char n-gram TF-IDF (2,4) ──────────────────────────── #
    print("\n" + "=" * 55)
    print("Feature: char TF-IDF (ngram 2-4)")
    print("=" * 55)
    char_enc = TFIDFTextEncoder(mode="char", ngram_range=(2, 4),
                                max_features=args.text_dim)
    X_char_tr = char_enc.fit_transform(texts_tr)  # fit on train only
    X_char_val = char_enc.transform(texts_val)
    X_char_te = char_enc.transform(texts_te)
    print(f"  Char TF-IDF dim: {X_char_tr.shape[1]}", flush=True)

    char_results = []

    print("\n  Training mlp_baseline (char)...", flush=True)
    res = train_mlp(
        MLPBaseline(input_dim=X_char_tr.shape[1], hidden_dims=(256, 128), dropout=0.3),
        "char_mlp_baseline",
        X_char_tr, y_tr, X_char_val, y_val, X_char_te, y_te,
        epochs=args.epochs, batch_size=args.batch_size, device=device, out_dir=out_dir,
    )
    char_results.append({**res, "feature": "char_tfidf"})

    print("\n  Training mlp_deep (char)...", flush=True)
    res = train_mlp(
        MLPDeep(input_dim=X_char_tr.shape[1], hidden_dims=(512, 256, 128, 64), dropout=0.4),
        "char_mlp_deep",
        X_char_tr, y_tr, X_char_val, y_val, X_char_te, y_te,
        epochs=args.epochs, batch_size=args.batch_size, device=device, out_dir=out_dir,
    )
    char_results.append({**res, "feature": "char_tfidf"})

    # ── Feature set 2: word n-gram TF-IDF (1,2) ──────────────────────────── #
    print("\n" + "=" * 55)
    print("Feature: word TF-IDF (ngram 1-2)")
    print("=" * 55)
    word_enc = TFIDFTextEncoder(mode="word", ngram_range=(1, 2),
                                max_features=args.text_dim)
    X_word_tr = word_enc.fit_transform(texts_tr)
    X_word_val = word_enc.transform(texts_val)
    X_word_te = word_enc.transform(texts_te)
    print(f"  Word TF-IDF dim: {X_word_tr.shape[1]}", flush=True)

    word_results = []

    print("\n  Training mlp_baseline (word)...", flush=True)
    res = train_mlp(
        MLPBaseline(input_dim=X_word_tr.shape[1], hidden_dims=(256, 128), dropout=0.3),
        "word_mlp_baseline",
        X_word_tr, y_tr, X_word_val, y_val, X_word_te, y_te,
        epochs=args.epochs, batch_size=args.batch_size, device=device, out_dir=out_dir,
    )
    word_results.append({**res, "feature": "word_tfidf"})

    print("\n  Training mlp_deep (word)...", flush=True)
    res = train_mlp(
        MLPDeep(input_dim=X_word_tr.shape[1], hidden_dims=(512, 256, 128, 64), dropout=0.4),
        "word_mlp_deep",
        X_word_tr, y_tr, X_word_val, y_val, X_word_te, y_te,
        epochs=args.epochs, batch_size=args.batch_size, device=device, out_dir=out_dir,
    )
    word_results.append({**res, "feature": "word_tfidf"})

    # ── Save ──────────────────────────────────────────────────────────────── #
    all_results = char_results + word_results
    df = pd.DataFrame(all_results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_table.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 55)
    print("РЕЗУЛЬТАТЫ (Text modality)")
    print("=" * 55)
    print(df.to_string(index=False))
    print(f"\nСохранено: {out_path}", flush=True)


if __name__ == "__main__":
    main()
