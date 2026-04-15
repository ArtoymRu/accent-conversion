"""
Compare multiple classifiers for accent classification on HuBERT features.

Methods compared:
  - Logistic Regression (L2)
  - SVM (RBF kernel)
  - Random Forest
  - Hist Gradient Boosting  (fast sklearn implementation)
  - K-Nearest Neighbours
  - MLP (PyTorch)

Two evaluation protocols are run:
  first_try/           — random utterance-level split (inflated; leaks speaker identity)
  speaker_independent/ — hold out 1 speaker per accent for test (proper generalization)

Features are cached to outputs/features/cache/ as .npz files to avoid
re-loading thousands of .pt files on repeated runs.

Usage:
    python src/analysis/compare_classifiers.py [--features_dir outputs/features] [--layers 6,9,12]
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.accent_map import ACCENTS, ACCENT_TO_SPEAKERS
from src.analysis.mlp_classifier import train_mlp, evaluate as evaluate_mlp


# ---------------------------------------------------------------------------
# Feature loading with cache (stores speaker codes alongside X, y)
# ---------------------------------------------------------------------------

def load_features(features_dir: Path, layer: int, cache_dir: Path | None = None):
    """
    Load mean-pooled HuBERT features + labels + speaker codes.
    Returns (X, y, speakers) where speakers is an array of speaker-code strings.
    Cache is invalidated if it's missing the 'speakers' key (old format).
    """
    if cache_dir is not None:
        cache_path = cache_dir / f"hubert_layer{layer}.npz"
        if cache_path.exists():
            d = np.load(cache_path, allow_pickle=True)
            if "speakers" in d:
                print(f"  [cache] loading layer {layer} from {cache_path}")
                X, y, speakers = d["X"], d["y"], d["speakers"]
                return X, y, speakers
            else:
                print(f"  [cache] old format detected, rebuilding cache for layer {layer}")
                del d  # release file handle before deleting on Windows
                try:
                    cache_path.unlink()
                except PermissionError:
                    cache_path = cache_dir / f"hubert_layer{layer}_v2.npz"
                    print(f"  [cache] could not delete old cache, writing to {cache_path}")

    root = features_dir / "hubert" / f"layer{layer}"
    if not root.exists():
        raise FileNotFoundError(f"Not found: {root}")

    from src.data.accent_map import SPEAKER_TO_ACCENT
    valid_speakers = set(SPEAKER_TO_ACCENT.keys())

    print(f"  Loading .pt files for layer {layer}...", flush=True)
    X_list, y_list, speaker_list = [], [], []
    pt_files = [p for p in sorted(root.glob("*.pt"))
                if p.stem.split("_")[0].upper() in valid_speakers]
    for pt_path in pt_files:
        speaker = pt_path.stem.split("_")[0].upper()
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        feat = data["features"]
        if feat.dim() == 3:
            feat = feat.squeeze(0)
        x = feat.mean(dim=0).numpy()
        accent_id = int(data["accent_id"])
        if accent_id < 0:
            continue
        X_list.append(x)
        y_list.append(accent_id)
        speaker_list.append(speaker)

    X = np.stack(X_list)
    y = np.array(y_list)
    speakers = np.array(speaker_list)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, speakers=speakers)
        print(f"  [cache] saved to {cache_path}", flush=True)

    return X, y, speakers


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------

def random_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Utterance-level stratified random split. Leaks speaker identity."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tv_idx, test_idx = next(sss.split(X, y))
    X_tv, y_tv = X[tv_idx], y[tv_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    val_frac = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    tr_idx, val_idx = next(sss2.split(X_tv, y_tv))
    return (X_tv[tr_idx], y_tv[tr_idx],
            X_tv[val_idx], y_tv[val_idx],
            X_test, y_test)


def speaker_independent_split(X, y, speakers, random_state=42):
    """
    Hold out exactly 1 speaker per accent for test; remaining 3 used for train+val.
    Val = 20% of train utterances (random stratified).
    The held-out speaker is chosen deterministically (last alphabetically per accent).
    """
    rng = np.random.default_rng(random_state)

    test_mask = np.zeros(len(y), dtype=bool)
    for accent_id, accent in enumerate(ACCENTS):
        spk_list = sorted(ACCENT_TO_SPEAKERS[accent])  # deterministic order
        held_out = spk_list[-1]                        # last alphabetically
        test_mask |= (speakers == held_out)

    train_val_mask = ~test_mask
    X_tv, y_tv = X[train_val_mask], y[train_val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    tr_idx, val_idx = next(sss.split(X_tv, y_tv))
    return (X_tv[tr_idx], y_tv[tr_idx],
            X_tv[val_idx], y_tv[val_idx],
            X_test, y_test)


# ---------------------------------------------------------------------------
# Sklearn classifier wrapper
# ---------------------------------------------------------------------------

def run_sklearn(clf, X_train, y_train, X_test, y_test, name):
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    per_class = f1_score(y_test, preds, average=None, labels=list(range(len(ACCENTS))))
    print(f"  {name:35s}  acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)", flush=True)
    return acc, f1, per_class, preds


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_comparison_bar(results_by_layer, out_path: Path, title_suffix=""):
    layers = sorted(results_by_layer.keys())
    methods = list(next(iter(results_by_layer.values())).keys())
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(layers)))
    for i, (layer, ax) in enumerate(zip(layers, axes)):
        accs = [results_by_layer[layer][m]["accuracy"] for m in methods]
        bars = ax.bar(x, accs, color=colors[i], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"HuBERT Layer {layer}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle(f"Accent Classification Accuracy — {title_suffix}", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_f1_heatmap(results_by_layer, out_path: Path, title_suffix=""):
    layers = sorted(results_by_layer.keys())
    methods = list(next(iter(results_by_layer.values())).keys())

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), len(methods) * 0.7 + 1))
    if len(layers) == 1:
        axes = [axes]

    for layer, ax in zip(layers, axes):
        data = np.array([results_by_layer[layer][m]["f1_per_class"] for m in methods])
        sns.heatmap(data, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=ACCENTS, yticklabels=methods,
                    vmin=0, vmax=1, ax=ax, cbar=True)
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Accent")

    plt.suptitle(f"Per-class F1 — {title_suffix}", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_accuracy_by_layer(results_by_layer, out_path: Path, title_suffix=""):
    layers = sorted(results_by_layer.keys())
    methods = list(next(iter(results_by_layer.values())).keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, method in enumerate(methods):
        accs = [results_by_layer[l][method]["accuracy"] for l in layers]
        ax.plot(layers, accs, marker=markers[i % len(markers)], label=method, linewidth=2)

    ax.set_xticks(layers)
    ax.set_xlabel("HuBERT Layer", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"Accuracy by HuBERT Layer — {title_suffix}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_protocol_comparison(first_try, speaker_indep, layers, out_path: Path):
    """Side-by-side bar chart comparing the two protocols for each method."""
    methods = list(next(iter(first_try.values())).keys())
    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, len(layers), figsize=(7 * len(layers), 5), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for layer, ax in zip(layers, axes):
        ft_accs = [first_try[layer][m]["accuracy"] for m in methods]
        si_accs = [speaker_indep[layer][m]["accuracy"] for m in methods]
        bars1 = ax.bar(x - width / 2, ft_accs, width, label="Random split (inflated)", color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x + width / 2, si_accs, width, label="Speaker-independent", color="#DD8452", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"Layer {layer}")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
        for bar, val in zip(list(bars1) + list(bars2), ft_accs + si_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle("Random split vs. Speaker-Independent evaluation", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Run one full evaluation pass
# ---------------------------------------------------------------------------

def run_evaluation(X_train, y_train, X_val, y_val, X_test, y_test,
                   sklearn_clfs, mlp_hidden, mlp_epochs, device, layer,
                   out_dir: Path, results_by_layer: dict):
    num_classes = len(ACCENTS)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results_by_layer[layer] = {}
    all_rows = []

    for name, clf in sklearn_clfs.items():
        import sklearn
        clf_fresh = sklearn.clone(clf)  # fresh clone so layers don't share state
        acc, f1, per_class, _ = run_sklearn(clf_fresh, X_train_sc, y_train, X_test_sc, y_test, name)
        results_by_layer[layer][name] = {"accuracy": acc, "f1_macro": f1, "f1_per_class": per_class}
        all_rows.append({"layer": layer, "method": name, "accuracy": acc, "f1_macro": f1})

    print(f"  {'MLP (512-256)':35s}", end="", flush=True)
    t0 = time.time()
    mlp_model, mlp_scaler, _ = train_mlp(
        X_train, y_train, X_val, y_val,
        hidden_dims=mlp_hidden,
        num_classes=num_classes,
        epochs=mlp_epochs,
        lr=1e-3,
        batch_size=256,
        dropout=0.3,
        device=device,
        patience=20,
    )
    acc, f1, preds, report, _ = evaluate_mlp(mlp_model, mlp_scaler, X_test, y_test, device=device)
    elapsed = time.time() - t0
    per_class = f1_score(y_test, preds, average=None, labels=list(range(num_classes)))
    print(f"  acc={acc:.4f}  f1={f1:.4f}  ({elapsed:.1f}s)", flush=True)
    results_by_layer[layer]["MLP (512-256)"] = {"accuracy": acc, "f1_macro": f1, "f1_per_class": per_class}
    all_rows.append({"layer": layer, "method": "MLP (512-256)", "accuracy": acc, "f1_macro": f1})

    with open(out_dir / f"layer{layer}_report.txt", "w") as f:
        f.write(f"HuBERT Layer {layer}\n{'='*60}\n")
        for method, res in results_by_layer[layer].items():
            f.write(f"{method:35s}  acc={res['accuracy']:.4f}  f1={res['f1_macro']:.4f}\n")

    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LEAKAGE_NOTE = """\
NOTE — Why 'first_try' results are inflated (~99%)
===================================================
The random split assigns utterances from the SAME speaker to both train and
test sets.  HuBERT features (especially lower layers) strongly encode speaker
identity (timbre, pitch, vocal-tract shape) — effectively a voice fingerprint.
With 4 speakers per accent, classifiers learn to recognise voices, not accents.

The 'speaker_independent' evaluation holds out one entire speaker per accent
(last alphabetically: ZHAA, TXHC, TNI, YKWK, NJS, TLV) for testing.
This forces the model to generalise to unseen voices — the actual task.
"""


def main():
    parser = argparse.ArgumentParser(description="Compare classifiers on HuBERT accent features")
    parser.add_argument("--features_dir", default="outputs/features")
    parser.add_argument("--layers", default="6,9,12")
    parser.add_argument("--mlp_epochs", type=int, default=150)
    parser.add_argument("--mlp_hidden", default="512,256")
    parser.add_argument("--output_dir", default="outputs/classification/comparison")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = base / features_dir
    base_out = Path(args.output_dir)
    if not base_out.is_absolute():
        base_out = base / base_out

    ft_dir = base_out / "first_try"
    si_dir = base_out / "speaker_independent"
    ft_dir.mkdir(parents=True, exist_ok=True)
    si_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = None if args.no_cache else (features_dir / "cache")
    layers = [int(x) for x in args.layers.split(",")]
    mlp_hidden = [int(x) for x in args.mlp_hidden.split(",")]

    sklearn_clfs = {
        "LogReg (L2)": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Hist Grad. Boost": HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1),
    }

    ft_all, si_all = [], []
    ft_by_layer, si_by_layer = {}, {}

    # Print held-out speakers info upfront
    print(LEAKAGE_NOTE)
    print("Held-out test speakers (speaker-independent eval):")
    for accent in ACCENTS:
        held_out = sorted(ACCENT_TO_SPEAKERS[accent])[-1]
        print(f"  {accent:12s}: {held_out}")
    print()

    for layer in layers:
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {layer}", flush=True)
        print(f"{'='*60}", flush=True)

        X, y, speakers = load_features(features_dir, layer, cache_dir)
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}", flush=True)

        # ---- first_try (random split) ----
        print(f"\n  [first_try] random utterance-level split", flush=True)
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = random_split(X, y)
        print(f"  train={len(X_tr)}, val={len(X_vl)}, test={len(X_te)}", flush=True)
        rows = run_evaluation(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                              sklearn_clfs, mlp_hidden, args.mlp_epochs,
                              args.device, layer, ft_dir, ft_by_layer)
        ft_all.extend(rows)

        # ---- speaker_independent ----
        print(f"\n  [speaker_independent] held-out speaker per accent", flush=True)
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = speaker_independent_split(X, y, speakers)
        print(f"  train={len(X_tr)}, val={len(X_vl)}, test={len(X_te)}", flush=True)
        rows = run_evaluation(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                              sklearn_clfs, mlp_hidden, args.mlp_epochs,
                              args.device, layer, si_dir, si_by_layer)
        si_all.extend(rows)

    # Save CSVs
    for rows, out_dir, label in [(ft_all, ft_dir, "first_try"), (si_all, si_dir, "speaker_independent")]:
        csv_path = out_dir / "comparison.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["layer", "method", "accuracy", "f1_macro"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {csv_path}")

    # Save leakage note
    with open(base_out / "README.txt", "w") as f:
        f.write(LEAKAGE_NOTE)

    # Plots
    print("\nGenerating plots...", flush=True)
    for rbyl, out_dir, label in [(ft_by_layer, ft_dir, "Random split (inflated)"),
                                  (si_by_layer, si_dir, "Speaker-independent")]:
        plot_comparison_bar(rbyl, out_dir / "accuracy_by_method.png", label)
        plot_f1_heatmap(rbyl, out_dir / "f1_per_class_heatmap.png", label)
        if len(layers) > 1:
            plot_accuracy_by_layer(rbyl, out_dir / "accuracy_by_layer.png", label)

    plot_protocol_comparison(ft_by_layer, si_by_layer, layers,
                             base_out / "protocol_comparison.png")

    # Summary tables
    for rows, label in [(ft_all, "FIRST TRY (random split — inflated)"),
                         (si_all, "SPEAKER-INDEPENDENT (proper evaluation)")]:
        print(f"\n{'='*72}")
        print(f"{label}")
        print(f"{'Layer':<8}{'Method':<35}{'Accuracy':>10}{'F1-macro':>12}")
        print("-" * 72)
        for row in rows:
            print(f"{row['layer']:<8}{row['method']:<35}{float(row['accuracy']):>10.4f}{float(row['f1_macro']):>12.4f}")
        print("=" * 72)

    print(f"\nAll outputs saved under {base_out}")


if __name__ == "__main__":
    main()
