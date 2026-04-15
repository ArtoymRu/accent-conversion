"""
Plot heatmap of feature specialization (features x accents).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


def main():
    parser = argparse.ArgumentParser(description="Plot feature specialization heatmap")
    parser.add_argument("--heatmap_npy", type=str, default="outputs/figures/feature_specialization/heatmap_layer12.npy")
    parser.add_argument("--accents_txt", type=str, default="outputs/figures/feature_specialization/accents_layer12.txt")
    parser.add_argument("--output", type=str, default="outputs/figures/heatmap_specialization.png")
    parser.add_argument("--layer", type=int, default=12, help="Layer label for title")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    heatmap_path = Path(args.heatmap_npy)
    accents_path = Path(args.accents_txt)
    if not heatmap_path.is_absolute():
        heatmap_path = base / heatmap_path
    if not accents_path.is_absolute():
        accents_path = base / accents_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = base / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = np.load(heatmap_path)
    if accents_path.exists():
        with open(accents_path, encoding="utf-8") as f:
            accents = [l.strip() for l in f if l.strip()]
    else:
        accents = [f"acc_{i}" for i in range(heatmap.shape[1])]

    # Optionally subsample features for readability (e.g. every 10th or top varying)
    n_feat, n_acc = heatmap.shape
    if n_feat > 500:
        step = max(1, n_feat // 500)
        heatmap = heatmap[::step]
        feat_labels = [f"f{i*step}" for i in range(heatmap.shape[0])]
    else:
        feat_labels = [f"f{i}" for i in range(n_feat)]

    fig, ax = plt.subplots(figsize=(max(8, len(accents) * 1.2), max(6, heatmap.shape[0] * 0.02)))
    sns.heatmap(
        heatmap,
        xticklabels=accents,
        yticklabels=feat_labels if len(feat_labels) <= 80 else False,
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": "Activation frequency"},
    )
    ax.set_xlabel("Accent")
    ax.set_ylabel("SAE feature")
    ax.set_title(f"Feature specialization (HuBERT layer {args.layer})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
