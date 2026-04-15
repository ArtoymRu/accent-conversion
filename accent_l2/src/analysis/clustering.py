"""
Clustering & separability analysis on HuBERT features.

Answers the question: "Are accent groups actually separable in feature space,
or do speaker-identity effects dominate?"

Steps:
  1. Load cached mean-pooled HuBERT features (layers 6, 9, 12)
  2. PCA: variance explained, 2D scatter coloured by accent / speaker
  3. t-SNE 2D scatter (subsample for speed) coloured by accent / speaker
  4. UMAP 2D scatter coloured by accent / speaker
  5. K-Means (k=6 and k=24) — compare cluster assignment vs. accent & speaker
  6. Metrics: ARI, NMI, Silhouette (accent), Silhouette (speaker)
  7. Save a text report + all figures

Usage:
    python src/analysis/clustering.py [--layer 6] [--n_samples 8000]
"""

import argparse
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    silhouette_samples,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.accent_map import ACCENTS, ACCENT_TO_SPEAKERS

try:
    import umap as umap_module
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

ACCENT_PALETTE = {
    accent: color for accent, color in zip(
        ACCENTS,
        ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#8338EC"]
    )
}

# 24 distinct colours for 24 speakers (tab20 + tab20b extensions)
_cmap24 = (list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors))[:24]


def accent_legend(ax):
    patches = [mpatches.Patch(color=ACCENT_PALETTE[a], label=a) for a in ACCENTS]
    ax.legend(handles=patches, fontsize=8, loc="best", framealpha=0.7,
              title="Accent", title_fontsize=8)


def speaker_legend(ax, speakers_sorted, colors):
    patches = [mpatches.Patch(color=colors[i], label=s)
               for i, s in enumerate(speakers_sorted)]
    ax.legend(handles=patches, fontsize=6, loc="best", framealpha=0.7,
              title="Speaker", title_fontsize=7, ncol=2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cached(features_dir: Path, layer: int):
    cache_path = features_dir / "cache" / f"hubert_layer{layer}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_path}\n"
            "Run compare_classifiers.py first to build the cache."
        )
    d = np.load(cache_path, allow_pickle=True)
    X, y = d["X"], d["y"]
    speakers = d["speakers"] if "speakers" in d else None
    if speakers is None:
        raise RuntimeError("Cache missing 'speakers' key. Re-run compare_classifiers.py.")
    return X, y, speakers


def subsample(X, y, speakers, n_per_class: int, rng):
    """Stratified subsample: at most n_per_class utterances per accent."""
    idx = []
    for accent_id in range(len(ACCENTS)):
        mask = np.where(y == accent_id)[0]
        chosen = rng.choice(mask, size=min(n_per_class, len(mask)), replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx], speakers[idx]


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def run_pca(X_scaled, n_components=50):
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
    emb = pca.fit_transform(X_scaled)
    return emb, pca


def run_tsne(X_pca, perplexity=40, n_iter=1000):
    t0 = time.time()
    import sklearn
    from packaging.version import Version
    tsne_kwargs = dict(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
    # 'n_iter' was renamed to 'max_iter' in sklearn 1.5
    if Version(sklearn.__version__) >= Version("1.5"):
        tsne_kwargs["max_iter"] = n_iter
    else:
        tsne_kwargs["n_iter"] = n_iter
    tsne = TSNE(**tsne_kwargs)
    emb = tsne.fit_transform(X_pca[:, :50])  # use top-50 PCs as input
    print(f"    t-SNE done in {time.time()-t0:.1f}s", flush=True)
    return emb


def run_umap(X_pca, n_neighbors=30, min_dist=0.1):
    t0 = time.time()
    reducer = umap_module.UMAP(n_components=2, n_neighbors=n_neighbors,
                               min_dist=min_dist, random_state=42)
    emb = reducer.fit_transform(X_pca[:, :50])
    print(f"    UMAP done in {time.time()-t0:.1f}s", flush=True)
    return emb


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def scatter_2d(ax, emb, labels, palette_or_colors, label_names,
               title="", alpha=0.35, s=6):
    """Generic 2-D scatter: labels are integer indices into palette_or_colors/label_names."""
    for i, name in enumerate(label_names):
        mask = labels == i
        if isinstance(palette_or_colors, dict):
            color = palette_or_colors[name]
        else:
            color = palette_or_colors[i]
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[color], label=name, alpha=alpha, s=s, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def save_scatter_pair(emb, y, speaker_ids, speakers_sorted, out_path,
                      method_name="", layer=6):
    """Two side-by-side scatters: left=accent, right=speaker."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter_2d(ax1, emb, y, ACCENT_PALETTE, ACCENTS,
               title=f"{method_name} — coloured by accent (layer {layer})")
    accent_legend(ax1)

    scatter_2d(ax2, emb, speaker_ids, _cmap24, speakers_sorted,
               title=f"{method_name} — coloured by speaker (layer {layer})")
    speaker_legend(ax2, speakers_sorted, _cmap24)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# PCA-specific plots
# ---------------------------------------------------------------------------

def plot_pca_variance(pca, out_path: Path, layer: int):
    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)
    n = min(50, len(var))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(range(1, n + 1), var[:n] * 100, color="#457B9D", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title(f"Per-component variance (layer {layer})")

    ax2.plot(range(1, n + 1), cumvar[:n] * 100, color="#E63946", linewidth=2)
    ax2.axhline(80, linestyle="--", color="grey", linewidth=1, label="80%")
    ax2.axhline(90, linestyle=":", color="grey", linewidth=1, label="90%")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative explained variance (%)")
    ax2.set_title(f"Cumulative variance (layer {layer})")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 101)

    # Annotate how many PCs for 80% / 90%
    for thr in [0.80, 0.90]:
        k = int(np.searchsorted(cumvar, thr)) + 1
        ax2.annotate(f"{k} PCs → {thr*100:.0f}%",
                     xy=(k, thr * 100), xytext=(k + 2, thr * 100 - 6),
                     fontsize=8, color="grey",
                     arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_pca_3d(emb_pca, y, out_path: Path, layer: int):
    """3-D PCA scatter (PC1-2-3) coloured by accent."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i, accent in enumerate(ACCENTS):
        mask = y == i
        ax.scatter(emb_pca[mask, 0], emb_pca[mask, 1], emb_pca[mask, 2],
                   c=[ACCENT_PALETTE[accent]], label=accent, alpha=0.3, s=4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"PCA (PC1–PC3) — accent (layer {layer})")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Silhouette plot
# ---------------------------------------------------------------------------

def plot_silhouette(X_scaled, y, speakers, out_path: Path, layer: int,
                    n_max=5000, rng=None):
    """Silhouette scores per sample, grouped by accent (top) and by speaker (bottom)."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Subsample for speed
    if len(y) > n_max:
        idx = rng.choice(len(y), size=n_max, replace=False)
        X_s, y_s, spk_s = X_scaled[idx], y[idx], speakers[idx]
    else:
        X_s, y_s, spk_s = X_scaled, y, speakers

    # Encode speakers to int
    le = LabelEncoder()
    spk_int = le.fit_transform(spk_s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, labels, names, title, palette in [
        (axes[0], y_s, ACCENTS, f"Silhouette by accent (layer {layer})",
         [ACCENT_PALETTE[a] for a in ACCENTS]),
        (axes[1], spk_int, le.classes_,
         f"Silhouette by speaker (layer {layer})", _cmap24),
    ]:
        sil_vals = silhouette_samples(X_s, labels, metric="euclidean")
        y_lower = 10
        for i in range(len(names)):
            vals = sil_vals[labels == i]
            vals.sort()
            size = len(vals)
            y_upper = y_lower + size
            color = palette[i] if not isinstance(palette[i], str) else palette[i]
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                             facecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + size / 2, names[i], fontsize=6, va="center")
            y_lower = y_upper + 5
        mean_sil = sil_vals.mean()
        ax.axvline(mean_sil, color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_sil:.3f}")
        ax.set_xlabel("Silhouette coefficient")
        ax.set_title(title)
        ax.set_yticks([])
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_and_score(X_scaled, y, speakers, layer, report_lines):
    """Run KMeans k=6 and k=24, score against accent & speaker labels."""
    le = LabelEncoder()
    spk_int = le.fit_transform(speakers)

    results = []
    for k, label_str in [(6, "accent"), (24, "speaker")]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(X_scaled)

        ari_acc = adjusted_rand_score(y, clusters)
        nmi_acc = normalized_mutual_info_score(y, clusters)
        ari_spk = adjusted_rand_score(spk_int, clusters)
        nmi_spk = normalized_mutual_info_score(spk_int, clusters)

        results.append({
            "k": k, "init_label": label_str,
            "ARI_accent": ari_acc, "NMI_accent": nmi_acc,
            "ARI_speaker": ari_spk, "NMI_speaker": nmi_spk,
        })
        line = (f"  KMeans k={k:2d}  |  vs accent: ARI={ari_acc:.3f} NMI={nmi_acc:.3f}"
                f"  |  vs speaker: ARI={ari_spk:.3f} NMI={nmi_spk:.3f}")
        print(line, flush=True)
        report_lines.append(line)

    return results


def plot_cluster_heatmap(X_scaled, y, speakers, out_path: Path, layer: int):
    """Heatmap: for each KMeans-6 cluster × accent, how many samples."""
    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    clusters = km.fit_predict(X_scaled)

    mat = np.zeros((6, len(ACCENTS)), dtype=int)
    for c, acc in zip(clusters, y):
        mat[c, acc] += 1

    # Sort clusters by dominant accent for readability
    dominant = mat.argmax(axis=1)
    order = np.argsort(dominant)
    mat = mat[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=ACCENTS,
                yticklabels=[f"Cluster {i}" for i in range(6)],
                ax=ax)
    ax.set_xlabel("True accent")
    ax.set_ylabel("KMeans cluster (k=6)")
    ax.set_title(f"KMeans k=6 cluster composition (layer {layer})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clustering & separability analysis on HuBERT features")
    parser.add_argument("--features_dir", default="outputs/features")
    parser.add_argument("--layers", default="6,9,12", help="Comma-separated HuBERT layers to analyse")
    parser.add_argument("--n_samples", type=int, default=8000,
                        help="Max samples per accent for t-SNE/UMAP (use 0 for all)")
    parser.add_argument("--output_dir", default="outputs/clustering")
    parser.add_argument("--tsne_perplexity", type=float, default=40)
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = base / features_dir
    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = base / out_base
    out_base.mkdir(parents=True, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]
    rng = np.random.default_rng(42)

    all_report_lines = []
    all_report_lines.append("=" * 70)
    all_report_lines.append("CLUSTERING & SEPARABILITY REPORT — HuBERT features on L2-ARCTIC")
    all_report_lines.append("=" * 70)
    all_report_lines.append(
        "\nDataset: 26,867 utterances | 6 accents | 24 speakers (4 per accent)"
        "\nFeatures: mean-pooled HuBERT hidden states (dim=768)"
        "\nSplit protocol: full dataset (no train/test split for this analysis)\n"
    )
    all_report_lines.append(
        "KEY QUESTION: Do HuBERT features cluster by accent or by speaker?\n"
        "High ARI/NMI vs. speaker + low ARI/NMI vs. accent => features encode\n"
        "speaker identity more than accent => explains ~99% classification scores.\n"
    )

    for layer in layers:
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {layer}", flush=True)
        print(f"{'='*60}", flush=True)

        out_dir = out_base / f"layer{layer}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load
        X, y, speakers = load_cached(features_dir, layer)
        print(f"  Loaded {len(X)} samples, {X.shape[1]} features", flush=True)

        # Speaker encoding
        all_speakers_sorted = sorted(set(speakers.tolist()))
        spk_le = LabelEncoder()
        spk_le.fit(all_speakers_sorted)
        speaker_ids = spk_le.transform(speakers)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ---- PCA ----
        print("  Running PCA...", flush=True)
        emb_pca, pca = run_pca(X_scaled, n_components=50)

        plot_pca_variance(pca, out_dir / "pca_variance.png", layer)

        # Subsample for scatter plots
        n_per = args.n_samples if args.n_samples > 0 else 99999
        X_sub, y_sub, spk_sub = subsample(X_scaled, y, speakers, n_per, rng)
        spk_sub_ids = spk_le.transform(spk_sub)
        pca_sub = PCA(n_components=50, random_state=42).fit_transform(X_sub)

        save_scatter_pair(pca_sub[:, :2], y_sub, spk_sub_ids, all_speakers_sorted,
                          out_dir / "pca_scatter.png", "PCA (PC1–PC2)", layer)

        plot_pca_3d(pca_sub, y_sub, out_dir / "pca_3d.png", layer)

        # ---- Silhouette (on full scaled data, subsampled) ----
        print("  Computing silhouette scores...", flush=True)
        plot_silhouette(X_scaled, y, speakers, out_dir / "silhouette.png",
                        layer, n_max=5000, rng=rng)

        sil_accent = silhouette_score(X_sub, y_sub, metric="euclidean")
        sil_speaker = silhouette_score(X_sub, spk_sub_ids, metric="euclidean")
        print(f"  Silhouette (accent) = {sil_accent:.4f}", flush=True)
        print(f"  Silhouette (speaker)= {sil_speaker:.4f}", flush=True)

        # ---- t-SNE ----
        print("  Running t-SNE...", flush=True)
        emb_tsne = run_tsne(pca_sub, perplexity=args.tsne_perplexity)
        save_scatter_pair(emb_tsne, y_sub, spk_sub_ids, all_speakers_sorted,
                          out_dir / "tsne_scatter.png", "t-SNE", layer)

        # ---- UMAP ----
        if HAS_UMAP:
            print("  Running UMAP...", flush=True)
            emb_umap = run_umap(pca_sub)
            save_scatter_pair(emb_umap, y_sub, spk_sub_ids, all_speakers_sorted,
                              out_dir / "umap_scatter.png", "UMAP", layer)
        else:
            print("  UMAP not installed, skipping.", flush=True)

        # ---- KMeans clustering ----
        print("  Running KMeans clustering...", flush=True)
        report_lines = [f"\nLayer {layer}",
                        f"  Silhouette (accent label): {sil_accent:.4f}",
                        f"  Silhouette (speaker label): {sil_speaker:.4f}",
                        f"  {'>'*50}",
                        f"  Silhouette interpretation: 1=perfect, 0=overlapping, <0=wrong cluster",
                        f"  Higher sil(speaker) vs sil(accent) => features encode speaker > accent",
                        ""]
        cluster_results = cluster_and_score(X_scaled, y, speakers, layer, report_lines)
        plot_cluster_heatmap(X_scaled, y, speakers, out_dir / "kmeans_heatmap.png", layer)

        all_report_lines.extend(report_lines)

        # Per-accent sample count
        all_report_lines.append("\n  Samples per accent:")
        for accent_id, cnt in sorted(Counter(y.tolist()).items()):
            all_report_lines.append(f"    {ACCENTS[accent_id]:12s}: {cnt}")

        print(f"  Outputs in {out_dir}", flush=True)

    # Save text report
    report_path = out_base / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_report_lines) + "\n")
    print(f"\nReport saved to {report_path}", flush=True)

    # Print report to stdout as well
    print("\n" + "\n".join(all_report_lines))


if __name__ == "__main__":
    main()
