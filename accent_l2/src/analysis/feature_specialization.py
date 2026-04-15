"""
Compute per-accent activation frequency for each SAE feature.
Output: structure for heatmap (features x accents).
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features_dir(features_dir: Path, layer: int = None):
    """Load all .pt files from features_dir (or layer subdir). Yield (features, accent_id, path)."""
    if layer is not None:
        root = features_dir / f"layer{layer}"
    else:
        root = features_dir
    if not root.exists():
        return
    for pt_path in root.glob("*.pt"):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        feat = data["features"]  # (1, T, D) or (T, D)
        if feat.dim() == 3:
            feat = feat.squeeze(0)
        accent_id = data.get("accent_id", -1)
        yield feat.numpy(), accent_id, str(pt_path)


def compute_activation_frequency(features_dir: Path, layer: int, threshold: float = 0.0):
    """
    For each feature dimension, compute mean activation count per accent (normalized by samples).
    Returns: (n_features, n_accents) array, accent_ids order, feature_idx.
    """
    from src.data.accent_map import ACCENTS

    n_accents = len(ACCENTS)
    # Collect per-accent: list of (n_features,) mean activation per sample
    per_accent = defaultdict(list)
    for feat, accent_id, _ in load_features_dir(features_dir, layer):
        # feat: (T, D), we want per-feature activation (e.g. mean over time or count above threshold)
        if threshold > 0:
            active = (np.abs(feat) > threshold).astype(np.float32)
        else:
            active = np.abs(feat)
        mean_per_feature = active.mean(axis=0)  # (D,)
        if 0 <= accent_id < n_accents:
            per_accent[accent_id].append(mean_per_feature)

    if not per_accent:
        return None, ACCENTS, None

    n_features = next(iter(per_accent.values()))[0].shape[0]
    heatmap = np.zeros((n_features, n_accents), dtype=np.float64)
    for acc_id in range(n_accents):
        if acc_id not in per_accent:
            continue
        arr = np.stack(per_accent[acc_id], axis=0)
        heatmap[:, acc_id] = arr.mean(axis=0)
    return heatmap, ACCENTS, list(range(n_features))


def main():
    parser = argparse.ArgumentParser(description="Compute feature specialization (activation frequency per accent)")
    parser.add_argument("--features_dir", type=str, default="outputs/features/hubert", help="Path to extracted features")
    parser.add_argument("--layer", type=int, default=12, help="Layer index")
    parser.add_argument("--output", type=str, default="outputs/figures/feature_specialization", help="Output dir for .npy")
    parser.add_argument("--threshold", type=float, default=0.0, help="Activation threshold (0 = use mean abs)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = base / features_dir
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap, accents, feature_idx = compute_activation_frequency(features_dir, args.layer, args.threshold)
    if heatmap is None:
        print("No features found.")
        return
    np.save(out_dir / f"heatmap_layer{args.layer}.npy", heatmap)
    with open(out_dir / f"accents_layer{args.layer}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(accents))
    print(f"Saved heatmap shape {heatmap.shape} to {out_dir}")


if __name__ == "__main__":
    main()
