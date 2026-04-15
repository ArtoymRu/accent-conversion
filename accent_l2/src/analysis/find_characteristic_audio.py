"""
Find audio segments that maximally activate a given "accent" feature index.
Output: list of (wav_path, segment_info, activation_score) for qualitative listening.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features_dir(features_dir: Path, layer: int):
    """Yield (features, accent_id, wav_path) for each .pt file."""
    root = features_dir / "hubert" / f"layer{layer}"
    if not root.exists():
        return
    for pt_path in root.glob("*.pt"):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        feat = data["features"]
        if feat.dim() == 3:
            feat = feat.squeeze(0)
        wav_path = data.get("wav_path", str(pt_path))
        yield feat.numpy(), data.get("accent_id", -1), wav_path


def main():
    parser = argparse.ArgumentParser(description="Find audio that maximally activates a feature")
    parser.add_argument("--features_dir", type=str, default="outputs/features")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--feature_idx", type=int, default=0, help="SAE feature index to inspect")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top segments to list")
    parser.add_argument("--output", type=str, default="outputs/figures/top_activations.txt")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = base / features_dir
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = base / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scores = []
    for feat, accent_id, wav_path in load_features_dir(features_dir, args.layer):
        if feat.shape[1] <= args.feature_idx:
            continue
        act = np.abs(feat[:, args.feature_idx]).mean()
        scores.append((act, wav_path, accent_id))
    scores.sort(key=lambda x: -x[0])
    top = scores[: args.top_k]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Feature index: {args.feature_idx} (layer {args.layer})\n")
        f.write(f"Top {args.top_k} segments by mean activation\n\n")
        for i, (act, path, acc_id) in enumerate(top, 1):
            f.write(f"{i}. {act:.6f}  {path}  accent_id={acc_id}\n")
    print(f"Saved top activations to {out_path}")


if __name__ == "__main__":
    main()
