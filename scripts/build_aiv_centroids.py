"""Compute per-accent AIV centroids and save to disk.

Expected directory layout
--------------------------
data_dir/
  hindi/        ← accent name = directory name
    utt001.wav
    utt002.wav
    ...
  mandarin/
    ...

Usage
-----
python scripts/build_aiv_centroids.py --data_dir /path/to/accented_data
python scripts/build_aiv_centroids.py --data_dir /path/to/data --out checkpoints/aiv_centroids.npy
"""
import sys
import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.accent import AccentFeatureExtractor
from config import cfg


def main():
    parser = argparse.ArgumentParser(description="Build per-accent AIV centroids")
    parser.add_argument("--data_dir", required=True,
                        help="Root dir whose sub-dirs are accent groups")
    parser.add_argument("--out", default=cfg.centroids_path, help="Output .npy path")
    parser.add_argument("--min_files", type=int, default=1,
                        help="Skip accent groups with fewer valid files")
    args = parser.parse_args()

    extractor = AccentFeatureExtractor()
    centroids: dict[str, np.ndarray] = {}

    data_dir = Path(args.data_dir)
    accent_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not accent_dirs:
        raise RuntimeError(f"No sub-directories found in {data_dir}")

    for accent_dir in accent_dirs:
        accent_name = accent_dir.name
        wav_files   = list(accent_dir.rglob("*.wav"))
        print(f"\n── {accent_name}: {len(wav_files)} files")

        vecs = []
        for f in tqdm(wav_files, desc=accent_name, leave=False):
            try:
                audio, sr = librosa.load(f, sr=16000, mono=True)
                feat = extractor.extract(audio, sr=16000)
                if feat is not None:
                    vecs.append(feat)
            except Exception as e:
                print(f"  Skip {f.name}: {e}")

        if len(vecs) < args.min_files:
            print(f"  → skipped ({len(vecs)} valid utterances < min {args.min_files})")
            continue

        centroid = np.mean(vecs, axis=0).astype(np.float32)
        centroids[accent_name] = centroid
        print(f"  → centroid from {len(vecs)} utterances  shape={centroid.shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, centroids)
    print(f"\nSaved {len(centroids)} centroids → {out_path}")
    print("Keys:", list(centroids.keys()))


if __name__ == "__main__":
    main()
