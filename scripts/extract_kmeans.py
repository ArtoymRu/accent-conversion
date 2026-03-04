"""Fit k-means on HuBERT layer-6 features extracted from a directory of .wav files.

Usage
-----
python scripts/extract_kmeans.py --data_dir /path/to/l2arctic --max_files 500
"""
import sys
import argparse
import numpy as np
import librosa
import joblib
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.content import ContentEncoder
from config import cfg


def main():
    parser = argparse.ArgumentParser(description="Train k-means on HuBERT features")
    parser.add_argument("--data_dir",  required=True, help="Root directory with .wav files")
    parser.add_argument("--out",       default=cfg.kmeans_path, help="Output .pkl path")
    parser.add_argument("--max_files", type=int, default=None,  help="Cap number of files")
    parser.add_argument("--k",         type=int, default=cfg.kmeans_k, help="Number of clusters")
    args = parser.parse_args()

    encoder = ContentEncoder(kmeans_path=None)   # don't load k-means yet

    wav_files = sorted(Path(args.data_dir).rglob("*.wav"))
    if args.max_files:
        wav_files = wav_files[: args.max_files]
    print(f"Found {len(wav_files)} .wav files")

    all_features = []
    for f in tqdm(wav_files, desc="Extracting HuBERT features"):
        try:
            audio, sr = librosa.load(f, sr=16000, mono=True)
            feats = encoder.extract_features(audio, sr=16000)   # (T, dim)
            all_features.append(feats)
        except Exception as e:
            print(f"  Skip {f.name}: {e}")

    if not all_features:
        raise RuntimeError("No features collected — check --data_dir.")

    X = np.concatenate(all_features, axis=0)
    print(f"Feature matrix: {X.shape[0]} frames × {X.shape[1]} dims")

    print(f"Fitting MiniBatchKMeans K={args.k}…")
    km = MiniBatchKMeans(
        n_clusters=args.k,
        batch_size=10_000,
        n_init=10,
        random_state=42,
        verbose=0,
    )
    km.fit(X)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, out_path)
    print(f"Saved k-means → {out_path}")

    # Quick sanity check: cluster distribution
    labels = km.predict(X[:5000])
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Active clusters in sample: {len(unique)}/{args.k}")
    print(f"  Top-5 cluster counts: {sorted(counts, reverse=True)[:5]}")


if __name__ == "__main__":
    main()
