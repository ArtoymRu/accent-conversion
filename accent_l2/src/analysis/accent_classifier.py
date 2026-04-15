"""
Train a linear classifier (logistic regression) on SAE sparse representations for accent classification.
Reports Accuracy and F1-score per layer.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features_for_layer(features_dir: Path, layer: int):
    """Load all samples from layer dir: X (n_samples, n_features), y (accent_id)."""
    root = features_dir / "hubert" / f"layer{layer}"
    if not root.exists():
        return None, None
    X_list, y_list = [], []
    for pt_path in root.glob("*.pt"):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        feat = data["features"]
        if feat.dim() == 3:
            feat = feat.squeeze(0)
        # (T, D) -> flatten or mean pool for classifier
        x = feat.mean(dim=0).numpy()
        X_list.append(x)
        y_list.append(data["accent_id"])
    if not X_list:
        return None, None
    return np.stack(X_list), np.array(y_list)


def main():
    parser = argparse.ArgumentParser(description="Accent classification on SAE features")
    parser.add_argument("--features_dir", type=str, default="outputs/features")
    parser.add_argument("--layers", type=str, default="6,9,12", help="Comma-separated layer indices")
    parser.add_argument("--output", type=str, default="outputs/tables/accent_classification.csv")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2]
    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = base / features_dir
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = base / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]
    clf = LogisticRegression(max_iter=1000, random_state=42)
    results = []
    for layer in layers:
        X, y = load_features_for_layer(features_dir, layer)
        if X is None or len(y) < 10:
            results.append({"layer": layer, "accuracy": np.nan, "f1_macro": np.nan})
            continue
        # Filter valid labels
        valid = y >= 0
        X, y = X[valid], y[valid]
        scores = cross_validate(clf, X, y, cv=args.cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        acc = scores["test_accuracy"].mean()
        f1 = scores["test_f1_macro"].mean()
        results.append({"layer": layer, "accuracy": acc, "f1_macro": f1})
        print(f"Layer {layer}: Accuracy={acc:.4f}, F1_macro={f1:.4f}")

    # Save table
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "accuracy", "f1_macro"])
        w.writeheader()
        w.writerows(results)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
