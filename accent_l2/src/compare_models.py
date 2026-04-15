"""
Compare models (HuBERT layers 6,9,12, EnCodec, WavLM): table with MSE, L0, accent classification accuracy.
Reads from outputs/features and outputs/checkpoints; writes outputs/tables/comparison.csv and optional figures.
"""

import argparse
import csv
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare models for report Table 1")
    parser.add_argument("--features_dir", type=str, default="outputs/features")
    parser.add_argument("--checkpoints_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--output", type=str, default="outputs/tables/comparison.csv")
    parser.add_argument("--classification_csv", type=str, default="outputs/tables/accent_classification.csv")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = base / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Rows: HuBERT layer 6, 9, 12, EnCodec, WavLM
    rows = [
        {"model": "HuBERT", "layer": "6", "mse": "", "l0": "", "accuracy": "", "f1": ""},
        {"model": "HuBERT", "layer": "9", "mse": "", "l0": "", "accuracy": "", "f1": ""},
        {"model": "HuBERT", "layer": "12", "mse": "", "l0": "", "accuracy": "", "f1": ""},
        {"model": "EnCodec", "layer": "-", "mse": "", "l0": "", "accuracy": "", "f1": ""},
        {"model": "WavLM", "layer": "-", "mse": "", "l0": "", "accuracy": "", "f1": ""},
    ]

    # Fill classification from accent_classifier output if present
    cls_path = Path(args.classification_csv)
    if not cls_path.is_absolute():
        cls_path = base / cls_path
    if cls_path.exists():
        with open(cls_path, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                layer = row.get("layer", "")
                for r0 in rows:
                    if r0["model"] == "HuBERT" and str(r0["layer"]) == str(layer):
                        r0["accuracy"] = row.get("accuracy", "")
                        r0["f1"] = row.get("f1_macro", "")
                        break

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer", "mse", "l0", "accuracy", "f1"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved comparison table to {out_path}. Fill MSE and L0 from SAE training logs when available.")


if __name__ == "__main__":
    main()
