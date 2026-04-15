"""
Train Sparse Autoencoders (SAE) on extracted features.
Designed to plug in AudioSAE when available (third_party/audiosae).
Until then, saves config and placeholder for checkpoints.
"""

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_sae_placeholder(features_dir: Path, checkpoints_dir: Path, config: dict) -> None:
    """
    Placeholder: create checkpoint dir and save training config.
    Replace this with actual AudioSAE training when third_party/audiosae is available.
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoints_dir / "sae_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Config saved to {config_path}. AudioSAE training not run (code not integrated).")


def main():
    parser = argparse.ArgumentParser(description="Train SAE on extracted features")
    parser.add_argument("--config", type=str, default="configs/sae.yaml", help="Path to SAE config")
    args = parser.parse_args()

    config = load_config(args.config)
    base = Path(__file__).resolve().parents[1]
    features_dir = Path(config["features_dir"])
    checkpoints_dir = Path(config["checkpoints_dir"])
    if not features_dir.is_absolute():
        features_dir = base / features_dir
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = base / checkpoints_dir

    # Try to use AudioSAE if available
    audiosae_path = base / "third_party" / "audiosae"
    if audiosae_path.is_dir():
        try:
            # When AudioSAE is present, import and run their training loop with our config
            # import sys
            # sys.path.insert(0, str(audiosae_path))
            # from train import main as audiosae_train
            # audiosae_train(config)
            pass
        except Exception as e:
            print(f"AudioSAE integration failed: {e}. Using placeholder.")
            train_sae_placeholder(features_dir, checkpoints_dir, config)
    else:
        train_sae_placeholder(features_dir, checkpoints_dir, config)


if __name__ == "__main__":
    main()
