"""
Unpack zip archives from data/l2_arctic into the same directory.
Archives are expected to be already present (no download).
"""

import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Unpack L2-ARCTIC zip archives")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data/l2_arctic (default: project_root/data/l2_arctic)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else base / "data" / "l2_arctic"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    zips = list(data_dir.glob("*.zip"))
    if not zips:
        print(f"No .zip files found in {data_dir}. Place L2-ARCTIC zip archives there.")
        return

    for zpath in zips:
        print(f"Unpacking {zpath.name} ...")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(data_dir)
    print("Done.")


if __name__ == "__main__":
    main()
