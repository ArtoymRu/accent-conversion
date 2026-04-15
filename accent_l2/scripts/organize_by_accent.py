"""
Organize unpacked L2-ARCTIC speaker dirs into accent folders (arabic, hindi, ...).
Creates symlinks (or copies) under data/l2_arctic/<accent>/ pointing to speaker wav/ content.
"""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.accent_map import SPEAKER_TO_ACCENT


def main():
    parser = argparse.ArgumentParser(description="Organize L2-ARCTIC by accent")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data/l2_arctic (default: project_root/data/l2_arctic)",
    )
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink (slower, more portable)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else base / "data" / "l2_arctic"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    use_symlink = not args.copy
    if use_symlink and os.name == "nt":
        try:
            (data_dir / "arabic").symlink_to(data_dir / "ABA", target_is_directory=True)
        except OSError:
            use_symlink = False
            print("Symlinks not available, using copy.")

    for name in sorted(data_dir.iterdir()):
        if not name.is_dir():
            continue
        speaker = name.name.upper()
        if speaker not in SPEAKER_TO_ACCENT:
            continue
        accent = SPEAKER_TO_ACCENT[speaker]
        accent_dir = data_dir / accent
        accent_dir.mkdir(parents=True, exist_ok=True)
        # One subdir per speaker under accent (so paths stay unique)
        target = accent_dir / speaker
        if target.exists():
            continue
        if use_symlink:
            target.symlink_to(name.resolve(), target_is_directory=True)
        else:
            import shutil
            shutil.copytree(name, target)
        print(f"  {speaker} -> {accent}/")

    print("Done.")


if __name__ == "__main__":
    main()
