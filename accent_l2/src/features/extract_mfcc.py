"""
Извлечение MFCC-признаков для всех высказываний L2-ARCTIC.

Признаки (39-мерный вектор на фрейм):
  - MFCC 1-13 (13 коэффициентов)
  - delta-MFCC (13)
  - delta2-MFCC (13)

Для каждого WAV файла сохраняется .pt файл с ключами:
  'features': Tensor(T, 39) — последовательность фреймов
  'mean':     Tensor(39,)   — mean-pooled вектор
  'speaker':  str
  'accent':   str

Аргументы:
  --data_root   путь к data/l2_arctic
  --out_dir     outputs/features/mfcc
  --n_mfcc      число MFCC коэффициентов (default: 13)
  --sr          целевой sample rate (default: 16000)

Пример:
  python src/features/extract_mfcc.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

try:
    import librosa
except ImportError:
    raise ImportError("pip install librosa")

from src.data.accent_map import SPEAKER_TO_ACCENT

ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]


def find_wav_files(data_root: Path):
    """Находит все WAV файлы в структуре ACCENT/SPEAKER/wav/*.wav."""
    entries = []
    for accent_dir in sorted(data_root.iterdir()):
        if not accent_dir.is_dir():
            continue
        for speaker_dir in sorted(accent_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker = speaker_dir.name.upper()
            if speaker not in SPEAKER_TO_ACCENT:
                continue
            wav_dir = speaker_dir / "wav"
            if not wav_dir.is_dir():
                continue
            for wav_file in sorted(wav_dir.glob("*.wav")):
                entries.append((speaker, wav_file))
    return entries


def extract_mfcc_features(wav_path: Path, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """
    Извлекает MFCC + delta + delta2 из WAV файла.
    Returns: ndarray shape (T, 39)
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if len(y) == 0:
        return None

    # MFCC (13 коэффициентов, T фреймов)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=512, hop_length=160, win_length=400)  # (13, T)
    delta = librosa.feature.delta(mfcc, order=1)                              # (13, T)
    delta2 = librosa.feature.delta(mfcc, order=2)                             # (13, T)

    features = np.concatenate([mfcc, delta, delta2], axis=0)  # (39, T)
    return features.T.astype(np.float32)                        # (T, 39)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--n_mfcc", type=int, default=13)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else _ROOT / "data" / "l2_arctic"
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "outputs" / "features" / "mfcc"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root:  {data_root}")
    print(f"Output dir: {out_dir}")
    print(f"MFCC dim:   {args.n_mfcc * 3} (MFCC + delta + delta2)", flush=True)

    entries = find_wav_files(data_root)
    print(f"WAV files found: {len(entries)}", flush=True)

    ok, skip = 0, 0
    for i, (speaker, wav_path) in enumerate(entries):
        accent = SPEAKER_TO_ACCENT[speaker]
        stem = f"{speaker.lower()}_{wav_path.stem}"
        out_path = out_dir / f"{stem}.pt"

        if out_path.exists():
            ok += 1
            continue

        try:
            feat = extract_mfcc_features(wav_path, sr=args.sr, n_mfcc=args.n_mfcc)
            if feat is None or feat.shape[0] == 0:
                skip += 1
                continue

            feat_t = torch.from_numpy(feat)        # (T, 39)
            mean_t = feat_t.mean(dim=0)            # (39,)

            torch.save({
                "features": feat_t,
                "mean": mean_t,
                "speaker": speaker,
                "accent": accent,
            }, out_path)
            ok += 1
        except Exception as e:
            skip += 1
            continue

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(entries)}] ok={ok}, skip={skip}", flush=True)

    print(f"\nГотово: сохранено {ok}, пропущено {skip}", flush=True)
    print(f"Директория: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
