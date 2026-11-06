"""
Загрузка кадровых признаков L2-ARCTIC и фолды leave-one-speaker-out (LOSO).

fold_heldout(f) — в каждом акценте held-out диктор = sorted(speakers)[f]
                  (f = 3 совпадает с HELD_OUT_SPEAKERS из utils/splits.py).
Всего 4 фолда; каждый диктор ровно один раз оказывается в тесте.
"""

from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch

from src.utils.splits import ACCENT_TO_SPEAKERS

N_FOLDS = 4


def fold_heldout(fold: int) -> Set[str]:
    """Held-out дикторы фолда: по одному на акцент."""
    if not 0 <= fold < N_FOLDS:
        raise ValueError(f"fold must be in [0, {N_FOLDS}), got {fold}")
    return {sorted(spks)[fold] for spks in ACCENT_TO_SPEAKERS.values()}


def load_utts(feat_dir: Path, speaker: str, n: int, rng: np.random.Generator) -> List[np.ndarray]:
    """До n случайных высказываний диктора: список (T, D) float32."""
    files = sorted(Path(feat_dir).glob(f"{speaker}_*.pt"))
    if len(files) > n:
        files = [files[i] for i in rng.choice(len(files), n, replace=False)]
    out = []
    for f in files:
        x = torch.load(f, map_location="cpu", weights_only=True)["features"]
        out.append(x.squeeze(0).numpy().astype(np.float32))
    return out


def load_all_speakers(feat_dir: Path, speaker_to_accent: Dict[str, str], n: int,
                      rng: np.random.Generator) -> Dict[str, List[np.ndarray]]:
    """speaker -> список высказываний (T, D)."""
    return {spk: load_utts(feat_dir, spk, n, rng) for spk in speaker_to_accent}


def sample_frames(utts: List[np.ndarray], per_utt: int, rng: np.random.Generator) -> np.ndarray:
    chunks = []
    for u in utts:
        idx = rng.choice(len(u), min(per_utt, len(u)), replace=False)
        chunks.append(u[idx])
    return np.concatenate(chunks)


def subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]


def utterance_means(utts: List[np.ndarray]) -> np.ndarray:
    return np.stack([u.mean(0) for u in utts])


def load_utts_ids(feat_dir: Path, speaker: str, n: int, rng: np.random.Generator):
    """Как load_utts, но дополнительно возвращает id высказываний: (ids, utts)."""
    files = sorted(Path(feat_dir).glob(f"{speaker}_*.pt"))
    if len(files) > n:
        files = [files[i] for i in rng.choice(len(files), n, replace=False)]
    ids, out = [], []
    for f in files:
        x = torch.load(f, map_location="cpu", weights_only=True)["features"]
        ids.append(f.stem.split("_", 1)[1])
        out.append(x.squeeze(0).numpy().astype(np.float32))
    return ids, out
