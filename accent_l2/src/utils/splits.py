"""
Стратегии разбивки данных для оценки классификаторов акцентов.

speaker_independent_split  — правильная оценка: тест содержит только спикеров,
                             которых модель НЕ видела при обучении.
                             1 спикер на акцент (последний по алфавиту) → test.
                             Оставшиеся 3 спикера → train + val.

random_split               — утечка спикеров: одни высказывания спикера
                             в train, другие в test. Завышает метрики.

Held-out test speakers (speaker-independent):
  arabic:     ZHAA
  mandarin:   TXHC
  hindi:      TNI
  korean:     YKWK
  spanish:    NJS
  vietnamese: TLV
"""

from typing import List, Tuple, Optional
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


ACCENTS = ["arabic", "mandarin", "hindi", "korean", "spanish", "vietnamese"]

ACCENT_TO_SPEAKERS = {
    "arabic":     ["ABA", "SKA", "YBAA", "ZHAA"],
    "mandarin":   ["BWC", "LXC", "NCC", "TXHC"],
    "hindi":      ["ASI", "RRBI", "SVBI", "TNI"],
    "korean":     ["HJK", "HKK", "YDCK", "YKWK"],
    "spanish":    ["EBVS", "ERMS", "MBMPS", "NJS"],
    "vietnamese": ["HQTV", "PNV", "THV", "TLV"],
}

# Детерминированные held-out спикеры (последние по алфавиту в каждом акценте)
HELD_OUT_SPEAKERS = {
    accent: sorted(speakers)[-1]
    for accent, speakers in ACCENT_TO_SPEAKERS.items()
}
# = {"arabic": "ZHAA", "mandarin": "TXHC", "hindi": "TNI",
#    "korean": "YKWK", "spanish": "NJS", "vietnamese": "TLV"}


def speaker_independent_split(
    speakers: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Разделяет индексы на train / val / test по speaker-independent протоколу.

    Returns:
        idx_train, idx_val, idx_test — массивы индексов.
    """
    held_out = set(HELD_OUT_SPEAKERS.values())
    test_mask = np.array([s in held_out for s in speakers])

    idx_test = np.where(test_mask)[0]
    idx_tv = np.where(~test_mask)[0]

    y_tv = y[idx_tv]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state)
    tr_local, val_local = next(sss.split(idx_tv, y_tv))

    idx_train = idx_tv[tr_local]
    idx_val = idx_tv[val_local]

    return idx_train, idx_val, idx_test


def random_utterance_split(
    y: np.ndarray,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Случайное разбиение по высказываниям (с утечкой спикеров).
    Используется только для сравнения с speaker-independent.
    """
    idx = np.arange(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    idx_tv, idx_test = next(sss.split(idx, y))

    y_tv = y[idx_tv]
    val_rel = val_frac / (1.0 - test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=random_state)
    tr_local, val_local = next(sss2.split(idx_tv, y_tv))

    return idx_tv[tr_local], idx_tv[val_local], idx_test


def speaker_from_pt_filename(stem: str) -> Optional[str]:
    """
    Извлекает код спикера из имени файла формата 'SPEAKER_utterance'.
    Например: 'ABA_arctic_a0001' → 'ABA'.
    """
    from src.data.accent_map import SPEAKER_TO_ACCENT
    parts = stem.split("_", 1)
    if len(parts) >= 1:
        spk = parts[0].upper()
        if spk in SPEAKER_TO_ACCENT:
            return spk
    return None


def print_split_info(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    speakers: np.ndarray,
    y: np.ndarray,
    protocol: str = "speaker_independent",
):
    """Выводит статистику по разбивке."""
    print(f"\n  [{protocol}] split sizes: "
          f"train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    if protocol == "speaker_independent":
        test_spks = set(speakers[idx_test])
        train_spks = set(speakers[idx_train])
        print(f"  Held-out test speakers: {sorted(test_spks)}")
        print(f"  Train speakers: {sorted(train_spks)}")
        overlap = test_spks & train_spks
        if overlap:
            print(f"  ⚠️  OVERLAP detected: {overlap}")
        else:
            print(f"  ✓ No speaker overlap between train and test")
