"""
Вычисление метрики Speaker Clustering Purity для дискретных представлений речи.

Определение:
  Purity = (1/N) * Σ_k max_j count(cluster=k, speaker=j)

  Для каждого дискретного токена k находим «доминирующего» диктора j,
  суммируем количество его фреймов по всем токенам и нормируем на N.
  Purity = 1: каждый токен принадлежит ровно одному диктору (максимальное разделение).
  Purity = 1/S: случайное распределение по S дикторам.

Поддерживает:
  - k-means токены HuBERT/WavLM: .pt с ключом 'tokens' Tensor(T,)
  - EnCodec RVQ коды:            .pt с ключом 'codes'  Tensor(nq, T)

Использование как модуль:
  from src.analysis.speaker_purity import compute_purity_from_dir
  purity, n_frames, vocab_size = compute_purity_from_dir(
      Path("outputs/features/hubert_discrete/k100")
  )

Использование как скрипт:
  python src/analysis/speaker_purity.py outputs/features/hubert_discrete/k100
  python src/analysis/speaker_purity.py outputs/features/encodec/nq2 --mode encodec --encodec_nq 2
"""

import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


def _codes_to_joint_tokens(codes: np.ndarray, nq: int) -> np.ndarray:
    """
    Преобразует коды RVQ (nq, T) в одномерный массив составных токенов (T,).
    Для nq=1 возвращает codes[0] напрямую.
    Для nq>1 вычисляет полиномиальный хэш: tok = Σ codes[q] * 1031^q
    (prime > 1024, ключи уникальны при отсутствии коллизий, что верно для нашего диапазона).
    """
    nq = min(nq, codes.shape[0])
    if nq == 1:
        return codes[0].astype(np.int64)

    # Полиномиальный хэш в int64 (wraps around, приемлемо для dict-ключей при анализе)
    # Модульный хэш (MOD — простое Мерсенна, исключает переполнение int64)
    MOD = np.int64(2**61 - 1)
    PRIME = np.int64(1031)
    result = codes[0].astype(np.int64) % MOD
    for q in range(1, nq):
        result = (result * PRIME + codes[q].astype(np.int64)) % MOD
    return result


def compute_purity_from_dir(
    feat_dir: Path,
    mode: str = "kmeans",
    encodec_nq: int = 1,
) -> Tuple[float, int, int]:
    """
    Вычисляет Speaker Clustering Purity по всем .pt файлам в директории.

    Args:
        feat_dir:   директория с .pt файлами дискретных токенов
        mode:       "kmeans" (ключ 'tokens') или "encodec" (ключ 'codes')
        encodec_nq: для mode="encodec" — число используемых кодовых книг

    Returns:
        purity:    float [0, 1]
        n_frames:  суммарное число фреймов
        vocab_size: число уникальных токенов (активный словарь)
    """
    # token_id -> {speaker_id -> count}
    token_speaker_counts: dict = defaultdict(lambda: defaultdict(int))
    speaker_to_id: dict = {}

    def spk_id(speaker: str) -> int:
        if speaker not in speaker_to_id:
            speaker_to_id[speaker] = len(speaker_to_id)
        return speaker_to_id[speaker]

    n_frames = 0
    pt_files = sorted(feat_dir.glob("*.pt"))
    if not pt_files:
        return 0.0, 0, 0

    for pt_path in pt_files:
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            speaker = data["speaker"]
            sid = spk_id(speaker)

            if mode == "kmeans":
                tokens = data["tokens"].numpy().astype(np.int64)  # (T,)
            elif mode == "encodec":
                codes = data["codes"].numpy()  # (nq, T)
                tokens = _codes_to_joint_tokens(codes, encodec_nq)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Counter быстрее, чем поэлементный цикл (реализован на C)
            counts = Counter(tokens.tolist())
            for tok, cnt in counts.items():
                token_speaker_counts[tok][sid] += cnt
            n_frames += len(tokens)

        except Exception as e:
            print(f"  Warning: {pt_path.name}: {e}", flush=True)
            continue

    if n_frames == 0:
        return 0.0, 0, 0

    majority_sum = sum(
        max(spk_counts.values())
        for spk_counts in token_speaker_counts.values()
    )
    purity = majority_sum / n_frames
    vocab_size = len(token_speaker_counts)

    return purity, n_frames, vocab_size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Speaker Clustering Purity")
    parser.add_argument("feat_dir", type=str, help="Директория с .pt файлами токенов")
    parser.add_argument("--mode", choices=["kmeans", "encodec"], default="kmeans")
    parser.add_argument("--encodec_nq", type=int, default=1,
                        help="Число кодовых книг EnCodec (только для --mode encodec)")
    cli = parser.parse_args()

    purity, n_frames, vocab_size = compute_purity_from_dir(
        Path(cli.feat_dir),
        mode=cli.mode,
        encodec_nq=cli.encodec_nq,
    )
    n_speakers = 24
    random_baseline = 1.0 / n_speakers
    print(f"Purity:          {purity:.4f}")
    print(f"Random baseline: {random_baseline:.4f}  (1/{n_speakers} speakers)")
    print(f"Frames:          {n_frames:,}")
    print(f"Active vocab:    {vocab_size:,}")
