"""
Метрики качества речи: MCD (Mel Cepstral Distortion) и WER.
"""

import numpy as np
from typing import Optional


def compute_mcd(mfcc_ref: np.ndarray, mfcc_syn: np.ndarray) -> float:
    """
    Mel Cepstral Distortion (MCD) между двумя последовательностями MFCC.
    MCD = (10 / ln10) * sqrt(2 * sum((c_ref - c_syn)^2))
    Принимает mfcc матрицы (n_frames, n_mfcc) или векторы средних (n_mfcc,).
    При разных длинах усредняет до минимального числа фреймов.
    """
    mfcc_ref = np.atleast_2d(mfcc_ref)
    mfcc_syn = np.atleast_2d(mfcc_syn)
    min_frames = min(mfcc_ref.shape[0], mfcc_syn.shape[0])
    if min_frames == 0:
        return float("nan")
    r = mfcc_ref[:min_frames, 1:]  # skip C0 (energy) per MCD convention
    s = mfcc_syn[:min_frames, 1:]
    diff = r - s
    mcd = (10.0 / np.log(10)) * np.sqrt(2.0 * np.mean(np.sum(diff ** 2, axis=1)))
    return float(mcd)


def compute_mcd_from_audio(
    y_ref: np.ndarray, y_syn: np.ndarray, sr: int = 16000,
    n_mfcc: int = 13, n_fft: int = 1024, hop_length: int = 256
) -> float:
    """Compute MCD between two raw audio signals using librosa MFCC."""
    import librosa
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop_length).T
    mfcc_syn = librosa.feature.mfcc(y=y_syn, sr=sr, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop_length).T
    return compute_mcd(mfcc_ref, mfcc_syn)


def compute_wer(hypothesis: str, reference: str) -> float:
    """
    Word Error Rate = (S + D + I) / N
    Uses jiwer library if available, falls back to manual computation.
    """
    try:
        from jiwer import wer
        return float(wer(reference, hypothesis))
    except ImportError:
        pass
    # Manual WER via dynamic programming
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    n = len(ref_words)
    if n == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    m = len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[n, m] / n


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for WER comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text
