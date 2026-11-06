"""
Извлечение текстовых признаков из транскриптов L2-ARCTIC.

Два метода:
1. TF-IDF (лёгкий baseline): позволяет поймать лексические паттерны
2. Character n-gram TF-IDF: ловит акцентные паттерны на уровне фонем/букв

Транскрипты одинаковые у всех спикеров (CMU-ARCTIC sentences),
поэтому текстовые признаки одинаковы для всех акцентов — интерпретация:
в мультимодальной модели текст помогает устранить контентные различия
и усилить сигнал акцента в аудио-части.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def load_transcripts(data_root: Path) -> dict:
    """
    Загружает все транскрипты.
    Returns: dict {(speaker, stem): text}
    """
    from src.data.accent_map import SPEAKER_TO_ACCENT
    transcripts = {}
    for accent_dir in sorted(data_root.iterdir()):
        if not accent_dir.is_dir():
            continue
        for speaker_dir in sorted(accent_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker = speaker_dir.name.upper()
            if speaker not in SPEAKER_TO_ACCENT:
                continue
            txt_dir = speaker_dir / "transcript"
            if not txt_dir.is_dir():
                continue
            for txt_file in sorted(txt_dir.glob("*.txt")):
                text = txt_file.read_text(encoding="utf-8").strip()
                transcripts[(speaker, txt_file.stem)] = text
    return transcripts


class TFIDFTextEncoder:
    """TF-IDF encoder для транскриптов. Поддерживает word и char n-gram режимы."""

    def __init__(self, mode: str = "char", ngram_range: Tuple = (2, 4), max_features: int = 512):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.mode = mode
        if mode == "char":
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=ngram_range,
                max_features=max_features, sublinear_tf=True
            )
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2),
                max_features=max_features, sublinear_tf=True
            )
        self.is_fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.is_fitted = True
        return self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
