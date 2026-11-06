"""
Сохранность диктора после OT-переноса на уровне признаков HuBERT.

Настоящих speaker embeddings (ECAPA / WavLM-SV) на выходе OT получить нельзя:
карты работают в пространстве кадров HuBERT, а ресинтеза в аудио в этом
репозитории для кадров нет (только units → mel → HiFi-GAN). Поэтому используется
признаковый прокси:

  эмбеддинг высказывания = LDA по меткам train-дикторов от stats-pooling
  (среднее и std кадров), как LDA в i-vector системах; косинусное сходство.

Протокол верификации с «зачислением» (enrollment): для каждого диктора — центроид
эмбеддингов его зачисляемых высказываний (train-диктора — train-высказывания,
held-out — отдельная половина его высказываний, не участвующая в тесте).

Метрики для отображённых высказываний диктора src:
  top1       — ближайший центроид (из всех дикторов) = src
  cos_src    — среднее косинусное сходство с центроидом src
  capture    — ближайший центроид принадлежит train-дикторам ЦЕЛЕВОГО акцента
               (перенос голоса вместо/вместе с акцентом)
  margin     — cos_src минус максимум cos по остальным дикторам (>0 — диктор сохранён)

Ограничение: прокси измеряет то, что различает дикторов в HuBERT-признаках,
а не восприятие; финальная оценка — speaker embedding на синтезе и SMOS.
"""

from typing import Dict, Iterable, List

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def pool(utts: List[np.ndarray]) -> np.ndarray:
    """Stats pooling: [mean, std] по кадрам каждого высказывания → (N, 2D)."""
    return np.stack([np.concatenate([u.mean(0), u.std(0)]) for u in utts])


class SpeakerEmbedder:
    def __init__(self, shrinkage="auto"):
        self.shrinkage = shrinkage
        self.scaler = None
        self.lda = None
        self.centroids: Dict[str, np.ndarray] = {}

    def fit(self, utts_by_speaker: Dict[str, List[np.ndarray]]) -> "SpeakerEmbedder":
        X, y = [], []
        for spk, us in utts_by_speaker.items():
            X.append(pool(us))
            y += [spk] * len(us)
        X = np.concatenate(X)
        self.scaler = StandardScaler().fit(X)
        self.lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=self.shrinkage).fit(
            self.scaler.transform(X), y)
        for spk, us in utts_by_speaker.items():
            self.enroll(spk, us)
        return self

    def embed(self, utts: List[np.ndarray]) -> np.ndarray:
        Z = self.lda.transform(self.scaler.transform(pool(utts)))
        return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-12)

    def enroll(self, name: str, utts: List[np.ndarray]) -> None:
        c = self.embed(utts).mean(0)
        self.centroids[name] = c / max(np.linalg.norm(c), 1e-12)

    def retention(self, mapped_utts: List[np.ndarray], true_speaker: str,
                  target_speakers: Iterable[str]) -> Dict[str, float]:
        names = list(self.centroids)
        C = np.stack([self.centroids[n] for n in names])            # (S, d)
        sim = self.embed(mapped_utts) @ C.T                          # (N, S)
        si = names.index(true_speaker)
        tgt = np.array([names.index(s) for s in target_speakers if s in names], dtype=int)
        nearest = sim.argmax(1)
        other = np.delete(sim, si, axis=1).max(1)
        return {
            "top1": float((nearest == si).mean()),
            "cos_src": float(sim[:, si].mean()),
            "capture": float(np.isin(nearest, tgt).mean()) if len(tgt) else float("nan"),
            "margin": float((sim[:, si] - other).mean()),
        }
