"""
Диктор-устойчивый классификатор акцента на признаках высказывания.

Проблема: в L2-ARCTIC всего 4 диктора на акцент, поэтому классификатор на
сырых признаках легко опирается на дикторские/канальные смещения. Любой
перенос, сдвигающий кадры в область train-дикторов целевого акцента
(в т.ч. чистый перенос голоса), даёт p_tgt ≈ 1.

Решение — Nuisance Attribute Projection (NAP) по дикторам ВНУТРИ акцента:
  1. вычесть среднее акцента (остаётся только межличностная вариация);
  2. LDA по меткам дикторов → r направлений, различающих дикторов одного
     акцента (это заведомо не акцентные оси);
  3. спроецировать признаки на ортогональное дополнение этих направлений;
  4. логистическая регрессия по акценту на проекциях.
Оси между акцентами сохраняются, т.к. LDA обучается на дикторах одного акцента.
Ограничение: r ≤ n_speakers_train - n_accents (при 3 train-дикторах на акцент: 12).

within_accent_speaker_acc — probe: точность различения дикторов одного акцента
по проекциям (шанс = 1/число дикторов). Чем ближе к шансу, тем меньше дикторской
информации осталось.
"""

from typing import Optional

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class SpeakerRobustAccentClassifier:
    def __init__(self, n_nuisance: int = 8, C: float = 0.5, lda_shrinkage="auto"):
        self.n_nuisance, self.C, self.lda_shrinkage = n_nuisance, C, lda_shrinkage
        self.scaler: Optional[StandardScaler] = None
        self.Q: Optional[np.ndarray] = None   # (D, r) ортонормированный базис дикторской подпространства
        self.clf: Optional[LogisticRegression] = None

    def fit(self, X: np.ndarray, accents: np.ndarray, speakers: np.ndarray) -> "SpeakerRobustAccentClassifier":
        accents, speakers = np.asarray(accents), np.asarray(speakers)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        if self.n_nuisance > 0:
            Xr = Xs.copy()
            for a in np.unique(accents):
                m = accents == a
                Xr[m] -= Xs[m].mean(0)
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=self.lda_shrinkage)
            lda.fit(Xr, speakers)
            r = min(self.n_nuisance, lda.scalings_.shape[1])
            self.Q, _ = np.linalg.qr(lda.scalings_[:, :r])
        else:
            self.Q = None
        self.clf = LogisticRegression(max_iter=1000, C=self.C).fit(self._project(Xs), accents)
        return self

    def _project(self, Xs: np.ndarray) -> np.ndarray:
        return Xs if self.Q is None else Xs - (Xs @ self.Q) @ self.Q.T

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Признаки после стандартизации и удаления дикторского подпространства."""
        return self._project(self.scaler.transform(X))

    @property
    def classes_(self) -> np.ndarray:
        return self.clf.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.transform(X))


def within_accent_speaker_acc(Z: np.ndarray, accents: np.ndarray, speakers: np.ndarray,
                              seed: int = 0) -> float:
    """
    Probe: различение дикторов внутри акцента по признакам Z (50/50 сплит
    высказываний тех же дикторов). Среднее по акцентам; шанс = 1/(дикторов в акценте).
    """
    accents, speakers = np.asarray(accents), np.asarray(speakers)
    rng = np.random.default_rng(seed)
    accs = []
    for a in np.unique(accents):
        idx = np.where(accents == a)[0]
        if len(np.unique(speakers[idx])) < 2:
            continue
        perm = rng.permutation(idx)
        half = len(perm) // 2
        tr, te = perm[:half], perm[half:]
        clf = LogisticRegression(max_iter=500, C=0.5).fit(Z[tr], speakers[tr])
        accs.append((clf.predict(Z[te]) == speakers[te]).mean())
    return float(np.mean(accs))
