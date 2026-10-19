"""
Frame-level OT в непрерывном пространстве SSL-признаков (HuBERT/WavLM).

Два отображения акцент-источник → акцент-цель, обучаемые на облаках кадров:

GaussianOTMap — карта Монжа между гауссианами (Bures–Wasserstein), закрытая форма:
    T(x) = m_t + A (x - m_s),  A = S_s^{-1/2} (S_s^{1/2} S_t S_s^{1/2})^{1/2} S_s^{-1/2}
    Быстрый, интерпретируемый baseline; линейная карта.

EntropicOTMap — энтропийная (Sinkhorn) барицентрическая проекция с
    out-of-sample расширением (Pooladian & Niles-Weed, 2021):
    T(x) = sum_j softmax_j( log b_j + (g_j - c(x, y_j)) / eps ) y_j,
    где g — потенциал Sinkhorn на целевых кадрах. Нелинейная карта.

Оба отображения поддерживают интерполяцию по λ (displacement interpolation):
    x_λ = x + λ (T(x) - x),  λ ∈ [0, 1]
— это замена линейной интерполяции акцентных эмбеддингов a(λ) в AccentFlow.

bures_wasserstein_sq — квадрат расстояния Bures–Wasserstein между гауссианами.
quantize             — повторная квантизация кадров в k-means единицы.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .cost import normalize_cost, pairwise_cost
from .sinkhorn import sinkhorn

_DT = torch.float64


def _t(x) -> torch.Tensor:
    return x.to(_DT) if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x), dtype=_DT)


# ── Гауссианы ─────────────────────────────────────────────────────────────────

@dataclass
class GaussianStats:
    mean: torch.Tensor  # (D,)
    cov: torch.Tensor   # (D, D)


def fit_gaussian(X, shrink: float = 1e-2) -> GaussianStats:
    """
    Среднее и ковариация с усадкой к масштабированной единичной матрице:
    S = (1-shrink) S_emp + shrink * tr(S_emp)/D * I. Нужна при D=768 и малых N.
    """
    X = _t(X)
    mean = X.mean(0)
    Xc = X - mean
    cov = Xc.T @ Xc / max(len(X) - 1, 1)
    D = cov.shape[0]
    cov = (1.0 - shrink) * cov + shrink * cov.diagonal().mean() * torch.eye(D, dtype=_DT)
    return GaussianStats(mean, cov)


def _sym_sqrt(S: torch.Tensor, inverse: bool = False, floor: float = 1e-12) -> torch.Tensor:
    w, V = torch.linalg.eigh(S)
    w = w.clamp_min(floor)
    p = -0.5 if inverse else 0.5
    return (V * w.pow(p)) @ V.T


def bures_wasserstein_sq(g1: GaussianStats, g2: GaussianStats) -> float:
    """W_2^2(N(m1,S1), N(m2,S2)) = |m1-m2|^2 + tr(S1 + S2 - 2 (S1^½ S2 S1^½)^½)."""
    r1 = _sym_sqrt(g1.cov)
    cross = _sym_sqrt(r1 @ g2.cov @ r1)
    bures = torch.trace(g1.cov) + torch.trace(g2.cov) - 2.0 * torch.trace(cross)
    return float((g1.mean - g2.mean).pow(2).sum() + bures.clamp_min(0.0))


class GaussianOTMap:
    """
    Карта Монжа между гауссиановыми приближениями облаков кадров.

    cov_mode:
      full — полная ковариация (Bures–Wasserstein), нужны N >> D кадров;
      diag — покомпонентное масштабирование sqrt(var_t / var_s), N ≳ 20 достаточно;
      mean — только сдвиг средних (A = I), самый устойчивый при малых N.
    """

    def __init__(self, shrink: float = 1e-2, cov_mode: str = "full"):
        if cov_mode not in ("full", "diag", "mean"):
            raise ValueError(f"cov_mode must be full|diag|mean, got {cov_mode!r}")
        self.shrink, self.cov_mode = shrink, cov_mode
        self.src: Optional[GaussianStats] = None
        self.tgt: Optional[GaussianStats] = None
        self.A: Optional[torch.Tensor] = None       # (D, D) для full
        self.scale: Optional[torch.Tensor] = None   # (D,) для diag/mean

    @staticmethod
    def _diag_stats(X, shrink: float) -> GaussianStats:
        X = _t(X)
        mean = X.mean(0)
        var = X.var(0, unbiased=len(X) > 1) if len(X) > 1 else torch.ones_like(mean)
        var = (1.0 - shrink) * var + shrink * var.mean()
        return GaussianStats(mean, torch.diag(var))

    def fit(self, src_frames, tgt_frames) -> "GaussianOTMap":
        if self.cov_mode == "full":
            self.src = fit_gaussian(src_frames, self.shrink)
            self.tgt = fit_gaussian(tgt_frames, self.shrink)
            rs, rs_inv = _sym_sqrt(self.src.cov), _sym_sqrt(self.src.cov, inverse=True)
            self.A = rs_inv @ _sym_sqrt(rs @ self.tgt.cov @ rs) @ rs_inv
        else:
            self.src = self._diag_stats(src_frames, self.shrink)
            self.tgt = self._diag_stats(tgt_frames, self.shrink)
            if self.cov_mode == "diag":
                self.scale = (self.tgt.cov.diagonal() / self.src.cov.diagonal().clamp_min(1e-12)).sqrt()
            else:
                self.scale = torch.ones_like(self.src.mean)
        return self

    @property
    def distance_sq(self) -> float:
        return bures_wasserstein_sq(self.src, self.tgt)

    def __call__(self, X, lam: float = 1.0) -> torch.Tensor:
        X = _t(X)
        if self.cov_mode == "full":
            T = self.tgt.mean + (X - self.src.mean) @ self.A.T
        else:
            T = self.tgt.mean + (X - self.src.mean) * self.scale
        return X + lam * (T - X)


# ── Энтропийная карта ─────────────────────────────────────────────────────────

class EntropicOTMap:
    """Sinkhorn-барицентрическая проекция с out-of-sample расширением."""

    def __init__(self, metric: str = "sqeuclidean", eps: float = 0.05,
                 n_max: int = 2000, n_iter: int = 300, seed: int = 0):
        self.metric, self.eps, self.n_max, self.n_iter, self.seed = metric, eps, n_max, n_iter, seed
        self.Y: Optional[torch.Tensor] = None
        self.g: Optional[torch.Tensor] = None
        self.log_b: Optional[torch.Tensor] = None
        self.scale: float = 1.0

    def _subsample(self, X) -> torch.Tensor:
        X = _t(X)
        if len(X) <= self.n_max:
            return X
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(self.seed))[: self.n_max]
        return X[idx]

    def fit(self, src_frames, tgt_frames) -> "EntropicOTMap":
        X, Y = self._subsample(src_frames), self._subsample(tgt_frames)
        C = pairwise_cost(X, Y, self.metric)
        self.scale = float(C.mean())
        C = C / self.scale  # безразмерный eps
        a = torch.full((len(X),), 1.0 / len(X), dtype=_DT)
        b = torch.full((len(Y),), 1.0 / len(Y), dtype=_DT)
        res = sinkhorn(a, b, C, eps=self.eps, n_iter=self.n_iter)
        self.Y, self.g, self.log_b = Y, res.g, b.log()
        self.cost = float(res.cost) * self.scale
        return self

    def __call__(self, X, lam: float = 1.0, batch: int = 4096) -> torch.Tensor:
        X = _t(X)
        out = []
        for s in range(0, len(X), batch):
            xb = X[s:s + batch]
            c = pairwise_cost(xb, self.Y, self.metric) / self.scale
            w = torch.softmax(self.log_b[None, :] + (self.g[None, :] - c) / self.eps, dim=1)
            out.append(w @ self.Y)
        T = torch.cat(out)
        return X + lam * (T - X)


# ── Квантизация ───────────────────────────────────────────────────────────────

def quantize(X, centroids, batch: int = 8192) -> np.ndarray:
    """Ближайший центроид k-means для каждого кадра → единицы (T,)."""
    c = _t(centroids)
    X = _t(X)
    out = [pairwise_cost(X[s:s + batch], c, "sqeuclidean").argmin(1) for s in range(0, len(X), batch)]
    return torch.cat(out).numpy()


# ── Условный по единице OT ────────────────────────────────────────────────────

class UnitConditionalOTMap:
    """
    OT отдельно для каждой k-means единицы u: T_u переносит кадры src-акцента
    с единицей u в распределение кадров tgt-акцента с той же единицей u.

    Убирает confound содержания (единица ≈ псевдо-фонема); перенос остаётся
    внутри ячейки Вороного, поэтому единицы почти не меняются.

    Единицы, где кадров меньше min_frames в src или tgt, не отображаются
    (тождественно) — доля покрытых кадров в `coverage`.

    kind: "bw" (GaussianOTMap, cov_mode full|diag|mean) или "ent" (EntropicOTMap).
    """

    def __init__(self, kind: str = "bw", cov_mode: str = "diag", shrink: float = 0.3,
                 min_frames: int = 30, metric: str = "sqeuclidean", eps: float = 0.05,
                 n_max: int = 300, seed: int = 0):
        if kind not in ("bw", "ent"):
            raise ValueError(f"kind must be bw|ent, got {kind!r}")
        self.kind, self.cov_mode, self.shrink = kind, cov_mode, shrink
        self.min_frames, self.metric, self.eps, self.n_max, self.seed = min_frames, metric, eps, n_max, seed
        self.maps: dict = {}
        self.coverage: float = 0.0

    def _new_map(self):
        if self.kind == "bw":
            return GaussianOTMap(self.shrink, self.cov_mode)
        return EntropicOTMap(self.metric, self.eps, self.n_max, seed=self.seed)

    def fit(self, src_frames, tgt_frames, src_units, tgt_units) -> "UnitConditionalOTMap":
        src_frames, tgt_frames = _t(src_frames), _t(tgt_frames)
        src_units, tgt_units = np.asarray(src_units), np.asarray(tgt_units)
        self.maps = {}
        for u in np.intersect1d(np.unique(src_units), np.unique(tgt_units)):
            si, ti = np.where(src_units == u)[0], np.where(tgt_units == u)[0]
            if len(si) >= self.min_frames and len(ti) >= self.min_frames:
                self.maps[int(u)] = self._new_map().fit(src_frames[si], tgt_frames[ti])
        self.coverage = float(np.isin(src_units, list(self.maps)).mean()) if len(src_units) else 0.0
        return self

    def __call__(self, X, units, lam: float = 1.0) -> torch.Tensor:
        X = _t(X)
        units = np.asarray(units)
        out = X.clone()
        for u in np.unique(units):
            m = self.maps.get(int(u))
            if m is not None:
                idx = torch.as_tensor(np.where(units == u)[0])
                out[idx] = m(X[idx], lam=lam)
        return out
