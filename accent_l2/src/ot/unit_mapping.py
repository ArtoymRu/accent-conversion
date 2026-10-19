"""
OT на дискретных единицах (k-means токены HuBERT).

Акцент = распределение по K единицам. План переноса между распределением
акцента-источника и целевого акцента задаёт «словарь замены» единиц.
Поскольку C_ii = 0, OT оставляет совпадающую массу на месте и переносит
только разницу распределений — т.е. меняет минимум токенов.

Режимы применения к последовательности токенов:
  hard          — j = argmax_j P[i, j]
  soft          — j ~ P[i, :]^(1/T)  (стохастический)
  interpolated  — смещение центроида c_i к барицентрической цели
                  m_i = sum_j P[i,j] c_j на долю lam ∈ [0,1], затем ближайший
                  центроид (аналог λ из AccentFlow; displacement interpolation).

Данные: outputs/features/<model>_discrete/k<K>/<SPEAKER>_<utt>.pt
        {'tokens': Tensor(T,), 'speaker': str, 'accent': str}
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from .cost import normalize_cost, pairwise_cost
from .sinkhorn import SinkhornResult, sinkhorn


# ── Распределения акцентов ────────────────────────────────────────────────────

def load_accent_units(
    token_dir: Path,
    speaker_to_accent: Dict[str, str],
    include_speakers: Optional[Iterable[str]] = None,
    exclude_speakers: Optional[Iterable[str]] = None,
    dedup: bool = False,
) -> Dict[str, List[np.ndarray]]:
    """
    Загружает токены, группируя по акценту.

    Args:
        include_speakers / exclude_speakers: фильтры по дикторам
            (для speaker-independent протокола: fit на train, eval на held-out).
        dedup: схлопнуть подряд идущие повторы (убирает влияние темпа).
    """
    inc = set(include_speakers) if include_speakers is not None else None
    exc = set(exclude_speakers or ())
    out: Dict[str, List[np.ndarray]] = {}
    for pt in sorted(Path(token_dir).glob("*.pt")):
        spk = pt.stem.split("_")[0].upper()
        accent = speaker_to_accent.get(spk)
        if accent is None or spk in exc or (inc is not None and spk not in inc):
            continue
        tok = torch.load(pt, map_location="cpu", weights_only=True)["tokens"].numpy()
        if dedup and len(tok):
            tok = tok[np.concatenate([[True], tok[1:] != tok[:-1]])]
        out.setdefault(accent, []).append(tok)
    return out


def unit_histogram(token_seqs: List[np.ndarray], K: int, alpha: float = 1e-3) -> np.ndarray:
    """Нормированная гистограмма единиц со сглаживанием add-alpha (нет нулевых масс)."""
    counts = np.zeros(K, dtype=np.float64)
    for t in token_seqs:
        counts += np.bincount(t, minlength=K)[:K]
    counts += alpha
    return counts / counts.sum()


# ── Перенос ───────────────────────────────────────────────────────────────────

@dataclass
class UnitTransport:
    plan: np.ndarray        # (K, K) совместное распределение (src_unit, tgt_unit)
    src_hist: np.ndarray    # (K,)
    tgt_hist: np.ndarray    # (K,)
    centroids: np.ndarray   # (K, D) центроиды k-means
    eps: float
    cost: float             # <plan, C>

    @property
    def conditional(self) -> np.ndarray:
        """P[i, j] = p(tgt=j | src=i); строка с нулевой массой → тождественная."""
        mass = self.plan.sum(1, keepdims=True)
        K = self.plan.shape[0]
        P = np.where(mass > 0, self.plan / np.maximum(mass, 1e-300), np.eye(K))
        return P / P.sum(1, keepdims=True)

    def map_hard(self, tokens: np.ndarray) -> np.ndarray:
        return self.conditional.argmax(1)[tokens]

    def map_soft(
        self,
        tokens: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        P = self.conditional ** (1.0 / temperature)
        P /= P.sum(1, keepdims=True)
        cdf = P.cumsum(1)
        u = rng.random(len(tokens))
        j = (u[:, None] > cdf[tokens]).sum(1)
        return np.minimum(j, P.shape[1] - 1)

    def map_interpolated(self, tokens: np.ndarray, lam: float) -> np.ndarray:
        """Сдвиг центроида на долю lam к барицентрической цели + ближайший центроид."""
        c = self.centroids
        bary = self.conditional @ c                      # (K, D)
        moved = (1.0 - lam) * c + lam * bary             # (K, D)
        d = pairwise_cost(moved, c, "sqeuclidean").numpy()
        return d.argmin(1)[tokens]

    def apply(self, tokens: np.ndarray, mode: str = "hard", **kw) -> np.ndarray:
        if mode == "hard":
            return self.map_hard(tokens)
        if mode == "soft":
            return self.map_soft(tokens, **kw)
        if mode == "interpolated":
            return self.map_interpolated(tokens, **kw)
        raise ValueError(f"mode must be hard|soft|interpolated, got {mode!r}")


def unit_cost_matrix(
    centroids: np.ndarray,
    metric: str = "cosine",
    normalize: str = "mean",
    **cost_kw,
) -> torch.Tensor:
    """Стоимость между единицами = расстояние между центроидами k-means."""
    C = pairwise_cost(centroids, centroids, metric, **cost_kw)
    return normalize_cost(C, normalize) if normalize else C


def fit_unit_transport(
    src_hist: np.ndarray,
    tgt_hist: np.ndarray,
    centroids: np.ndarray,
    metric: str = "cosine",
    eps: float = 0.05,
    n_iter: int = 1000,
    **cost_kw,
) -> UnitTransport:
    """Sinkhorn-план между распределениями единиц двух акцентов."""
    C = unit_cost_matrix(centroids, metric, **cost_kw)
    res: SinkhornResult = sinkhorn(
        torch.as_tensor(src_hist), torch.as_tensor(tgt_hist), C, eps=eps, n_iter=n_iter
    )
    return UnitTransport(
        plan=res.plan.numpy(),
        src_hist=np.asarray(src_hist),
        tgt_hist=np.asarray(tgt_hist),
        centroids=np.asarray(centroids),
        eps=eps,
        cost=float(res.cost),
    )


# ── Метрики сравнения распределений ───────────────────────────────────────────

def ot_distance(
    hist_a: np.ndarray, hist_b: np.ndarray, C: torch.Tensor, eps: float = 0.05
) -> float:
    """Транспортная стоимость между гистограммами (Sinkhorn)."""
    return float(
        sinkhorn(torch.as_tensor(hist_a), torch.as_tensor(hist_b), C, eps=eps, n_iter=1000).cost
    )


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon (нат). Не-OT baseline на гистограммах единиц."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log(x[mask] / y[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def change_rate(before: np.ndarray, after: np.ndarray) -> float:
    """Доля изменённых токенов (отображение покадровое → Hamming = edit distance)."""
    return float((before != after).mean()) if len(before) else 0.0
