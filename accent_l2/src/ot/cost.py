"""
Матрицы стоимости для оптимального транспорта (OT).

pairwise_cost      — стоимость между двумя наборами векторов
                     (sqeuclidean | euclidean | cosine | mahalanobis).
whitening_matrix   — матрица отбеливания (для Mahalanobis-стоимости).
accent_subspace    — LDA-проекция, оставляющая акцентно-дискриминативные оси.
normalize_cost     — приведение масштаба стоимости, чтобы eps в Sinkhorn
                     был сопоставим между метриками и слоями.

Все функции работают с torch.Tensor (CPU/GPU) и принимают np.ndarray.
"""

from typing import Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

METRICS = ("sqeuclidean", "euclidean", "cosine", "mahalanobis")


def _as_tensor(x: ArrayLike, dtype=torch.float64) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def whitening_matrix(X: ArrayLike, eps: float = 1e-6) -> torch.Tensor:
    """
    ZCA-отбеливание: W такая, что cov(X @ W) ≈ I.

    Для SSL-признаков (HuBERT/WavLM) отбеливание после вычитания
    дикторского среднего убирает доминирующие дикторские оси.
    """
    X = _as_tensor(X)
    Xc = X - X.mean(0, keepdim=True)
    cov = Xc.T @ Xc / max(len(X) - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    inv_sqrt = evecs @ torch.diag((evals.clamp_min(0) + eps).rsqrt()) @ evecs.T
    return inv_sqrt


def accent_subspace(
    X: ArrayLike, y: ArrayLike, n_components: Optional[int] = None
) -> torch.Tensor:
    """
    LDA-проекция (D, d), d ≤ n_classes - 1. Стоимость в этом подпространстве
    учитывает только оси, различающие акценты.

    Args:
        X: (N, D) признаки (кадры или центроиды кластеров с метками).
        y: (N,) метки акцентов.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(np.asarray(X), np.asarray(y))
    d = lda.n_components if lda.n_components is not None else lda.scalings_.shape[1]
    return torch.as_tensor(lda.scalings_[:, :d], dtype=torch.float64)


def pairwise_cost(
    x: ArrayLike,
    y: ArrayLike,
    metric: str = "sqeuclidean",
    whiten: Optional[ArrayLike] = None,
    projection: Optional[ArrayLike] = None,
) -> torch.Tensor:
    """
    Матрица стоимости C[i, j] = d(x_i, y_j), форма (n, m).

    Args:
        x, y:       (n, D), (m, D).
        metric:     sqeuclidean | euclidean | cosine | mahalanobis.
        whiten:     (D, D) матрица для mahalanobis (см. whitening_matrix).
        projection: (D, d) — применить к x и y до вычисления стоимости
                    (например, accent_subspace).
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    x, y = _as_tensor(x), _as_tensor(y)
    if projection is not None:
        P = _as_tensor(projection)
        x, y = x @ P, y @ P

    if metric == "mahalanobis":
        if whiten is None:
            raise ValueError("metric='mahalanobis' requires `whiten`")
        W = _as_tensor(whiten)
        x, y = x @ W, y @ W
        metric = "sqeuclidean"

    if metric == "cosine":
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        yn = y / y.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (1.0 - xn @ yn.T).clamp_min(0.0)

    sq = (x.pow(2).sum(1)[:, None] + y.pow(2).sum(1)[None, :] - 2.0 * x @ y.T).clamp_min(0.0)
    return sq.sqrt() if metric == "euclidean" else sq


def normalize_cost(C: torch.Tensor, how: str = "mean") -> torch.Tensor:
    """Делит C на среднее/медиану/максимум, чтобы eps был безразмерным."""
    if how == "mean":
        scale = C.mean()
    elif how == "median":
        scale = C.median()
    elif how == "max":
        scale = C.max()
    else:
        raise ValueError(f"how must be mean|median|max, got {how!r}")
    return C / scale.clamp_min(1e-12)
