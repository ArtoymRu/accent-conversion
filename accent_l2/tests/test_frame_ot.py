"""
Тесты frame-level OT на синтетике. Запуск:
    python tests/test_frame_ot.py      (без pytest)
    python -m pytest tests/test_frame_ot.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ot import (
    EntropicOTMap,
    GaussianOTMap,
    GaussianStats,
    bures_wasserstein_sq,
    fit_gaussian,
    quantize,
    sliced_wasserstein,
)

DT = torch.float64


def _cloud(n, mean, cov_scale, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = torch.tensor(cov_scale, dtype=DT)
    return torch.randn(n, d, generator=g, dtype=DT) * L + torch.tensor(mean, dtype=DT)


def test_bures_wasserstein_1d_closed_form():
    # W2^2(N(0,1), N(m,s^2)) = m^2 + (1-s)^2
    g1 = GaussianStats(torch.zeros(1, dtype=DT), torch.eye(1, dtype=DT))
    g2 = GaussianStats(torch.tensor([3.0], dtype=DT), torch.tensor([[4.0]], dtype=DT))
    assert abs(bures_wasserstein_sq(g1, g2) - (9.0 + 1.0)) < 1e-9
    assert abs(bures_wasserstein_sq(g1, g1)) < 1e-9


def test_gaussian_map_matches_target_moments():
    X = _cloud(5000, [0, 0, 0], [1.0, 2.0, 0.5], seed=1)
    Y = _cloud(5000, [4, -2, 1], [0.5, 1.0, 3.0], seed=2)
    m = GaussianOTMap(shrink=0.0).fit(X, Y)
    Z = m(X)
    gz, gy = fit_gaussian(Z, 0.0), fit_gaussian(Y, 0.0)
    assert (gz.mean - gy.mean).abs().max() < 1e-8
    assert (gz.cov - gy.cov).abs().max() < 1e-6
    assert bures_wasserstein_sq(gz, gy) < 1e-8


def test_gaussian_map_interpolation():
    X = _cloud(3000, [0, 0, 0], [1, 1, 1], seed=3)
    Y = _cloud(3000, [6, 0, 0], [2, 1, 1], seed=4)
    m = GaussianOTMap(shrink=0.0).fit(X, Y)
    assert (m(X, lam=0.0) - X).abs().max() < 1e-12
    gy = fit_gaussian(Y, 0.0)
    d = [bures_wasserstein_sq(fit_gaussian(m(X, lam), 0.0), gy) for lam in (0.0, 0.5, 1.0)]
    assert d[0] > d[1] > d[2]
    # W2 вдоль геодезической линейно по λ: sqrt(d(λ)) = (1-λ) sqrt(d(0))
    assert abs(d[1] ** 0.5 - 0.5 * d[0] ** 0.5) < 0.05 * d[0] ** 0.5


def test_entropic_map_shift():
    X = _cloud(1500, [0, 0], [1, 1], d=2, seed=5)
    Y = _cloud(1500, [3, 1], [1, 1], d=2, seed=6)
    m = EntropicOTMap("sqeuclidean", eps=0.05, n_max=1500, n_iter=500).fit(X, Y)
    Xnew = _cloud(500, [0, 0], [1, 1], d=2, seed=7)  # out-of-sample
    Z = m(Xnew)
    disp = (Z - Xnew).mean(0)
    assert (disp - torch.tensor([3.0, 1.0], dtype=DT)).abs().max() < 0.5
    assert sliced_wasserstein(Z, Y) < 0.5 * sliced_wasserstein(Xnew, Y)
    assert (m(Xnew, lam=0.0) - Xnew).abs().max() < 1e-12
    half = m(Xnew, lam=0.5)
    assert torch.allclose(half, Xnew + 0.5 * (Z - Xnew))


def test_entropic_map_cosine_runs():
    X = _cloud(300, [1, 0, 0], [0.3, 0.3, 0.3], seed=8)
    Y = _cloud(300, [0, 1, 0], [0.3, 0.3, 0.3], seed=9)
    Z = EntropicOTMap("cosine", eps=0.05, n_max=300, n_iter=200).fit(X, Y)(X)
    assert Z.shape == X.shape and torch.isfinite(Z).all()


def test_quantize_nearest_centroid():
    c = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = np.array([[0.1, 0.2], [9.0, 1.0], [1.0, 8.0], [-1.0, -1.0]])
    assert quantize(x, c).tolist() == [0, 1, 2, 0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
