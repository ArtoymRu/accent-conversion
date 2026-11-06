"""
Тесты OT-модуля на синтетике. Запуск:
    python tests/test_ot.py        (без pytest)
    python -m pytest tests/test_ot.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ot import (
    UnitTransport,
    change_rate,
    fit_unit_transport,
    js_divergence,
    normalize_cost,
    ot_distance,
    pairwise_cost,
    sinkhorn,
    sinkhorn_divergence,
    sinkhorn_unbalanced,
    sliced_wasserstein,
    unit_cost_matrix,
    whitening_matrix,
)


def _rand_cost(n, m, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(m, d, generator=g, dtype=torch.float64)
    return x, y, pairwise_cost(x, y, "sqeuclidean")


# ── cost ──────────────────────────────────────────────────────────────────────

def test_cost_metrics():
    x, y, _ = _rand_cost(6, 5)
    for metric in ("sqeuclidean", "euclidean", "cosine"):
        C = pairwise_cost(x, y, metric)
        assert C.shape == (6, 5) and (C >= 0).all()
    assert pairwise_cost(x, x, "cosine").diagonal().abs().max() < 1e-9
    assert pairwise_cost(x, x, "euclidean").diagonal().abs().max() < 1e-6
    W = whitening_matrix(torch.randn(500, 3, dtype=torch.float64))
    assert pairwise_cost(x, y, "mahalanobis", whiten=W).shape == (6, 5)
    try:
        pairwise_cost(x, y, "mahalanobis")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_normalize_cost():
    _, _, C = _rand_cost(10, 10)
    assert abs(normalize_cost(C, "mean").mean().item() - 1.0) < 1e-9


# ── sinkhorn ──────────────────────────────────────────────────────────────────

def test_sinkhorn_marginals_and_exact_cost():
    n = 12
    _, _, C = _rand_cost(n, n)
    C = normalize_cost(C)
    a = b = torch.full((n,), 1.0 / n, dtype=torch.float64)
    res = sinkhorn(a, b, C, eps=0.01, n_iter=5000)
    assert (res.plan.sum(1) - a).abs().max() < 1e-4
    assert (res.plan.sum(0) - b).abs().max() < 1e-4
    r, c = linear_sum_assignment(C.numpy())
    exact = C.numpy()[r, c].mean()  # равные веса 1/n → OT = среднее назначения
    assert res.cost.item() >= exact - 1e-9
    assert res.cost.item() - exact < 0.05 * exact + 0.01


def test_sinkhorn_zero_mass_rows():
    _, _, C = _rand_cost(5, 5)
    a = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0], dtype=torch.float64)
    b = torch.full((5,), 0.2, dtype=torch.float64)
    res = sinkhorn(a, b, normalize_cost(C), eps=0.05, n_iter=2000)
    assert torch.isfinite(res.plan).all()
    assert res.plan[2:].abs().max() == 0


def test_sinkhorn_divergence():
    x, y, _ = _rand_cost(10, 10)
    a = torch.full((10,), 0.1, dtype=torch.float64)
    Cxx, Cyy, Cxy = (pairwise_cost(p, q) for p, q in ((x, x), (y, y), (x, y)))
    s = sinkhorn_divergence(a, a, Cxy, Cxx, Cyy, eps=0.1)
    assert s.item() > 0
    s0 = sinkhorn_divergence(a, a, Cxx, Cxx, Cxx, eps=0.1)
    assert abs(s0.item()) < 1e-8


def test_unbalanced_large_tau_close_to_balanced():
    n = 8
    _, _, C = _rand_cost(n, n)
    C = normalize_cost(C)
    a = b = torch.full((n,), 1.0 / n, dtype=torch.float64)
    bal = sinkhorn(a, b, C, eps=0.05, n_iter=3000).cost.item()
    unb = sinkhorn_unbalanced(a, b, C, eps=0.05, tau=1e4, n_iter=3000).cost.item()
    assert abs(bal - unb) < 1e-2 * max(bal, 1e-3) + 1e-3


def test_unbalanced_small_tau_transports_less():
    n = 8
    _, _, C = _rand_cost(n, n)
    C = normalize_cost(C)
    a = b = torch.full((n,), 1.0 / n, dtype=torch.float64)
    big = sinkhorn_unbalanced(a, b, C, eps=0.05, tau=1e3, n_iter=1000).plan.sum().item()
    small = sinkhorn_unbalanced(a, b, C, eps=0.05, tau=0.01, n_iter=1000).plan.sum().item()
    assert small < big


def test_sliced_wasserstein():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2000, 4, generator=g, dtype=torch.float64)
    assert sliced_wasserstein(x, x).item() < 1e-9
    shift = torch.tensor([2.0, 0, 0, 0], dtype=torch.float64)
    y = torch.randn(1500, 4, generator=g, dtype=torch.float64) + shift  # другая длина
    sw = sliced_wasserstein(x, y, n_proj=512).item()
    assert abs(sw - 2.0 / 2.0) < 0.15  # SW_2 = |shift| / sqrt(D)


# ── unit mapping ──────────────────────────────────────────────────────────────

def _line_centroids(K=8):
    return np.stack([np.arange(K, dtype=np.float64), np.ones(K)], axis=1)  # точки на прямой


def test_unit_transport_identity_when_equal():
    K = 8
    h = np.full(K, 1.0 / K)
    tr = fit_unit_transport(h, h, _line_centroids(K), metric="sqeuclidean", eps=0.01)
    toks = np.random.default_rng(0).integers(0, K, 200)
    assert change_rate(toks, tr.map_hard(toks)) == 0.0
    assert change_rate(toks, tr.map_interpolated(toks, lam=1.0)) == 0.0


def test_unit_transport_moves_mass_toward_target():
    K = 8
    C = _line_centroids(K)
    src = np.array([0.4, 0.3, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])
    tgt = src[::-1].copy()
    tr = fit_unit_transport(src, tgt, C, metric="sqeuclidean", eps=0.01)
    assert np.allclose(tr.plan.sum(1), src, atol=1e-6)
    assert np.allclose(tr.plan.sum(0), tgt, atol=1e-6)

    rng = np.random.default_rng(0)
    toks = rng.choice(K, size=20000, p=src)
    Cn = unit_cost_matrix(C, "sqeuclidean")
    hist = lambda t: np.bincount(t, minlength=K) / len(t)

    d_before = ot_distance(hist(toks), tgt, Cn)
    for mode, kw in (("hard", {}), ("soft", {"rng": rng}), ("interpolated", {"lam": 1.0})):
        mapped = tr.apply(toks, mode, **kw)
        assert mapped.shape == toks.shape and mapped.min() >= 0 and mapped.max() < K
        assert ot_distance(hist(mapped), tgt, Cn) < d_before, mode
    assert js_divergence(hist(tr.map_soft(toks, rng)), tgt) < js_divergence(hist(toks), tgt)


def test_interpolation_lam_zero_is_identity_and_monotone():
    K = 8
    src = np.array([0.4, 0.3, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])
    tr = fit_unit_transport(src, src[::-1].copy(), _line_centroids(K), "sqeuclidean", eps=0.01)
    toks = np.arange(K)
    assert change_rate(toks, tr.map_interpolated(toks, 0.0)) == 0.0
    r = [change_rate(toks, tr.map_interpolated(toks, lam)) for lam in (0.0, 0.5, 1.0)]
    assert r[0] <= r[1] <= r[2]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
