"""
Решатели OT на torch (без зависимости от POT).

sinkhorn            — энтропийно-регуляризованный OT, лог-домен (устойчив к малым eps).
sinkhorn_unbalanced — OT с мягкими маргинальными ограничениями (KL-штраф tau):
                      масса может создаваться/уничтожаться (разная длительность/ритм).
sinkhorn_divergence — S_eps(a,b) = OT_eps(a,b) - (OT_eps(a,a)+OT_eps(b,b))/2,
                      неотрицательна и равна нулю при a == b.
sliced_wasserstein  — быстрое приближение W_p для облаков точек произвольной длины.

Соглашения: a (n,), b (m,) — неотрицательные веса, сумма 1 (для balanced);
C (n, m) — стоимость; eps — абсолютная (используйте cost.normalize_cost,
чтобы eps был безразмерным).
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SinkhornResult:
    plan: torch.Tensor      # (n, m) план переноса
    cost: torch.Tensor      # <plan, C> — транспортная стоимость (без энтропии)
    reg_cost: torch.Tensor  # <plan, f+g> — регуляризованная стоимость (для дивергенции)
    f: torch.Tensor         # потенциал (n,)
    g: torch.Tensor         # потенциал (m,)
    n_iter: int
    err: float              # ошибка строчной маргинали (L1)


def _log(w: torch.Tensor) -> torch.Tensor:
    return torch.where(w > 0, w.clamp_min(1e-300).log(), torch.full_like(w, float("-inf")))


def _solve(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    eps: float,
    n_iter: int,
    tol: float,
    damping: float,
) -> SinkhornResult:
    """Общий цикл; damping = 1 для balanced, tau/(tau+eps) для unbalanced."""
    a, b, C = a.to(C.dtype), b.to(C.dtype), C
    log_a, log_b = _log(a), _log(b)
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    # ε-scaling: грубые eps → целевой eps с тёплым стартом потенциалов
    # (сильно ускоряет сходимость при малых eps)
    # Только для balanced: у unbalanced damping зависит от eps.
    if damping == 1.0:
        for e in [eps * 2.0 ** k for k in range(6, 0, -1)]:
            for _ in range(20):
                f = -e * torch.logsumexp(log_b[None, :] + (g[None, :] - C) / e, dim=1)
                g = -e * torch.logsumexp(log_a[:, None] + (f[:, None] - C) / e, dim=0)

    err = float("inf")
    it = 0
    for it in range(1, n_iter + 1):
        f = -damping * eps * torch.logsumexp(log_b[None, :] + (g[None, :] - C) / eps, dim=1)
        g = -damping * eps * torch.logsumexp(log_a[:, None] + (f[:, None] - C) / eps, dim=0)
        if it % 10 == 0 or it == n_iter:
            plan = torch.exp(log_a[:, None] + log_b[None, :] + (f[:, None] + g[None, :] - C) / eps)
            err = (plan.sum(1) - a).abs().sum().item()
            if damping == 1.0 and err < tol:
                break
            if damping != 1.0:
                # unbalanced: маргиналь не обязана совпадать с a; смотрим на сдвиг потенциалов
                err = float("nan")

    plan = torch.exp(log_a[:, None] + log_b[None, :] + (f[:, None] + g[None, :] - C) / eps)
    cost = (plan * C).sum()
    reg_cost = (plan.sum(1) * f).sum() + (plan.sum(0) * g).sum()
    return SinkhornResult(plan, cost, reg_cost, f, g, it, err)


def sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 500,
    tol: float = 1e-8,
) -> SinkhornResult:
    """Энтропийный OT в лог-домене. Возвращает SinkhornResult."""
    return _solve(a, b, C, eps, n_iter, tol, damping=1.0)


def sinkhorn_unbalanced(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    eps: float = 0.05,
    tau: float = 1.0,
    n_iter: int = 500,
) -> SinkhornResult:
    """
    Несбалансированный OT с KL-штрафом tau на отклонение маргиналей.

    tau → ∞ даёт сбалансированный OT; малые tau позволяют «не переносить»
    массу, плохо сопоставимую по стоимости.
    """
    damping = tau / (tau + eps)
    return _solve(a, b, C, eps, n_iter, tol=0.0, damping=damping)


def sinkhorn_divergence(
    a: torch.Tensor,
    b: torch.Tensor,
    C_ab: torch.Tensor,
    C_aa: torch.Tensor,
    C_bb: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 500,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Sinkhorn divergence; требует три матрицы стоимости (a-b, a-a, b-b)."""
    ab = sinkhorn(a, b, C_ab, eps, n_iter, tol).reg_cost
    aa = sinkhorn(a, a, C_aa, eps, n_iter, tol).reg_cost
    bb = sinkhorn(b, b, C_bb, eps, n_iter, tol).reg_cost
    return ab - 0.5 * (aa + bb)


def sliced_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    n_proj: int = 128,
    p: int = 2,
    n_quantiles: int = 256,
    seed: Optional[int] = 0,
) -> torch.Tensor:
    """
    Sliced W_p между облаками x (n, D) и y (m, D) с равными весами.
    Длины могут различаться: 1D-распределения сравниваются по квантилям.
    """
    x, y = x.to(torch.float64), y.to(torch.float64)
    gen = torch.Generator(device=x.device).manual_seed(seed) if seed is not None else None
    theta = torch.randn(x.shape[1], n_proj, generator=gen, dtype=x.dtype, device=x.device)
    theta = theta / theta.norm(dim=0, keepdim=True)
    px, py = x @ theta, y @ theta  # (n, P), (m, P)

    if px.shape[0] == py.shape[0]:
        qx, qy = px.sort(0).values, py.sort(0).values
    else:
        u = (torch.arange(n_quantiles, dtype=x.dtype, device=x.device) + 0.5) / n_quantiles
        qx = torch.quantile(px, u, dim=0)
        qy = torch.quantile(py, u, dim=0)
    return (qx - qy).abs().pow(p).mean().pow(1.0 / p)
