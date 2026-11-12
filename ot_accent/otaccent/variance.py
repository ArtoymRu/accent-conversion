"""Where accent lives: variance components of model (1), and the 1/S law.

Model:  x = mu_c + u_s + v_{a(s),c} + w_{s,c} + eps.

Each component is a difference of three averaged inner products:
  self : the same speaker, two disjoint halves of utterances
  same : two different speakers of the same accent
  diff : two speakers of different accents
taken once for speaker centres and once for speaker-centred phone profiles.
Different speakers and different halves have independent noise, and any offset
common to all speakers cancels in the differences, so the estimates are
unbiased under a random-effects reading of the model.
"""
import numpy as np

from .stats import centres, profiles, conditional_delta, global_delta


def _self_same_diff(a_half0, a_half1, a_pooled, accents):
    """Mean inner products <a_s, a_s'>; a_* are (S, m) arrays."""
    S = len(accents)
    gram = a_pooled @ a_pooled.T
    same_acc = accents[:, None] == accents[None, :]
    off = ~np.eye(S, dtype=bool)
    return (np.einsum("sm,sm->s", a_half0, a_half1).mean(),
            gram[same_acc & off].mean(),
            gram[~same_acc].mean())


def components(tab, p=None, accents=None) -> dict:
    """Trace of each variance component, per frame, under content distribution p."""
    p = tab.content() if p is None else p
    accents = tab.accents if accents is None else accents
    w = np.sqrt(p)[None, :, None]                       # p-weighted inner product on profiles
    flat = lambda r: (r * w).reshape(len(r), -1)
    m0, m1, m = tab.means(0), tab.means(1), tab.means()
    c_self, c_same, c_diff = _self_same_diff(centres(m0, p), centres(m1, p), centres(m, p), accents)
    r_self, r_same, r_diff = _self_same_diff(flat(profiles(m0, p)), flat(profiles(m1, p)),
                                             flat(profiles(m, p)), accents)
    n = tab.n.sum(1)
    second = (p * (tab.sxx.sum((1, 3)) / np.maximum(n, 1)).mean(0)).sum()     # E ||x||^2
    return {
        "content": r_diff,
        "speaker": c_self - c_same,
        "accent (global)": c_same - c_diff,
        "speaker x content": r_self - r_same,
        "accent x content": r_same - r_diff,
        "residual": second - c_self - r_self,
        "total": second - c_diff,
    }


def sampling_noise(tab, p=None) -> dict:
    """Variance of a speaker's centre and profile that is due to having finitely many frames.

    Read off the disagreement of the two halves: Var(pooled mean) = |half0 - half1|^2 / 4.
    Negligible with a full corpus; it dominates when cells hold a handful of frames.
    """
    p = tab.content() if p is None else p
    m0, m1 = tab.means(0), tab.means(1)
    dc = centres(m0, p) - centres(m1, p)
    dr = (profiles(m0, p) - profiles(m1, p)) * np.sqrt(p)[None, :, None]
    return {"speaker": (dc ** 2).sum(1).mean() / 4, "speaker x content": (dr ** 2).sum((1, 2)).mean() / 4}


def permutation_null(tab, p=None, n_perm=200, seed=0) -> list[dict]:
    """Components with accent labels shuffled across speakers: the placebo level of the accent terms."""
    rng = np.random.default_rng(seed)
    p = tab.content() if p is None else p
    return [components(tab, p, rng.permutation(tab.accents)) for _ in range(n_perm)]


def leave_one_speaker_out(tab, p=None) -> list[dict]:
    """Components recomputed with each speaker removed in turn.

    With four speakers per accent no confidence interval deserves the name. The
    spread of these values shows how much an estimate hangs on a single speaker.
    """
    p = tab.content() if p is None else p
    every = np.arange(len(tab.speakers))
    return [components(tab.subset(every != s), p) for s in every]


def delta_error(tab, src_accent, tgt_accent, S, p=None, n_draws=200, seed=0) -> dict:
    """Error and signal of deltas estimated from S speakers per accent (Proposition 1).

    Two estimates are made from disjoint speaker sets. Their half squared
    distance is the error of one estimate; their inner product is the squared
    norm of the true delta. Needs at least 2S speakers per accent.
    """
    rng = np.random.default_rng(seed)
    p = tab.content() if p is None else p
    m, w = tab.means(), np.sqrt(p)[:, None]
    out = {"global error": [], "global signal": [], "conditional error": [], "conditional signal": []}
    for _ in range(n_draws):
        a, b = rng.permutation(tab.idx(src_accent)), rng.permutation(tab.idx(tgt_accent))
        g1, g2 = global_delta(m, p, a[:S], b[:S]), global_delta(m, p, a[S:2 * S], b[S:2 * S])
        d1 = conditional_delta(m, p, a[:S], b[:S]) * w
        d2 = conditional_delta(m, p, a[S:2 * S], b[S:2 * S]) * w
        out["global error"].append(0.5 * ((g1 - g2) ** 2).sum())
        out["global signal"].append((g1 * g2).sum())
        out["conditional error"].append(0.5 * ((d1 - d2) ** 2).sum())
        out["conditional signal"].append((d1 * d2).sum())
    return {k: float(np.mean(v)) for k, v in out.items()}
