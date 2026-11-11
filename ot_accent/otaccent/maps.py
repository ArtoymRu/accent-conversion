"""Editors: maps applied to the frames of one utterance.

X is (T, d) frames, c is (T,) phone ids with -1 for silence. Silence is never
edited. Every editor interpolates  x + lam * (T(x) - x)  (displacement
interpolation, Section 3.4), so lam = 0 is the identity.
"""
import numpy as np


def shift(X, c, Delta=None, delta=None, lam=1.0, kappa=0.0):
    """Eq. (6):  x + lam * Delta_c + kappa * delta.

    Delta (K, d) is the conditional delta, delta (d,) the global one.
    lam alone leaves the speaker centre where it is; kappa alone is AccentFlow
    at frame level.
    """
    Y = np.array(X, dtype=np.float32)
    speech = c >= 0
    if Delta is not None:
        Y[speech] += lam * Delta[c[speech]]
    if delta is not None:
        Y[speech] += kappa * delta
    return Y


def affine(X, c, src, tgt, centre=0.0, lam=1.0):
    """Diagonal Gaussian transport, eq. (5) with diagonal covariances.

    src, tgt are (mean, std) pairs from stats.moments, shaped (K, d) for the
    conditional map or (1, d) for the global one. `centre` is the source
    speaker's own centre when the moments are speaker-centred, else 0.
    """
    (m_a, s_a), (m_b, s_b) = src, tgt
    Y = np.array(X, dtype=np.float32)
    speech = c >= 0
    k = c[speech] if len(m_a) > 1 else np.zeros(speech.sum(), dtype=int)
    z = Y[speech] - centre
    Y[speech] += lam * (m_b[k] + s_b[k] / s_a[k] * (z - m_a[k]) - z)
    return Y


def barycentre(X, c, pool_X, pool_c, k=4, lam=1.0, conditional=True):
    """Barycentric map: each frame becomes the mean of its k nearest target frames.

    Cosine distance, as in kNN-VC. If conditional, neighbours are searched
    among target frames of the same phone only. To use speaker-centred frames,
    centre X and pool_X before the call and add the source centre back after.
    """
    Y = np.array(X, dtype=np.float32)
    speech = np.flatnonzero(c >= 0)
    groups = [(speech[c[speech] == ph], np.flatnonzero(pool_c == ph)) for ph in np.unique(c[speech])] \
        if conditional else [(speech, np.flatnonzero(pool_c >= 0))]
    for rows, pool in groups:
        if len(pool) == 0:
            continue
        P = pool_X[pool].astype(np.float32)
        Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
        for chunk in np.array_split(rows, max(1, len(rows) // 512)):
            Q = Y[chunk]
            sim = (Q / np.linalg.norm(Q, axis=1, keepdims=True)) @ Pn.T
            kk = min(k, len(pool))
            nn = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
            Y[chunk] = Q + lam * (P[nn].mean(1) - Q)
    return Y
