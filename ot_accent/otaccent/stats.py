"""Sufficient statistics of frames per (speaker, half, phone).

Everything in E1 and E4, and every fitted map, is a function of this table.
The two halves are disjoint sets of utterances of the same speaker: products
across halves are free of within-cell noise, which is how variances are
estimated without bias.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Table:
    speakers: np.ndarray    # (S,)  names
    accents: np.ndarray     # (S,)  accent of each speaker
    n: np.ndarray           # (S, 2, K)     frames
    sx: np.ndarray          # (S, 2, K, d)  sum of frames
    sxx: np.ndarray         # (S, 2, K, d)  sum of squared frames

    def means(self, half=None) -> np.ndarray:
        """(S, K, d) mean frame of each speaker on each phone; both halves pooled if half is None."""
        n, sx = (self.n.sum(1), self.sx.sum(1)) if half is None else (self.n[:, half], self.sx[:, half])
        return sx / np.maximum(n, 1)[..., None]

    def content(self, min_frames=5) -> np.ndarray:
        """Reference content distribution p_c: pooled frame share, zero where any cell is too small."""
        p = self.n.sum((0, 1)).astype(float)
        p[(self.n < min_frames).any((0, 1))] = 0
        return p / p.sum()

    def idx(self, accent, exclude=()) -> np.ndarray:
        """Indices of the speakers of an accent, minus `exclude` (names)."""
        return np.array([i for i, (s, a) in enumerate(zip(self.speakers, self.accents))
                         if a == accent and s not in exclude])

    def subset(self, keep) -> "Table":
        """The table restricted to some speakers (indices or a boolean mask)."""
        return Table(self.speakers[keep], self.accents[keep], self.n[keep], self.sx[keep], self.sxx[keep])

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @staticmethod
    def load(path) -> "Table":
        z = np.load(path, allow_pickle=False)
        return Table(**{k: z[k] for k in z.files})


class TableBuilder:
    """Accumulates frames into a Table, one utterance at a time."""

    def __init__(self, n_phones, dim):
        self.K, self.d, self.rows = n_phones, dim, {}

    def add(self, speaker, accent, half, frames, labels):
        if speaker not in self.rows:
            self.rows[speaker] = (accent, np.zeros((2, self.K)), np.zeros((2, self.K, self.d)),
                                  np.zeros((2, self.K, self.d)))
        _, n, sx, sxx = self.rows[speaker]
        speech = labels >= 0
        x, c = frames[speech].astype(np.float64), labels[speech]
        np.add.at(n[half], c, 1)
        np.add.at(sx[half], c, x)
        np.add.at(sxx[half], c, x * x)

    def table(self) -> Table:
        names = sorted(self.rows)
        col = lambda i: np.stack([self.rows[s][i] for s in names])
        return Table(np.array(names), np.array([self.rows[s][0] for s in names]), col(1), col(2), col(3))


# ---- the estimators of the paper (Section 2, "Statistics") -------------------

def centres(means, p) -> np.ndarray:
    """(S, d) content-balanced speaker centres  x_s = sum_c p_c x_{s,c}."""
    return np.einsum("k,skd->sd", p, means)


def profiles(means, p) -> np.ndarray:
    """(S, K, d) speaker-centred phone means  x_{s,c} - x_s."""
    return means - centres(means, p)[:, None]


def global_delta(means, p, src, tgt) -> np.ndarray:
    """(d,) eq. (2): difference of the mean speaker centres of two speaker sets."""
    c = centres(means, p)
    return c[tgt].mean(0) - c[src].mean(0)


def conditional_delta(means, p, src, tgt) -> np.ndarray:
    """(K, d) eq. (3): difference of the mean speaker-centred profiles. Zero where p_c = 0."""
    r = profiles(means, p)
    return (r[tgt].mean(0) - r[src].mean(0)) * (p > 0)[:, None]


def moments(tab: Table, idx, p, centred=True, conditional=True):
    """Mean and std of the frames of speakers `idx`, per phone (K, d) or pooled (1, d).

    centred: frames are taken relative to each speaker's own centre.
    """
    n, sx, sxx = tab.n.sum(1)[idx], tab.sx.sum(1)[idx], tab.sxx.sum(1)[idx]
    if centred:                                  # sum (x-c)^2 = sxx - 2 c sx + n c^2
        c = centres(tab.means(), p)[idx][:, None]
        sxx = sxx - 2 * c * sx + n[..., None] * c * c
        sx = sx - n[..., None] * c
    n, sx, sxx = n.sum(0), sx.sum(0), sxx.sum(0)                  # pool speakers
    if not conditional:
        keep = p > 0
        n, sx, sxx = n[keep].sum(0, keepdims=True), sx[keep].sum(0, keepdims=True), sxx[keep].sum(0, keepdims=True)
    mean = sx / np.maximum(n, 1)[:, None]
    var = sxx / np.maximum(n, 1)[:, None] - mean ** 2
    return mean, np.sqrt(np.maximum(var, 1e-8))
