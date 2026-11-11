"""A world that obeys model (1) exactly, with known variance components.

Used by the tests to check every estimator against the truth, and by
`--synthetic` runs of the experiments to exercise the code without data.
"""
import numpy as np

from .stats import TableBuilder


class World:
    """x = mu_c + u_s + m_a + v_{a,c} + w_{s,c} + eps, all terms Gaussian.

    `var` gives the trace of each component: content, speaker, accent
    (the global accent term m_a), accent_x_content, speaker_x_content, residual.
    """

    def __init__(self, n_accents=6, speakers_per_accent=4, n_phones=20, dim=32, var=None, seed=0):
        self.var = {"content": 4.0, "speaker": 2.0, "accent": 0.3, "accent_x_content": 0.5,
                    "speaker_x_content": 0.5, "residual": 3.0} | (var or {})
        self.A, self.S, self.K, self.d = n_accents, n_accents * speakers_per_accent, n_phones, dim
        rng = np.random.default_rng(seed)
        draw = lambda name, *shape: rng.normal(0, np.sqrt(self.var[name] / dim), (*shape, dim))
        self.accent_of = np.repeat(np.arange(self.A), speakers_per_accent)
        self.p = np.full(self.K, 1 / self.K)
        centre = lambda t: t - t.mean(-2, keepdims=True)        # sum_c p_c t_c = 0 (p uniform)
        self.mu = draw("content", self.K)
        self.m = draw("accent", self.A)
        self.u = draw("speaker", self.S)
        self.v = centre(draw("accent_x_content", self.A, self.K))
        self.w = centre(draw("speaker_x_content", self.S, self.K))
        self.rng = rng

    def mean(self, s, c):
        """Noise-free frame of speaker s on phone(s) c."""
        a = self.accent_of[s]
        return self.mu[c] + self.u[s] + self.m[a] + self.v[a, c] + self.w[s, c]

    def frames(self, s, labels):
        """Frames of speaker s for a label sequence (silence, -1, is pure noise)."""
        X = self.rng.normal(0, np.sqrt(self.var["residual"] / self.d), (len(labels), self.d))
        speech = labels >= 0
        X[speech] += self.mean(s, labels[speech])
        return X.astype(np.float32)

    def table(self, frames_per_cell=40):
        """Table of sufficient statistics, as stats.TableBuilder would build from audio."""
        tb = TableBuilder(self.K, self.d)
        labels = np.repeat(np.arange(self.K), frames_per_cell)
        for s in range(self.S):
            for half in (0, 1):
                tb.add(f"spk{s:03d}", f"acc{self.accent_of[s]}", half, self.frames(s, labels), labels)
        return tb.table()
