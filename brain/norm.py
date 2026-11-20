"""Running normalisers (design 4.4.1 and 4.4.2).

TrackMeanVar is Welford's single-pass estimator exactly as written in the
paper's algorithm: n += 1; mu_new = mu + (x - mu) / n; p += (x - mu)(x -
mu_new); var = p / (n - 1) for n >= 2 else 1.  The observation normaliser
updates its statistics with the observation before that observation is
used, and every observation is normalised exactly once.  The reward
scaler divides the reward by the running standard deviation of the
discounted reward trace U, which is never reset.
"""

from __future__ import annotations

import math

import numpy as np


class TrackMeanVar:
    """Running mean and unbiased sample variance for one stream."""

    def __init__(self, shape=(), eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.n = 0
        self.mu = np.zeros(shape)
        self.p = np.zeros(shape)

    @property
    def var(self):
        if self.n >= 2:
            return self.p / (self.n - 1)
        return np.ones_like(self.p)

    def update(self, x):
        x = np.asarray(x, dtype=float)
        self.n += 1
        delta = x - self.mu
        self.mu = self.mu + delta / self.n
        self.p = self.p + delta * (x - self.mu)
        return (x - self.mu) / np.sqrt(self.var + self.eps)


class ObservationNormalizer(TrackMeanVar):
    """Per-coordinate standardisation of the observation vector."""


class RewardScaler:
    """U <- gamma * U + r; scale r by the running std of U."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.stats = TrackMeanVar((), eps=self.eps)
        self.trace = 0.0

    def scale(self, reward: float, gamma: float) -> float:
        reward = float(reward)
        self.trace = float(gamma) * self.trace + reward
        self.stats.update(self.trace)
        return reward / math.sqrt(float(self.stats.var) + self.eps)
