"""Endogenous pacemaker bank (section 1.3 of the design).

K_p = 3 slow incommensurate oscillators with periods 7.3, 31.1 and 137.9 s.
Each is read as the phase pair (sin theta_j, cos theta_j).  P_raw is the
cumulative sum of the positive unwrapped observed phase increments, in
cycles.  Phases are integrated on the world grid; in the NOISY regime each
rate carries an OU fluctuation (SD 0.05, time constant 60 s), in CLEAN the
rate is fixed.
"""

from __future__ import annotations

import math

import numpy as np

from .world_signals import H_W, _ou_on_grid, grid_times

PERIODS = (7.3, 31.1, 137.9)
K_P = 3
ETA_SD = 0.05
ETA_TAU = 60.0


class Pacemaker:
    def __init__(self, noisy: bool = False, seed: int = 0):
        self.noisy = noisy
        self.seed = seed
        self._phase_cache: dict[tuple[int, float], np.ndarray] = {}

    # ---------------------------------------------------------------- phases
    def phases(self, n: int, h: float = H_W) -> np.ndarray:
        """Continuous phase theta_j at every grid point; shape (K_P, n)."""
        key = (int(n), float(h))
        cached = self._phase_cache.get(key)
        if cached is not None:
            return cached
        t = grid_times(n, h)
        out = np.empty((K_P, n), dtype=np.float64)
        for j, period in enumerate(PERIODS):
            if self.noisy:
                eta = _ou_on_grid(n, ETA_SD, ETA_TAU, self.seed * 131 + 7 + j, h)
                integ = np.empty(n, dtype=np.float64)
                integ[0] = 0.0
                if n > 1:
                    integ[1:] = np.cumsum(1.0 + eta[:-1]) * h
            else:
                integ = t.copy()
            out[j] = (2.0 * math.pi / period) * integ
        self._phase_cache[key] = out
        return out

    def pair_stream(self, j: int, n: int, h: float = H_W):
        """Observed (sin theta_j, cos theta_j) at every grid point."""
        period = PERIODS[j]
        if self.noisy:
            eta = _ou_on_grid(n, ETA_SD, ETA_TAU, self.seed * 131 + 7 + j, h)
            integ = np.empty(n, dtype=np.float64)
            integ[0] = 0.0
            if n > 1:
                integ[1:] = np.cumsum(1.0 + eta[:-1]) * h
        else:
            integ = grid_times(n, h)
        integ *= (2.0 * math.pi / period)
        return np.sin(integ), np.cos(integ)

    def pairs_at(self, ph: np.ndarray, idx: np.ndarray):
        """Observed (sin, cos) pairs at look indices; each shape (K_P, L)."""
        theta = ph[:, idx]
        return np.sin(theta), np.cos(theta)

    # ------------------------------------------------------------- increments
    @staticmethod
    def dtheta(ph: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Observed unwrapped increments between consecutive looks, (K_P, L).

        The first look has no previous look and gets increment 0.  Every
        other increment is wrapped to (-pi, pi]; the anti-aliasing margin
        (shortest period 7.3 s > 14 x slowest interval 0.5 s) keeps the true
        increment strictly inside that interval.
        """
        theta = ph[:, idx]
        d = np.zeros_like(theta)
        if theta.shape[1] > 1:
            raw = np.diff(theta, axis=1)
            d[:, 1:] = (raw + math.pi) % (2.0 * math.pi) - math.pi
        return d

    def assert_positive(self, d: np.ndarray) -> None:
        if d.shape[1] > 1:
            assert np.all(d[:, 1:] > 0.0), "non-positive pacemaker increment"

    @staticmethod
    def p_raw_increments(d: np.ndarray) -> np.ndarray:
        """Per-look increment of P_raw in cycles (sum over oscillators)."""
        return d.sum(axis=0) / (2.0 * math.pi)

    @staticmethod
    def p_raw(d: np.ndarray) -> np.ndarray:
        inc = Pacemaker.p_raw_increments(d)
        return np.cumsum(inc)
