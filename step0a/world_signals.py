"""World grid, look schedules and sensory/utility sources (section 1.1 of the design).

All quantities are in WORLD SECONDS.  The world is defined on a fine grid
h_w = 1/150 s; 150 is a common multiple of the look rates 2, 5, 15 and 50 Hz,
so every regular look time lies exactly on the grid.  The brain reads at
"looks"; there is no tick in any experience-time formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

H_W = 1.0 / 150.0
LOOK_RATES = (2, 5, 15, 50)


# --------------------------------------------------------------------------
# grid and look schedules
# --------------------------------------------------------------------------

def n_grid(life: float, h: float = H_W) -> int:
    """Number of grid points covering [0, life] inclusive."""
    return int(round(life / h)) + 1


def grid_times(n: int, h: float = H_W) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * h


def regular_look_indices(rate: float, n: int, h: float = H_W) -> np.ndarray:
    """Grid indices of a regular look schedule at ``rate`` looks/s."""
    dt = 1.0 / rate
    step = int(round(dt / h))
    if abs(step * h - dt) > 1e-12:
        raise ValueError(f"look rate {rate} does not lie on the grid")
    return np.arange(0, n, step, dtype=np.int64)


def irregular_look_indices(n: int, seed: int, h: float = H_W,
                           lo: float = 0.02, hi: float = 0.5) -> np.ndarray:
    """I.i.d. uniform [lo, hi] second intervals, rounded to the grid, seeded."""
    rng = np.random.default_rng(seed)
    spans: list[int] = []
    total = 0
    while total < n - 1:
        u = float(rng.uniform(lo, hi))
        k = max(1, int(round(u / h)))
        spans.append(k)
        total += k
    idx = np.empty(len(spans) + 1, dtype=np.int64)
    idx[0] = 0
    np.cumsum(spans, out=idx[1:])
    return idx[idx < n]


def look_indices(schedule: str, n: int, seed: int = 0,
                 h: float = H_W) -> np.ndarray:
    """Resolve a schedule name: '2', '5', '15', '50' or 'irregular'."""
    if schedule == "irregular":
        return irregular_look_indices(n, seed, h)
    return regular_look_indices(float(schedule), n, h)


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def _ou_on_grid(n: int, sigma: float, tau: float, seed: int,
                h: float = H_W) -> np.ndarray:
    """Exact stationary OU process on the grid.

    x_{k+1} = a x_k + sigma sqrt(1 - a^2) z_k,  a = exp(-h/tau).
    Computed with the exact linear filter so the result is a deterministic
    function of the seed.
    """
    if sigma <= 0.0:
        return np.zeros(n, dtype=np.float64)
    a = math.exp(-h / tau)
    b = 1.0 - a
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    drive = z * (sigma * math.sqrt(1.0 - a * a) / b)
    from .receptors import exact_filter
    return exact_filter(drive, a, b, x0=float(rng.standard_normal()) * sigma)


@dataclass
class Source:
    """A seeded source s_i(t) on the world grid.

    kind: 'silent' | 'steps' | 'bursts' | 'ramp'
    sigma_n: stationary SD of optional additive OU noise (time constant 0.5 s).
    """

    kind: str
    seed: int = 0
    sigma_n: float = 0.0
    tau_n: float = 0.5
    _tag: int = 0

    def raw_grid(self, n: int, h: float = H_W) -> np.ndarray | None:
        """Base source on the grid; None means exactly zero."""
        if self.kind == "silent":
            return None
        t = grid_times(n, h)
        if self.kind == "ramp":
            return np.sin(2.0 * math.pi * t / 300.0)
        if self.kind == "steps":
            rng = np.random.default_rng(self.seed * 1009 + 11)
            boundaries = [0.0]
            levels = [float(rng.normal())]
            tt = 0.0
            while True:
                tt += float(rng.exponential(20.0))
                if tt > t[-1] + 1.0:
                    break
                boundaries.append(tt)
                levels.append(float(rng.normal()))
            boundaries.append(t[-1] + 1.0)
            b = np.asarray(boundaries)
            lv = np.asarray(levels)
            seg = np.searchsorted(b, t, side="right") - 1
            seg = np.clip(seg, 0, len(lv) - 1)
            return lv[seg]
        if self.kind == "bursts":
            rng = np.random.default_rng(self.seed * 1009 + 23)
            out = np.zeros(n, dtype=np.float64)
            tt = 0.0
            dur = 0.3
            while True:
                tt += float(rng.exponential(30.0))
                if tt > t[-1]:
                    break
                i0 = int(round(tt / h))
                i1 = int(round((tt + dur) / h))
                if i0 >= n:
                    break
                out[i0:min(i1, n)] = 3.0
            return out
        raise ValueError(f"unknown source kind {self.kind!r}")

    def grid(self, n: int, h: float = H_W) -> np.ndarray | None:
        base = self.raw_grid(n, h)
        if self.sigma_n <= 0.0:
            return base
        noise = _ou_on_grid(n, self.sigma_n, self.tau_n,
                            self.seed * 1009 + 37 + self._tag, h)
        if base is None:
            return noise
        return base + noise

    def to_grid(self, n: int, h: float = H_W) -> np.ndarray:
        """Always materialised (silent -> zeros); for tests and calibration."""
        g = self.grid(n, h)
        if g is None:
            return np.zeros(n, dtype=np.float64)
        return g


def scaled_source(src: Source, factor: float) -> Source:
    """A source whose signal and noise are both multiplied by ``factor``."""
    return Source(kind=src.kind, seed=src.seed, sigma_n=src.sigma_n * factor,
                  tau_n=src.tau_n, _tag=src._tag)


# --------------------------------------------------------------------------
# utility source and its integrals
# --------------------------------------------------------------------------

@dataclass
class UtilitySource:
    """Piecewise-constant utility u(t) on the grid.

    events: list of (t_start, t_end, value) in world seconds; zero elsewhere.
    """

    events: list[tuple[float, float, float]] = field(default_factory=list)

    def grid(self, n: int, h: float = H_W) -> np.ndarray:
        t = grid_times(n, h)
        u = np.zeros(n, dtype=np.float64)
        for t0, t1, v in self.events:
            mask = (t >= t0) & (t < t1)
            u[mask] = v
        return u

    @staticmethod
    def cumulative_integrals(u: np.ndarray, h: float = H_W):
        """Rectangle-rule cumulative integrals of max(u,0) and max(-u,0).

        Returns arrays (cum_plus, cum_minus) with cum[0] = 0 and
        cum[k] = sum over grid intervals [j, j+1), j < k.
        """
        pos = np.maximum(u, 0.0)
        neg = np.maximum(-u, 0.0)
        cp = np.zeros(len(u) + 1, dtype=np.float64)
        cm = np.zeros(len(u) + 1, dtype=np.float64)
        np.cumsum(pos * h, out=cp[1:])
        np.cumsum(neg * h, out=cm[1:])
        return cp, cm

    @staticmethod
    def interval_integrals(cp: np.ndarray, cm: np.ndarray,
                           idx: np.ndarray):
        """Per-look integrals over the intervals between consecutive looks.

        Look k covers grid indices (idx[k-1], idx[k]]; look 0 covers [0, 0]
        (zero length).
        """
        ip = np.zeros(len(idx), dtype=np.float64)
        im = np.zeros(len(idx), dtype=np.float64)
        ip[1:] = cp[idx[1:]] - cp[idx[:-1]]
        im[1:] = cm[idx[1:]] - cm[idx[:-1]]
        return ip, im
