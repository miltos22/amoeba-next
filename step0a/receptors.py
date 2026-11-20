"""Physical receptors (section 1.2 of the design).

Each receptor filters its source continuously in world time:

    tau_i dx_i/dt = s_i(t) - x_i(t)

integrated exactly on the world grid with s held constant over each grid
interval:  x <- s + (x - s) * exp(-h_w / tau_i).

Looks never enter this file: the receptor trajectory is a function of the
source in world time alone, so any look schedule samples the same curve.
"""

from __future__ import annotations

import math

import numpy as np

from .world_signals import H_W, Source, UtilitySource, n_grid

TAU_DEFAULT = 0.2


# --------------------------------------------------------------------------
# exact linear filter
# --------------------------------------------------------------------------

def _kernel_length(a: float, n: int) -> int:
    if a <= 0.0:
        return 1
    k = int(math.ceil(math.log(1e-16) / math.log(a)))
    return min(max(k, 1), max(n, 1))


def exact_filter(s: np.ndarray, a: float, b: float,
                 x0: float | None = None) -> np.ndarray:
    """Exact solution of x_{k+1} = a x_k + b s_k on the grid.

    Implemented as a causal convolution with the exponential kernel
    k[0] = 0, k[m] = b a^(m-1), plus the homogeneous term a^n x0.  The
    kernel is truncated where a^m < 1e-16, and the convolution is done
    blockwise with FFTs (overlap-add) so the largest rigs stay affordable.
    """
    n = len(s)
    if n == 0:
        return s.copy()
    s = np.asarray(s, dtype=np.float64)
    if x0 is None:
        x0 = float(s[0])
    K = _kernel_length(a, n)
    kern = np.zeros(K + 1, dtype=np.float64)
    if a > 0.0 and K >= 1:
        kern[1:] = b * np.power(a, np.arange(K, dtype=np.float64))
    # block size chosen so the padded FFT is small and the overlap is K
    B = 1 << 15
    nfft = 1
    while nfft < B + K:
        nfft <<= 1
    fk = np.fft.rfft(kern, nfft)
    out = np.zeros(n, dtype=np.float64)
    for start in range(0, n, B):
        end = min(n, start + B)
        blk = s[start:end]
        fb = np.fft.rfft(blk, nfft)
        conv = np.fft.irfft(fb * fk, nfft)[: (end - start) + K]
        stop = min(n, start + len(conv))
        out[start:stop] += conv[: stop - start]
    # homogeneous term a^n x0
    if x0 != 0.0:
        out[0] += x0
        if n > 1:
            out[1:] += x0 * np.power(a, np.arange(1, n, dtype=np.float64))
    return out


# --------------------------------------------------------------------------
# receptor group
# --------------------------------------------------------------------------

class Receptor:
    """One filtered sensory coordinate."""

    def __init__(self, source: Source, tau: float = TAU_DEFAULT):
        self.source = source
        self.tau = tau

    def filtered(self, n: int, h: float = H_W) -> np.ndarray:
        s = self.source.to_grid(n, h)
        a = math.exp(-h / self.tau)
        return exact_filter(s, a, 1.0 - a)

    def sample(self, idx: np.ndarray, n: int, h: float = H_W) -> np.ndarray:
        return self.filtered(n, h)[idx]


def calibrate_sigma_hat(source: Source, tau: float = TAU_DEFAULT,
                        seconds: float = 600.0, seed: int = 999,
                        h: float = H_W) -> float:
    """Unscaled MAD of the filtered source over a calibration run.

    The source is rebuilt on a 600 s grid.  For sources that already carry a
    seed the calibration uses a dedicated calibration seed (999) so that the
    scale estimate is independent of the measurement life.
    """
    n = n_grid(seconds, h)
    src = Source(kind=source.kind, seed=seed, sigma_n=source.sigma_n,
                 tau_n=source.tau_n, _tag=source._tag)
    x = Receptor(src, tau).filtered(n, h)
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def unit_interval_integrals(n: int, h: float = H_W):
    """Helper: cumulative integrals of a unit utility (not used by rigs)."""
    return UtilitySource.cumulative_integrals(np.ones(n), h)
