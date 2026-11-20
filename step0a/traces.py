"""Near traces in experience units (section 1.7 of the design).

Three timescales lambda_a = 5, 50, 500 experience units.  Traces update only
at anchors (m_k >= 1), with m_k the anchor's realized span:

    T_x <- exp(-m/lam) T_x + (1 - exp(-m/lam)) xbar
    T_U <- exp(-m/lam) T_U + exp(-m/(2 lam)) (I_plus - I_minus)
    eps  = 0.5 (1 - exp(-m/lam)) (I_plus + I_minus)

The BROKEN variant treats utility intensively:
    T_U_broken <- exp(-m/lam) T_U_broken + (1 - exp(-m/lam)) u_at_look
with u_at_look the instantaneous utility at the anchor look (not the integral).
"""

from __future__ import annotations

import math

import numpy as np

LAMBDAS = (5.0, 50.0, 500.0)


class NearTraces:
    def __init__(self, n_sensory: int, lambdas=LAMBDAS):
        self.lambdas = tuple(float(x) for x in lambdas)
        self.n_sensory = int(n_sensory)
        n = len(self.lambdas)
        self.Tx = np.zeros((n, self.n_sensory), dtype=np.float64)
        self.TU = np.zeros(n, dtype=np.float64)
        self.TU_broken = np.zeros(n, dtype=np.float64)
        self.eps = np.zeros(n, dtype=np.float64)

    def step(self, m: int, xbar: np.ndarray, dU: float, absU: float,
             u_at_look: float) -> None:
        """Apply one anchor update (m >= 1)."""
        xbar = np.asarray(xbar, dtype=np.float64)
        for a, lam in enumerate(self.lambdas):
            e = math.exp(-m / lam)
            em = math.exp(-m / (2.0 * lam))
            one_minus = -math.expm1(-m / lam)
            self.Tx[a] = e * self.Tx[a] + one_minus * xbar
            self.TU[a] = e * self.TU[a] + em * dU
            self.eps[a] = 0.5 * one_minus * absU
            self.TU_broken[a] = e * self.TU_broken[a] + one_minus * u_at_look

    def snapshot(self) -> dict:
        return {
            "T_x": self.Tx.copy(),
            "T_U": self.TU.copy(),
            "T_U_broken": self.TU_broken.copy(),
            "eps": self.eps.copy(),
        }


def run_traces(anchors: dict, x_sensory_at_anchor: np.ndarray,
               u_at_anchor_look: np.ndarray,
               lambdas=LAMBDAS) -> dict:
    """Run all three timescales over an anchor sequence.

    x_sensory_at_anchor: (n_anchors, n_sensory)
    u_at_anchor_look:    (n_anchors,) instantaneous utility at the look.
    Returns per-anchor histories plus final state.
    """
    m = anchors["m"]
    ip = anchors["I_plus"]
    im = anchors["I_minus"]
    n_anchor = len(m)
    n_lam = len(lambdas)
    tr = NearTraces(x_sensory_at_anchor.shape[1], lambdas)

    Tx_hist = np.empty((n_anchor, n_lam, tr.n_sensory), dtype=np.float64)
    TU_hist = np.empty((n_anchor, n_lam), dtype=np.float64)
    TUb_hist = np.empty((n_anchor, n_lam), dtype=np.float64)
    eps_hist = np.empty((n_anchor, n_lam), dtype=np.float64)

    for k in range(n_anchor):
        tr.step(int(m[k]), x_sensory_at_anchor[k], float(ip[k] - im[k]),
                float(ip[k] + im[k]), float(u_at_anchor_look[k]))
        Tx_hist[k] = tr.Tx
        TU_hist[k] = tr.TU
        TUb_hist[k] = tr.TU_broken
        eps_hist[k] = tr.eps

    return {
        "T_x_history": Tx_hist,
        "T_U_history": TU_hist,
        "T_U_broken_history": TUb_hist,
        "eps_history": eps_hist,
        "final": tr.snapshot(),
    }
