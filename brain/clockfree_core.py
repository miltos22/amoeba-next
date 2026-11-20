"""The clock-free core: one trace, one weight matrix, one exposure store.

The rule generalises to n items and one life, plus the flow step
kept alongside the older accumulation step for comparison. No world
time, look count or interval length enters
anywhere: a look receives only the holding amounts since the last look.

State: trace e (n), weights W (n x n), exposure u (n).  For each look,
with ``acc`` the n non-negative holding amounts since the previous look:

    a   = sum(acc)                       (0: nothing happens at all)
    dec = exp(-a / TA)
    mf  = TA * (1 - dec) / a             (mean trace factor)
    eb  = e * mf + (acc / a) * TA * (1 - mf)
    du  = exp(-a / TU)
    u   = u * du + (1 - du) * eb**2      (u_new, used by both steps)

    step="flow" (default, T5):
        eps = acc / a - eb @ W
        eta = A0 / (KAP + u)
        k   = sum(eta * eb**2)
        g   = -expm1(-k a) / k           (g = a when k a < 1e-12)
        W  += outer(eta * eb, eps) * g

    step="new" (the design's old step, kept for comparison):
        d   = acc - (eb @ W) * a
        W  += outer((A0 / (KAP + u)) * eb, d)

    e   = e * dec + acc
"""

from __future__ import annotations

import math

import numpy as np


class ClockFreeCore:
    """The n-item clock-free core.  One instance is one life."""

    def __init__(
        self,
        n_items: int,
        ta: float,
        tu: float,
        a0: float = 0.05,
        kap: float = 1.0,
        step: str = "flow",
    ) -> None:
        if n_items < 1:
            raise ValueError("n_items must be >= 1")
        if not (math.isfinite(ta) and ta > 0.0):
            raise ValueError("ta must be finite and > 0")
        if not (math.isfinite(tu) and tu > 0.0):
            raise ValueError("tu must be finite and > 0")
        if not (math.isfinite(a0) and a0 >= 0.0):
            raise ValueError("a0 must be finite and >= 0")
        if not (math.isfinite(kap) and kap > 0.0):
            raise ValueError("kap must be finite and > 0")
        if step not in ("flow", "new"):
            raise ValueError(f"step must be 'flow' or 'new', got {step!r}")
        self.n = int(n_items)
        self.ta = float(ta)
        self.tu = float(tu)
        self.a0 = float(a0)
        self.kap = float(kap)
        self.step = step
        self.e = np.zeros(self.n)
        self.W = np.zeros((self.n, self.n))
        self.u = np.zeros(self.n)
        self.looks = 0

    def look(self, acc) -> None:
        """One look; ``acc`` is the n holding amounts since the last look."""
        amounts = np.asarray(acc, dtype=float)
        if amounts.shape != (self.n,):
            raise ValueError(f"acc must have shape ({self.n},), got {amounts.shape}")
        if np.any(amounts < 0.0):
            raise ValueError("core input amounts must be >= 0")
        a = float(amounts.sum())
        if a == 0.0:
            return
        dec = math.exp(-a / self.ta)
        mf = self.ta * (1.0 - dec) / a
        eb = self.e * mf + (amounts / a) * self.ta * (1.0 - mf)
        du = math.exp(-a / self.tu)
        u_new = self.u * du + (1.0 - du) * eb * eb
        if self.step == "flow":
            eps = amounts / a - eb @ self.W
            eta = self.a0 / (self.kap + u_new)
            k = float(np.sum(eta * eb * eb))
            if k * a < 1e-12:
                g = a
            else:
                g = -math.expm1(-k * a) / k
            self.W += np.outer(eta * eb, eps) * g
        else:
            d = amounts - (eb @ self.W) * a
            self.W += np.outer((self.a0 / (self.kap + u_new)) * eb, d)
        self.u = u_new
        self.e = self.e * dec + amounts
        self.looks += 1
        if not (
            np.all(np.isfinite(self.e))
            and np.all(np.isfinite(self.W))
            and np.all(np.isfinite(self.u))
        ):
            raise FloatingPointError(
                f"core state is not finite after look {self.looks} "
                f"(a={a!r}, max|W|={float(np.max(np.abs(self.W)))!r})"
            )
