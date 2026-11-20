"""World-time recurrent layer with exact online sensitivities.

State ``h`` in C^n, input ``x`` in R^d, parameters ``a``, ``b`` in R^n and
``W = W_re + i W_im`` in C^{n x d}.  A step advances the state, the
matrix sensitivity ``E = dh/dW_re`` and the vector sensitivities
``ea = dh/da``, ``eb = dh/db`` at the same parameters, so any head can
turn its gradient into recurrent parameter gradients.

Section 4.2 of the design defines the mathematics; ``input_gain`` selects the zero-order-hold gain ("zoh",
exact for a piecewise-constant input) or the paper's per-step gain
("paper", exact only at dt = 1 s against the published unit).
"""

from __future__ import annotations

import math

import numpy as np


class RTULayer:
    """One complex recurrent layer; n = 0 is a pass-through of the input."""

    def __init__(self, n_units: int, n_in: int, config, rng: np.random.Generator) -> None:
        self.n_units = int(n_units)
        self.n_in = int(n_in)
        self.config = config
        n, d = self.n_units, self.n_in
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.W_re = np.zeros((n, d))
        self.W_im = np.zeros((n, d))
        if n > 0:
            taus = np.geomspace(config.tau_min_s, config.tau_max_s, n)
            self.a[:] = np.log(1.0 / taus)
            log_lo = math.log(config.period_min_s)
            log_hi = math.log(config.period_max_s)
            periods = np.exp(rng.uniform(log_lo, log_hi, n))
            self.b[:] = np.log(2.0 * math.pi / periods)
        self.reset()

    def reset(self) -> None:
        n, d = self.n_units, self.n_in
        self.h = np.zeros(n, dtype=complex)
        self.E = np.zeros((n, d), dtype=complex)
        self.ea = np.zeros(n, dtype=complex)
        self.eb = np.zeros(n, dtype=complex)
        self.last_x = None
        self.last_f = None

    @property
    def tau(self) -> np.ndarray:
        if self.n_units == 0:
            return np.zeros(0)
        return 1.0 / np.exp(self.a)

    def _coefficients(self, dt: float):
        cfg = self.config
        dt = float(dt)
        nu = np.exp(self.a)
        om = np.exp(self.b)
        s = -nu + 1j * om
        lam = np.exp(s * dt)
        lm1 = np.expm1(s * dt)
        dlam_da = -nu * dt * lam
        dlam_db = 1j * om * dt * lam
        if cfg.input_gain == "zoh":
            c = nu * lm1 / s
            dc_da = c + nu * (dlam_da * s - lm1 * (-nu)) / s**2
            dc_db = nu * (dlam_db * s - lm1 * (1j * om)) / s**2
        elif cfg.input_gain == "paper":
            c_real = np.sqrt(-np.expm1(-2.0 * nu * dt))
            c = c_real.astype(complex)
            dc_da = (nu * dt * np.exp(-2.0 * nu * dt) / c_real).astype(complex)
            dc_db = np.zeros_like(c)
        else:
            raise ValueError(f"unknown input_gain {cfg.input_gain!r}")
        return lam, c, dlam_da, dlam_db, dc_da, dc_db

    def advance(self, x, dt: float) -> np.ndarray:
        """Advance state and sensitivities with input x over world time dt."""
        x = np.asarray(x, dtype=float)
        if self.n_units == 0:
            self.last_x = x
            self.last_f = x.copy()
            return self.last_f
        lam, c, dlam_da, dlam_db, dc_da, dc_db = self._coefficients(dt)
        W = self.W_re + 1j * self.W_im
        u = W @ x
        h_prev = self.h
        self.E = lam[:, None] * self.E + c[:, None] * x[None, :]
        self.ea = dlam_da * h_prev + lam * self.ea + dc_da * u
        self.eb = dlam_db * h_prev + lam * self.eb + dc_db * u
        self.h = lam * h_prev + c * u
        self.last_x = x
        f = np.concatenate([self.h.real, self.h.imag, x])
        self.last_f = f
        return f

    def peek(self, x, dt: float) -> np.ndarray:
        """Features for x over dt from a copy of the state; nothing is committed."""
        x = np.asarray(x, dtype=float)
        if self.n_units == 0:
            return x.copy()
        lam, c, _, _, _, _ = self._coefficients(dt)
        W = self.W_re + 1j * self.W_im
        h = lam * self.h + c * (W @ x)
        return np.concatenate([h.real, h.imag, x])

    def param_grads(self, g_re: np.ndarray, g_im: np.ndarray) -> dict[str, np.ndarray]:
        """Gradients for a, b, W_re, W_im given dy/dRe h and dy/dIm h."""
        n, d = self.n_units, self.n_in
        if n == 0:
            return {
                "a": np.zeros(0),
                "b": np.zeros(0),
                "W_re": np.zeros((0, d)),
                "W_im": np.zeros((0, d)),
            }
        E = self.E
        ea = self.ea
        eb = self.eb
        return {
            "a": g_re * ea.real + g_im * ea.imag,
            "b": g_re * eb.real + g_im * eb.imag,
            "W_re": g_re[:, None] * E.real + g_im[:, None] * E.imag,
            "W_im": -g_re[:, None] * E.imag + g_im[:, None] * E.real,
        }
