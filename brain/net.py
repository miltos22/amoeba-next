"""Heads over the recurrent features, with hand-written backward passes.

LayerNorm (no learned scale or shift, epsilon from the configuration),
LeakyReLU, and one Gaussian per actuator item.  The policy network and the
value network are separate networks, each with its own RTULayer.  Every
linear layer of one network is filled by SparseInit (design 4.4.4).
"""

from __future__ import annotations

import math

import numpy as np

from .rtu import RTULayer

LOG_2PI = math.log(2.0 * math.pi)


def sparse_init(layers, sparsity: float, rng: np.random.Generator) -> None:
    """SparseInit (design 4.4.4) over ``layers`` = [(name, array, fan_in),...]."""
    sizes = [int(arr.size) for _, arr, _ in layers]
    phis = [fin + (int(arr.shape[0]) if arr.ndim > 1 else 1) for _, arr, fin in layers]
    budget = (1.0 - float(sparsity)) * float(sum(sizes))
    chosen: set[int] = set()
    eps = 0.0
    while True:
        remaining = [i for i in range(len(layers)) if i not in chosen]
        if not remaining:
            break
        rem_n = budget - float(sum(sizes[i] for i in chosen))
        rem_phi = float(sum(phis[i] for i in remaining))
        eps = rem_n / rem_phi
        new = {i for i in remaining if eps * phis[i] >= sizes[i]}
        if not new:
            break
        chosen |= new
    for i, (_, arr, m) in enumerate(layers):
        if sizes[i] == 0:
            continue
        density = min(1.0, eps * phis[i] / sizes[i])
        if density < 0.0:
            density = 0.0
        bound = 1.0 / math.sqrt(m)
        arr[...] = rng.uniform(-bound, bound, size=arr.shape)
        k = int(math.ceil((1.0 - density) * m))
        k = max(0, min(int(m), k))
        flat = arr.reshape(-1, m)
        if k:
            for row in range(flat.shape[0]):
                cols = rng.choice(m, size=k, replace=False)
                flat[row, cols] = 0.0


def layer_norm(z: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    count = z.size
    centred = z - z.sum() / count
    var = float((centred * centred).sum()) / count
    inv = 1.0 / math.sqrt(var + eps)
    return centred * inv, inv


def layer_norm_backward(g: np.ndarray, nhat: np.ndarray, inv: float) -> np.ndarray:
    count = g.size
    centred = g - g.sum() / count
    slope = (g * nhat).sum() / count
    return inv * (centred - nhat * slope)


def leaky_relu(z: np.ndarray, slope: float) -> np.ndarray:
    return np.where(z > 0.0, z, slope * z)


def leaky_relu_grad(z: np.ndarray, slope: float) -> np.ndarray:
    return np.where(z > 0.0, 1.0, slope)


def softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class _Trunk:
    """A1 -> LayerNorm -> LeakyReLU -> A2 -> LayerNorm -> LeakyReLU."""

    def __init__(self, n_in: int, config, rng: np.random.Generator) -> None:
        self.config = config
        h = int(config.hidden)
        self.n_in = int(n_in)
        self.A1 = np.zeros((h, n_in))
        self.b1 = np.zeros(h)
        self.A2 = np.zeros((h, h))
        self.b2 = np.zeros(h)
        self._sparse_layers = [
            ("A1", self.A1, n_in),
            ("A2", self.A2, h),
        ]
        sparse_init(self._sparse_layers, config.sparsity, rng)
        self.cache = None

    def forward(self, f: np.ndarray) -> np.ndarray:
        cfg = self.config
        z1 = self.A1 @ f + self.b1
        n1, inv1 = layer_norm(z1, cfg.ln_eps)
        q1 = leaky_relu(n1, cfg.leaky_slope)
        z2 = self.A2 @ q1 + self.b2
        n2, inv2 = layer_norm(z2, cfg.ln_eps)
        q2 = leaky_relu(n2, cfg.leaky_slope)
        self.cache = (f, n1, inv1, q1, n2, inv2, q2)
        return q2

    def backward(self, g_q2: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        f, n1, inv1, q1, n2, inv2, q2 = self.cache
        slope = self.config.leaky_slope
        g_z2 = layer_norm_backward(g_q2 * leaky_relu_grad(n2, slope), n2, inv2)
        g_A2 = g_z2[:, None] * q1[None, :]
        g_q1 = self.A2.T @ g_z2
        g_z1 = layer_norm_backward(g_q1 * leaky_relu_grad(n1, slope), n1, inv1)
        g_A1 = g_z1[:, None] * f[None, :]
        g_f = self.A1.T @ g_z1
        grads = {"A1": g_A1, "b1": g_z1, "A2": g_A2, "b2": g_z2}
        return grads, g_f


class ValueNet:
    """v = a3 . q2 + b3 with its own recurrent layer."""

    def __init__(self, n_obs: int, config, rng: np.random.Generator) -> None:
        self.config = config
        self.n_units = int(config.n_units)
        self.rtu = RTULayer(config.n_units, n_obs, config, rng)
        d_in = 2 * self.n_units + int(n_obs)
        self.trunk = _Trunk(d_in, config, rng)
        h = int(config.hidden)
        self.a3 = np.zeros(h)
        self.b3 = np.zeros(())
        sparse_init([("a3", self.a3, h)], config.sparsity, rng)
        self.params = {
            "a": self.rtu.a,
            "b": self.rtu.b,
            "W_re": self.rtu.W_re,
            "W_im": self.rtu.W_im,
            "A1": self.trunk.A1,
            "b1": self.trunk.b1,
            "A2": self.trunk.A2,
            "b2": self.trunk.b2,
            "a3": self.a3,
            "b3": self.b3,
        }

    def advance(self, x: np.ndarray, dt: float) -> np.ndarray:
        return self.rtu.advance(x, dt)

    def forward(self, f: np.ndarray) -> float:
        q2 = self.trunk.forward(f)
        return float(self.a3 @ q2 + self.b3)

    def peek(self, x: np.ndarray, dt: float) -> float:
        f = self.rtu.peek(x, dt)
        z1 = self.trunk.A1 @ f + self.trunk.b1
        n1, _ = layer_norm(z1, self.config.ln_eps)
        q1 = leaky_relu(n1, self.config.leaky_slope)
        z2 = self.trunk.A2 @ q1 + self.trunk.b2
        n2, _ = layer_norm(z2, self.config.ln_eps)
        q2 = leaky_relu(n2, self.config.leaky_slope)
        return float(self.a3 @ q2 + self.b3)

    def backward(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        q2 = self.trunk.cache[-1]
        grads, g_f = self.trunk.backward(self.a3.copy())
        grads["a3"] = q2.copy()
        grads["b3"] = np.array(1.0)
        n = self.n_units
        grads.update(self.rtu.param_grads(g_f[:n], g_f[n : 2 * n]))
        return grads, g_f


class PolicyNet:
    """mu = A_mu q2 + b_mu; sigma = softplus(A_sg q2 + b_sg), per actuator."""

    def __init__(self, n_obs: int, n_actions: int, config, rng: np.random.Generator) -> None:
        self.config = config
        self.n_units = int(config.n_units)
        self.n_actions = int(n_actions)
        self.rtu = RTULayer(config.n_units, n_obs, config, rng)
        d_in = 2 * self.n_units + int(n_obs)
        self.trunk = _Trunk(d_in, config, rng)
        h = int(config.hidden)
        self.A_mu = np.zeros((self.n_actions, h))
        self.b_mu = np.zeros(self.n_actions)
        self.A_sg = np.zeros((self.n_actions, h))
        self.b_sg = np.zeros(self.n_actions)
        sparse_init(
            [
                ("A_mu", self.A_mu, h),
                ("A_sg", self.A_sg, h),
            ],
            config.sparsity,
            rng,
        )
        self.params = {
            "a": self.rtu.a,
            "b": self.rtu.b,
            "W_re": self.rtu.W_re,
            "W_im": self.rtu.W_im,
            "A1": self.trunk.A1,
            "b1": self.trunk.b1,
            "A2": self.trunk.A2,
            "b2": self.trunk.b2,
            "A_mu": self.A_mu,
            "b_mu": self.b_mu,
            "A_sg": self.A_sg,
            "b_sg": self.b_sg,
        }
        self.cache = None

    def advance(self, x: np.ndarray, dt: float) -> np.ndarray:
        return self.rtu.advance(x, dt)

    def forward(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q2 = self.trunk.forward(f)
        mu = self.A_mu @ q2 + self.b_mu
        s_raw = self.A_sg @ q2 + self.b_sg
        sigma = softplus(s_raw)
        self.cache = (mu, sigma, s_raw, q2)
        return mu, sigma

    def log_prob(self, action: np.ndarray) -> float:
        mu, sigma, _, _ = self.cache
        u = (action - mu) / sigma
        return float(np.sum(-0.5 * u * u - np.log(sigma) - 0.5 * LOG_2PI))

    def entropy(self) -> float:
        _, sigma, _, _ = self.cache
        return float(np.sum(0.5 + 0.5 * LOG_2PI + np.log(sigma)))

    def _backward_outputs(
        self, g_mu: np.ndarray, g_sigma: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        mu, sigma, s_raw, q2 = self.cache
        g_s_raw = g_sigma * sigmoid(s_raw)
        g_q2 = self.A_mu.T @ g_mu + self.A_sg.T @ g_s_raw
        grads, g_f = self.trunk.backward(g_q2)
        grads["A_mu"] = g_mu[:, None] * q2[None, :]
        grads["b_mu"] = np.asarray(g_mu).copy()
        grads["A_sg"] = g_s_raw[:, None] * q2[None, :]
        grads["b_sg"] = g_s_raw.copy()
        n = self.n_units
        grads.update(self.rtu.param_grads(g_f[:n], g_f[n : 2 * n]))
        return grads, g_f

    def upstream_log_prob(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sigma, _, _ = self.cache
        u = (action - mu) / sigma
        return u / sigma, u * u / sigma - 1.0 / sigma

    def upstream_entropy(self) -> tuple[np.ndarray, np.ndarray]:
        _, sigma, _, _ = self.cache
        return np.zeros_like(sigma), 1.0 / sigma

    def backward_from_upstream(
        self, g_mu: np.ndarray, g_sigma: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        return self._backward_outputs(g_mu, g_sigma)

    def backward_log_prob(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        g_mu, g_sigma = self.upstream_log_prob(action)
        return self._backward_outputs(g_mu, g_sigma)

    def backward_entropy(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        g_mu, g_sigma = self.upstream_entropy()
        return self._backward_outputs(g_mu, g_sigma)
