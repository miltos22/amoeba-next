"""World-time streaming actor-critic with eligibility traces (brief 4.4).

One decision, in the order of brief section 4.4.3:

the policy network's recurrent layer and head are advanced, a_k is sampled;
the value network's recurrent layer and head are advanced to get v_k; the
   gradients of v, log pi(a_k) and H are taken at these same parameters
   before step 7 changes anything;
targets are written, the world advanced, x_{k+1}, the feeling and dt read;
v' is peeked from a copy of the value state advanced by dt_{k+1};
delta = scaled reward + gamma(dt_{k+1}) * v' - v_k;
the trace step uses gl(dt_k) and the entropy term sign(delta) * tau_ent;
the StreamingOptimizer runs with beta(dt_{k+1}) and alpha(dt_{k+1}).

The gradient step and the trace step commute: no parameter changes between them, so the policy
head is walked backwards once with the combined upstream gradient
g_lp + tau_ent sign(delta) g_H (the same arithmetic as two walks added).
The parameters are held in one flat vector per network so the trace and
the optimizer are one set of element-wise array operations.

Every constant is given per reference interval and converted with
p = dt / ref_s at the step.  Nothing here imports torch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .net import PolicyNet, ValueNet
from .norm import ObservationNormalizer, RewardScaler


@dataclass
class BrainConfig:
    """Every constant of brief section 4, one object."""

    ref_s: float = 0.1
    gamma_ref: float = 0.99
    lam_ref: float = 0.8
    beta_ref: float = 0.99995
    alpha_ref_policy: float = 1e-4
    alpha_ref_value: float = 2e-4
    tau_ent: float = 0.01
    n_units: int = 48
    hidden: int = 128
    input_gain: str = "zoh"
    leaky_slope: float = 0.01
    ln_eps: float = 1e-5
    obs_norm_eps: float = 1e-8
    reward_norm_eps: float = 1e-8
    optimizer_eps: float = 1e-8
    tau_min_s: float = 0.2
    tau_max_s: float = 1000.0
    period_min_s: float = 0.4
    period_max_s: float = 4000.0
    sparsity: float = 0.95
    control_s: float = 0.1
    seed: int = 0

    def _p(self, dt: float) -> float:
        return float(dt) / self.ref_s

    def gamma(self, dt: float) -> float:
        return float(self.gamma_ref ** self._p(dt))

    def trace_decay(self, dt: float) -> float:
        return float((self.gamma_ref * self.lam_ref) ** self._p(dt))

    def beta(self, dt: float) -> float:
        return float(self.beta_ref ** self._p(dt))

    def alpha_policy(self, dt: float) -> float:
        return float(self.alpha_ref_policy * self._p(dt))

    def alpha_value(self, dt: float) -> float:
        return float(self.alpha_ref_value * self._p(dt))

    def as_dict(self) -> dict:
        return asdict(self)


class _ParamGroup:
    """One network's parameters, trace and optimizer state as flat vectors."""

    def __init__(self, params: dict[str, np.ndarray]) -> None:
        self.params = params
        self.names = list(params)
        self.shapes = {k: params[k].shape for k in self.names}
        self.slices = {}
        offset = 0
        for key in self.names:
            size = int(params[key].size)
            self.slices[key] = (offset, offset + size)
            offset += size
        self.z = np.zeros(offset)
        self.vmax = np.zeros(offset)

    def flat(self, grads: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([grads[k].reshape(-1) for k in self.names])

    def add_trace(self, g_flat: np.ndarray, decay: float) -> None:
        self.z = decay * self.z + g_flat

    def step(self, delta: float, beta: float, alpha: float, eps: float) -> None:
        u = delta * self.z
        np.multiply(self.vmax, beta, out=self.vmax)
        np.maximum(self.vmax, np.abs(u), out=self.vmax)
        u /= self.vmax + eps
        u *= alpha
        for key in self.names:
            start, stop = self.slices[key]
            self.params[key] += u[start:stop].reshape(self.shapes[key])

    def weight_norm(self) -> float:
        total = 0.0
        for arr in self.params.values():
            total += float(np.sum(arr * arr))
        return math.sqrt(total)


class Learner:
    """Two recurrent networks, running normalisers, traces and optimiser."""

    def __init__(self, n_obs: int, n_actions: int, config: BrainConfig | None = None) -> None:
        self.config = BrainConfig() if config is None else config
        self.n_obs = int(n_obs)
        self.n_actions = int(n_actions)
        seed = int(self.config.seed)
        self.policy = PolicyNet(
            self.n_obs, self.n_actions, self.config, np.random.default_rng(seed)
        )
        self.value = ValueNet(self.n_obs, self.config, np.random.default_rng(seed + 1))
        self.action_rng = np.random.default_rng(seed + 2)
        self.obs_norm = ObservationNormalizer(
            (self.n_obs,), eps=self.config.obs_norm_eps
        )
        self.reward_scaler = RewardScaler(eps=self.config.reward_norm_eps)
        self.policy_group = _ParamGroup(self.policy.params)
        self.value_group = _ParamGroup(self.value.params)
        self.cur_dt = float(self.config.control_s)
        self.last_action = np.zeros(self.n_actions)
        self.last_reward = 0.0
        self.last_scaled_reward = 0.0
        self.last_delta = 0.0
        self.last_v = 0.0
        self.last_v_peek = 0.0
        self.up_lp = None
        self.up_h = None

    # -- observations ------------------------------------------------------
    def observe_initial(self, obs) -> np.ndarray:
        return self.obs_norm.update(obs)

    def normalise(self, obs) -> np.ndarray:
        return self.obs_norm.update(obs)

    @property
    def obs_count(self) -> int:
        return int(self.obs_norm.n)

    # -- one decision ------------------------------------------------------
    def act(self, x: np.ndarray, dt: float) -> np.ndarray:
        self.cur_dt = float(dt)
        f_pi = self.policy.advance(x, self.cur_dt)
        mu, sigma = self.policy.forward(f_pi)
        action = mu + sigma * self.action_rng.standard_normal(self.n_actions)
        self.last_action = action
        self.up_lp = self.policy.upstream_log_prob(action)
        self.up_h = self.policy.upstream_entropy()
        f_v = self.value.advance(x, self.cur_dt)
        self.last_v = self.value.forward(f_v)
        return action

    def observe(self, x_next: np.ndarray, need_next: float, dt_next: float) -> dict:
        cfg = self.config
        dt_next = float(dt_next)
        reward = -abs(float(need_next)) * cfg._p(dt_next)
        return self.observe_reward(x_next, reward, dt_next)

    def observe_reward(self, x_next: np.ndarray, reward: float, dt_next: float) -> dict:
        cfg = self.config
        dt_next = float(dt_next)
        reward = float(reward)
        scaled = self.reward_scaler.scale(reward, cfg.gamma(dt_next))
        v_peek = self.value.peek(x_next, dt_next)
        delta = float(scaled + cfg.gamma(dt_next) * v_peek - self.last_v)
        ent_sign = 1.0 if delta > 0.0 else (-1.0 if delta < 0.0 else 0.0)
        ent_coeff = cfg.tau_ent * ent_sign
        g_mu = self.up_lp[0] + ent_coeff * self.up_h[0]
        g_sigma = self.up_lp[1] + ent_coeff * self.up_h[1]
        g_policy, _ = self.policy.backward_from_upstream(g_mu, g_sigma)
        g_value, _ = self.value.backward()
        decay = cfg.trace_decay(self.cur_dt)
        self.policy_group.add_trace(self.policy_group.flat(g_policy), decay)
        self.value_group.add_trace(self.value_group.flat(g_value), decay)
        beta = cfg.beta(dt_next)
        eps = cfg.optimizer_eps
        self.policy_group.step(delta, beta, cfg.alpha_policy(dt_next), eps)
        self.value_group.step(delta, beta, cfg.alpha_value(dt_next), eps)
        self.last_reward = float(reward)
        self.last_scaled_reward = float(scaled)
        self.last_delta = delta
        self.last_v_peek = float(v_peek)
        return {
            "reward": float(reward),
            "scaled_reward": float(scaled),
            "delta": delta,
            "v": float(self.last_v),
            "v_peek": float(v_peek),
        }

    # -- records ------------------------------------------------------------
    def hidden_activations(self) -> np.ndarray:
        f, n1, inv1, q1, n2, inv2, q2 = self.policy.trunk.cache
        return np.concatenate([q1, q2])

    def tau_ranges(self) -> dict:
        return {"policy": self.policy.rtu.tau, "value": self.value.rtu.tau}

    def param_count(self, params: dict) -> int:
        return int(sum(arr.size for arr in params.values()))
