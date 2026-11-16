"""Known-answer checks for the learner (brief DO 4).

1. n = 0: 20 decisions on random observations and rewards with irregular
   dt; after every decision every parameter equals a torch/autograd
   reference of brief section 4.4 written here, to 1e-9.
2. SparseInit: realised zero share over a network within 0.01 of 0.95,
   every bias zero.
3. TrackMeanVar equals numpy's mean and unbiased variance on 1000 random
   vectors to 1e-10.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from brain.learner import BrainConfig, Learner
from brain.norm import ObservationNormalizer

torch.set_num_threads(1)
LOG_2PI = math.log(2.0 * math.pi)


def _ln(z, eps):
    mean = z.mean()
    var = ((z - mean) ** 2).mean()
    return (z - mean) / torch.sqrt(var + eps)


def _trunk(p, x, eps, slope):
    z1 = p["A1"] @ x + p["b1"]
    n1 = _ln(z1, eps)
    q1 = torch.where(n1 > 0.0, n1, slope * n1)
    z2 = p["A2"] @ q1 + p["b2"]
    n2 = _ln(z2, eps)
    q2 = torch.where(n2 > 0.0, n2, slope * n2)
    return q2


def _policy(p, x, cfg):
    q2 = _trunk(p, x, cfg.ln_eps, cfg.leaky_slope)
    mu = p["A_mu"] @ q2 + p["b_mu"]
    s_raw = p["A_sg"] @ q2 + p["b_sg"]
    sigma = torch.logaddexp(torch.zeros_like(s_raw), s_raw)
    return mu, sigma


def _value(p, x, cfg):
    q2 = _trunk(p, x, cfg.ln_eps, cfg.leaky_slope)
    return p["a3"] @ q2 + p["b3"]


def _leaves(params):
    return {
        k: torch.tensor(v.copy(), dtype=torch.float64, requires_grad=True)
        for k, v in params.items()
    }


def _grads(y, leaves, retain=False):
    raw = torch.autograd.grad(
        y, list(leaves.values()), allow_unused=True, retain_graph=retain
    )
    out = {}
    for (k, leaf), g in zip(leaves.items(), raw):
        out[k] = torch.zeros_like(leaf) if g is None else g
    return out


def test_n0_matches_torch_reference():
    rng = np.random.default_rng(0)
    n_obs, n_actions, n_dec = 4, 3, 20
    cfg = BrainConfig(n_units=0, hidden=64, seed=0)
    learner = Learner(n_obs, n_actions, cfg)
    raw_obs = rng.uniform(-1.0, 1.0, size=(n_dec + 1, n_obs))
    needs = rng.uniform(-1.0, 1.0, size=n_dec)
    dts = rng.uniform(0.02, 0.5, size=n_dec + 1)

    pol = _leaves(learner.policy.params)
    val = _leaves(learner.value.params)
    z_pol = {k: torch.zeros_like(v) for k, v in pol.items()}
    z_val = {k: torch.zeros_like(v) for k, v in val.items()}
    vm_pol = {k: torch.zeros_like(v) for k, v in pol.items()}
    vm_val = {k: torch.zeros_like(v) for k, v in val.items()}

    obs_n = 0
    obs_mu = torch.zeros(n_obs, dtype=torch.float64)
    obs_p = torch.zeros(n_obs, dtype=torch.float64)

    def ref_normalise(raw):
        nonlocal obs_n, obs_mu, obs_p
        obs_n += 1
        x = torch.tensor(raw, dtype=torch.float64)
        delta = x - obs_mu
        obs_mu = obs_mu + delta / obs_n
        obs_p = obs_p + delta * (x - obs_mu)
        var = obs_p / (obs_n - 1) if obs_n >= 2 else torch.ones_like(obs_p)
        return (x - obs_mu) / torch.sqrt(var + cfg.obs_norm_eps)

    u_trace = 0.0
    rn = 0
    r_mu = torch.zeros((), dtype=torch.float64)
    r_p = torch.zeros((), dtype=torch.float64)

    def ref_scale(reward, gamma):
        nonlocal u_trace, rn, r_mu, r_p
        u_trace = gamma * u_trace + reward
        u = torch.tensor(u_trace, dtype=torch.float64)
        rn += 1
        delta = u - r_mu
        r_mu = r_mu + delta / rn
        r_p = r_p + delta * (u - r_mu)
        var = r_p / (rn - 1) if rn >= 2 else torch.ones_like(r_p)
        return reward / torch.sqrt(var + cfg.reward_norm_eps)

    x = learner.observe_initial(raw_obs[0])
    x_ref = ref_normalise(raw_obs[0])
    worst = 0.0
    for k in range(n_dec):
        action = learner.act(x, dts[k])
        a_t = torch.tensor(action, dtype=torch.float64)
        mu, sigma = _policy(pol, x_ref, cfg)
        u = (a_t - mu) / sigma
        log_pi = torch.sum(-0.5 * u * u - torch.log(sigma) - 0.5 * LOG_2PI)
        entropy = torch.sum(0.5 + 0.5 * LOG_2PI + torch.log(sigma))
        v = _value(val, x_ref, cfg)
        g_lp = _grads(log_pi, pol, retain=True)
        g_h = _grads(entropy, pol)
        g_v = _grads(v, val)

        x_next = learner.normalise(raw_obs[k + 1])
        info = learner.observe(x_next, needs[k], dts[k + 1])
        x_ref_next = ref_normalise(raw_obs[k + 1])
        dt_next = dts[k + 1]
        p_scale = dt_next / cfg.ref_s
        reward = -abs(float(needs[k])) * p_scale
        scaled = ref_scale(reward, cfg.gamma(dt_next))
        v_peek = _value(val, x_ref_next, cfg)
        delta = scaled + cfg.gamma(dt_next) * v_peek - v
        gl = cfg.trace_decay(dts[k])
        ent = cfg.tau_ent * torch.sign(delta)
        beta = cfg.beta(dt_next)
        alpha_p = cfg.alpha_policy(dt_next)
        alpha_v = cfg.alpha_value(dt_next)
        eps = cfg.optimizer_eps
        with torch.no_grad():
            for key in pol:
                z_pol[key] = gl * z_pol[key] + g_lp[key] + ent * g_h[key]
                u_t = delta * z_pol[key]
                vm_pol[key] = torch.maximum(beta * vm_pol[key], torch.abs(u_t))
                pol[key].add_(alpha_p * u_t / (vm_pol[key] + eps))
            for key in val:
                z_val[key] = gl * z_val[key] + g_v[key]
                u_t = delta * z_val[key]
                vm_val[key] = torch.maximum(beta * vm_val[key], torch.abs(u_t))
                val[key].add_(alpha_v * u_t / (vm_val[key] + eps))

        for key, ref_leaf in pol.items():
            if ref_leaf.numel() == 0:
                continue
            diff = float(
                np.max(np.abs(learner.policy.params[key] - ref_leaf.detach().numpy()))
            )
            worst = max(worst, diff)
        for key, ref_leaf in val.items():
            if ref_leaf.numel() == 0:
                continue
            diff = float(
                np.max(np.abs(learner.value.params[key] - ref_leaf.detach().numpy()))
            )
            worst = max(worst, diff)
        if k == 0:
            print(
                f"decision 1: delta learner={info['delta']!r} ref={float(delta.detach())!r} "
                f"scaled learner={info['scaled_reward']!r} ref={float(scaled)!r}"
            )
        x = x_next
        x_ref = x_ref_next
    print(f"n=0 worst max abs parameter difference over {n_dec} decisions = {worst!r}")
    assert worst < 1e-9


def test_sparse_init_share_and_biases():
    cfg = BrainConfig(n_units=48, hidden=128, seed=0)
    learner = Learner(38, 12, cfg)
    for name, params in (("policy", learner.policy.params), ("value", learner.value.params)):
        total = sum(int(arr.size) for arr in params.values())
        zeros = sum(int(np.count_nonzero(arr == 0.0)) for arr in params.values())
        share = zeros / total
        print(f"{name}: total={total} zeros={zeros} share={share!r}")
        assert abs(share - 0.95) < 0.01
        for key in ("b1", "b2", "b3", "b_mu", "b_sg"):
            if key in params:
                assert np.all(params[key] == 0.0), (name, key)


def test_normaliser_matches_numpy():
    rng = np.random.default_rng(5)
    xs = rng.normal(size=(1000, 7))
    norm = ObservationNormalizer((7,), eps=1e-8)
    for x in xs:
        out = norm.update(x)
    mean = xs.mean(axis=0)
    var = xs.var(axis=0, ddof=1)
    print(
        f"mean max abs difference = {np.max(np.abs(norm.mu - mean))!r} "
        f"var max abs difference = {np.max(np.abs(norm.var - var))!r}"
    )
    assert np.max(np.abs(norm.mu - mean)) < 1e-10
    assert np.max(np.abs(norm.var - var)) < 1e-10
    assert norm.n == 1000
