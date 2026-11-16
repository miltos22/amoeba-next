"""Known-answer checks for the heads (brief DO 3).

The hand-written backward pass of the value head, and of log pi and the
entropy H of the policy head, against torch autograd, for n = 0 and
n = 8.  The recurrent sensitivities themselves are checked in
tests/test_brain_rtu.py; here f is a leaf of its own so the check covers
every head parameter and the gradient handed back to the recurrent layer.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from brain.learner import BrainConfig
from brain.net import PolicyNet, ValueNet

torch.set_num_threads(1)
LOG_2PI = math.log(2.0 * math.pi)


def _torch_ln(z, eps):
    mean = z.mean()
    var = ((z - mean) ** 2).mean()
    return (z - mean) / torch.sqrt(var + eps)


def _torch_trunk(p, f, eps, slope):
    z1 = p["A1"] @ f + p["b1"]
    n1 = _torch_ln(z1, eps)
    q1 = torch.where(n1 > 0.0, n1, slope * n1)
    z2 = p["A2"] @ q1 + p["b2"]
    n2 = _torch_ln(z2, eps)
    q2 = torch.where(n2 > 0.0, n2, slope * n2)
    return q1, q2


def _torch_grads(params, f, kind, eps, slope, action=None):
    p = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in params.items()}
    ft = torch.tensor(np.asarray(f, dtype=float), dtype=torch.float64, requires_grad=True)
    q1, q2 = _torch_trunk(p, ft, eps, slope)
    if kind == "value":
        y = p["a3"] @ q2 + p["b3"]
    else:
        mu = p["A_mu"] @ q2 + p["b_mu"]
        s_raw = p["A_sg"] @ q2 + p["b_sg"]
        sigma = torch.logaddexp(torch.zeros_like(s_raw), s_raw)
        if kind == "logpi":
            u = (torch.tensor(action, dtype=torch.float64) - mu) / sigma
            y = torch.sum(-0.5 * u * u - torch.log(sigma) - 0.5 * LOG_2PI)
        else:
            y = torch.sum(0.5 + 0.5 * LOG_2PI + torch.log(sigma))
    y.backward()
    out = {k: v.grad.numpy() for k, v in p.items() if v.grad is not None}
    out["f"] = ft.grad.numpy()
    return float(y.detach()), out


def _rel(a, b):
    scale = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b)) / scale)


def _check(n_units):
    rng = np.random.default_rng(10 + n_units)
    cfg = BrainConfig(n_units=n_units, hidden=128, seed=0)
    policy = PolicyNet(3, 2, cfg, np.random.default_rng(0))
    value = ValueNet(3, cfg, np.random.default_rng(1))
    x = rng.normal(size=3)
    dt = 0.37
    action = rng.normal(size=2)
    eps = cfg.ln_eps
    slope = cfg.leaky_slope

    f_pi = policy.advance(x, dt)
    mu, sigma = policy.forward(f_pi)
    lp, g_f_lp = policy.backward_log_prob(action)
    gh, g_f_h = policy.backward_entropy()

    t_lp, t_grads_lp = _torch_grads(policy.params, f_pi, "logpi", eps, slope, action)
    t_h, t_grads_h = _torch_grads(policy.params, f_pi, "entropy", eps, slope)
    worst = 0.0
    for name in t_grads_lp:
        if name == "f":
            continue
        r1 = _rel(lp[name], t_grads_lp[name])
        worst = max(worst, r1)
        print(f"n={n_units} policy {name} logpi rel={r1!r}")
    for name in t_grads_h:
        if name == "f":
            continue
        r2 = _rel(gh[name], t_grads_h[name])
        worst = max(worst, r2)
        print(f"n={n_units} policy {name} entropy rel={r2!r}")
    r_f1 = _rel(g_f_lp, t_grads_lp["f"])
    r_f2 = _rel(g_f_h, t_grads_h["f"])
    worst = max(worst, r_f1, r_f2)
    print(f"n={n_units} policy f logpi rel={r_f1!r} entropy rel={r_f2!r}")
    own_lp = policy.log_prob(action)
    own_h = policy.entropy()
    print(
        f"n={n_units} logpi own={own_lp!r} torch={t_lp!r} "
        f"entropy own={own_h!r} torch={t_h!r}"
    )
    assert abs(own_lp - t_lp) < 1e-12
    assert abs(own_h - t_h) < 1e-12

    f_v = value.advance(x, dt)
    own_v = value.forward(f_v)
    gv, g_f_v = value.backward()
    t_v, t_grads_v = _torch_grads(value.params, f_v, "value", eps, slope)
    for name in t_grads_v:
        if name == "f":
            continue
        r = _rel(gv[name], t_grads_v[name])
        worst = max(worst, r)
        print(f"n={n_units} value {name} rel={r!r}")
    r_fv = _rel(g_f_v, t_grads_v["f"])
    worst = max(worst, r_fv)
    print(f"n={n_units} value f rel={r_fv!r}")
    print(f"n={n_units} v own={own_v!r} torch={t_v!r}")
    assert abs(own_v - t_v) < 1e-12
    print(f"n={n_units} worst max relative error = {worst!r}")
    assert worst < 1e-8


def test_value_and_policy_heads_n0():
    _check(0)


def test_value_and_policy_heads_n8():
    _check(8)
