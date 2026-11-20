"""Known-answer checks for the recurrent layer (design DO 2).

1. At dt = 1 s and gain "paper" the layer equals the paper's four real
   equations written literally here (rtu.tex:281-292).
2. The exact online sensitivities match torch autograd through the
   unrolled sequence, for both gains and irregular dt.
3. The same check with the dc_da term left out must fail.
4. World-time invariance: held input, one 5 s step against 50 x 0.1 s.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from brain.learner import BrainConfig
from brain.rtu import RTULayer

torch.set_num_threads(1)


def _layer(n, d, gain):
    cfg = BrainConfig(n_units=n, input_gain=gain)
    layer = RTULayer(n, d, cfg, np.random.default_rng(0))
    rng = np.random.default_rng(4)
    layer.W_re[:] = rng.normal(size=(n, d)) * 0.3
    layer.W_im[:] = rng.normal(size=(n, d)) * 0.3
    return layer


def test_paper_equality_at_dt_one():
    rng = np.random.default_rng(1)
    n, d, steps = 6, 5, 50
    layer = _layer(n, d, "paper")
    layer.a[:] = rng.normal(size=n) * 0.3
    layer.b[:] = rng.normal(size=n) * 0.3
    layer.W_re[:] = rng.normal(size=(n, d)) * 0.2
    layer.W_im[:] = rng.normal(size=(n, d)) * 0.2
    xs = rng.normal(size=(steps, d))

    g = np.exp(-np.exp(layer.a)) * np.cos(np.exp(layer.b))
    phi = np.exp(-np.exp(layer.a)) * np.sin(np.exp(layer.b))
    gamma = np.sqrt(1.0 - np.exp(-2.0 * np.exp(layer.a)))
    h1 = np.zeros(n)
    h2 = np.zeros(n)
    for x in xs:
        h1_new = g * h1 - phi * h2 + gamma * (layer.W_re @ x)
        h2_new = g * h2 + phi * h1 + gamma * (layer.W_im @ x)
        h1, h2 = h1_new, h2_new

    layer.reset()
    for x in xs:
        f = layer.advance(x, 1.0)
    diff = max(np.max(np.abs(f[:n] - h1)), np.max(np.abs(f[n : 2 * n] - h2)))
    print(f"paper equality max abs difference = {diff!r}")
    assert diff < 1e-12


def _torch_grads(a, b, W_re, W_im, dts, xs, proj, gain):
    ta = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    tWr = torch.tensor(W_re, dtype=torch.float64, requires_grad=True)
    tWi = torch.tensor(W_im, dtype=torch.float64, requires_grad=True)
    n = ta.numel()
    h = torch.zeros(n, dtype=torch.complex128)
    for dt, x in zip(dts, xs):
        nu = torch.exp(ta)
        s = -nu + 1j * torch.exp(tb)
        lam = torch.exp(s * float(dt))
        if gain == "zoh":
            c = nu * (lam - 1.0) / s
        else:
            c = torch.sqrt(-torch.expm1(-2.0 * nu * float(dt))).to(torch.complex128)
        W = tWr + 1j * tWi
        h = lam * h + c * (W @ torch.tensor(x, dtype=torch.complex128))
    y = torch.dot(torch.tensor(proj[:n], dtype=torch.float64), h.real) + torch.dot(
        torch.tensor(proj[n:], dtype=torch.float64), h.imag
    )
    y.backward()
    return ta.grad.numpy(), tb.grad.numpy(), tWr.grad.numpy(), tWi.grad.numpy()


def _manual(a, b, W_re, W_im, dts, xs, proj, gain, omit_dc_da):
    n = a.size
    h = np.zeros(n, complex)
    E = np.zeros_like(W_re, dtype=complex)
    ea = np.zeros(n, complex)
    eb = np.zeros(n, complex)
    W = W_re + 1j * W_im
    for dt, x in zip(dts, xs):
        nu = np.exp(a)
        om = np.exp(b)
        s = -nu + 1j * om
        lam = np.exp(s * dt)
        lm1 = np.expm1(s * dt)
        dlam_da = -nu * dt * lam
        dlam_db = 1j * om * dt * lam
        if gain == "zoh":
            c = nu * lm1 / s
            dc_da = c + nu * (dlam_da * s - lm1 * (-nu)) / s**2
            dc_db = nu * (dlam_db * s - lm1 * (1j * om)) / s**2
        else:
            c = np.sqrt(-np.expm1(-2.0 * nu * dt)).astype(complex)
            dc_da = (nu * dt * np.exp(-2.0 * nu * dt) / c).astype(complex)
            dc_db = np.zeros_like(c)
        if omit_dc_da:
            dc_da = np.zeros_like(dc_da)
        u = W @ x
        ea = dlam_da * h + lam * ea + dc_da * u
        eb = dlam_db * h + lam * eb + dc_db * u
        E = lam[:, None] * E + c[:, None] * x[None, :]
        h = lam * h + c * u
    gre = proj[:n]
    gim = proj[n:]
    return {
        "a": gre * ea.real + gim * ea.imag,
        "b": gre * eb.real + gim * eb.imag,
        "W_re": gre[:, None] * E.real + gim[:, None] * E.imag,
        "W_im": -gre[:, None] * E.imag + gim[:, None] * E.real,
    }


@pytest.mark.parametrize("gain", ["zoh", "paper"])
def test_exact_sensitivities_match_torch(gain):
    rng = np.random.default_rng(2)
    n, d, steps = 5, 4, 30
    dts = rng.uniform(0.02, 3.0, steps)
    xs = rng.normal(size=(steps, d))
    proj = rng.normal(size=2 * n)
    a = np.log(1.0 / np.geomspace(0.2, 1000.0, n))
    b = np.log(2.0 * math.pi / np.exp(rng.uniform(math.log(0.4), math.log(4000.0), n)))
    W_re = rng.normal(size=(n, d)) * 0.3
    W_im = rng.normal(size=(n, d)) * 0.3

    layer = _layer(n, d, gain)
    layer.a[:] = a
    layer.b[:] = b
    layer.W_re[:] = W_re
    layer.W_im[:] = W_im
    layer.reset()
    for dt, x in zip(dts, xs):
        layer.advance(x, dt)
    grads = layer.param_grads(proj[:n], proj[n:])
    ref = _torch_grads(a, b, W_re, W_im, dts, xs, proj, gain)
    names = ["a", "b", "W_re", "W_im"]
    worst = 0.0
    for name, r in zip(names, ref):
        scale = max(np.max(np.abs(r)), 1e-30)
        rel = float(np.max(np.abs(grads[name] - r)) / scale)
        worst = max(worst, rel)
        print(f"gain={gain} {name} max relative error = {rel!r}")
    assert worst < 1e-8


@pytest.mark.parametrize("gain", ["zoh", "paper"])
def test_missing_dc_da_must_fail(gain):
    rng = np.random.default_rng(3)
    n, d, steps = 5, 4, 30
    dts = rng.uniform(0.02, 3.0, steps)
    xs = rng.normal(size=(steps, d))
    proj = rng.normal(size=2 * n)
    a = np.log(1.0 / np.geomspace(0.2, 1000.0, n))
    b = np.log(2.0 * math.pi / np.exp(rng.uniform(math.log(0.4), math.log(4000.0), n)))
    W_re = rng.normal(size=(n, d)) * 0.3
    W_im = rng.normal(size=(n, d)) * 0.3
    ref = _torch_grads(a, b, W_re, W_im, dts, xs, proj, gain)
    broken = _manual(a, b, W_re, W_im, dts, xs, proj, gain, omit_dc_da=True)
    names = ["a", "b", "W_re", "W_im"]
    worst = 0.0
    for name, r in zip(names, ref):
        diff = float(np.max(np.abs(broken[name] - r)))
        worst = max(worst, diff)
        print(f"gain={gain} {name} missing-dc_da max abs difference = {diff!r}")
    assert worst > 1e-4


def _held(layer, gain, steps, total=5.0):
    cfg = BrainConfig(n_units=layer.n_units, input_gain=gain)
    layer.config = cfg
    layer.reset()
    x = np.ones(layer.n_in)
    dt = total / steps
    for _ in range(steps):
        f = layer.advance(x, dt)
    return f


def test_world_time_invariance():
    n, d = 6, 3
    zoh = _layer(n, d, "zoh")
    paper = _layer(n, d, "paper")
    paper.a[:] = zoh.a
    paper.b[:] = zoh.b
    paper.W_re[:] = zoh.W_re
    paper.W_im[:] = zoh.W_im
    a1 = _held(zoh, "zoh", 1)
    a50 = _held(zoh, "zoh", 50)
    d_zoh = float(np.max(np.abs(a1 - a50)))
    b1 = _held(paper, "paper", 1)
    b50 = _held(paper, "paper", 50)
    d_paper = float(np.max(np.abs(b1 - b50)))
    print(f"zoh 1 step vs 50: max abs difference = {d_zoh!r}")
    print(f"paper 1 step vs 50: max abs difference = {d_paper!r}")
    assert d_zoh < 1e-10
    assert d_paper > 1e-3
