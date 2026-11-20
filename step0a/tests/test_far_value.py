"""Tests for the far-value readout (revision 6, item 1)."""

from __future__ import annotations

import numpy as np

from step0a.far_value import (
    Stream, band_Y, compute_ubar, leverage, ubar_dict, v_b1, v_b1_fast, v_unp,
)


def _stream(M=300, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, M)
    dP = rng.uniform(0.05, 0.2, M)
    E = rng.uniform(0, 1, M)
    t = np.arange(M + 1) * 0.1
    return Stream(U, dP, E, t[:-1], t[1:])


def _brute_ubar(st, k, p, mode):
    q = 2 ** k
    half = 2 ** (k - 1)
    Mv = st.M - q
    b = int(st.bin[p])
    sy = sd = sya = sda = 0.0
    for i in range(Mv):
        if abs(i - p) < half:
            continue
        y = float(st.prefU[i + q + 1] - st.prefU[i + half + 1])
        d = float(half) if mode == "phi" else float(
            st.prefDP[i + q + 1] - st.prefDP[i + half + 1])
        sya += y
        sda += d
        if st.bin[i] == b:
            sy += y
            sd += d
    ub_all = sya / sda if sda > 0 else 0.0
    med = 1.0 if mode == "phi" else st.median_dP
    D0 = half * med * 10.0
    return (sy + D0 * ub_all) / (sd + D0) if (sd + D0) > 0 else 0.0


def test_prefix_bands_and_pools_match_brute_force():
    st = _stream()
    for k in range(1, 7):
        q = 2 ** k
        half = 2 ** (k - 1)
        Y, dP = band_Y(st, k)
        for p in (0, 10, 100, st.M - q - 1):
            by = float(st.U[p + 1 + half:p + q + 1].sum())
            bd = float(st.dP[p + 1 + half:p + q + 1].sum())
            assert abs(Y[p] - by) < 1e-12
            assert abs(dP[p] - bd) < 1e-12
            for mode in ("phys", "phi"):
                ub, _, _ = compute_ubar(st, k, np.array([p]), mode)
                assert abs(ub[0] - _brute_ubar(st, k, p, mode)) < 1e-12


def _brute_v_b1(st, anchors, ks, weights, n0, ub, mode):
    W = {}
    for e, k in zip(weights, ks):
        W[k] = W.get(k, 0.0) + e
    a = np.zeros(st.M)
    b = np.zeros(st.M)
    for p, k, e in zip(anchors, ks, weights):
        if p > st.M - 2 ** k - 1:
            continue
        q = 2 ** k
        half = 2 ** (k - 1)
        w = e / (n0 + W[k])
        lo, hi = p + 1 + half, p + q + 1
        a[lo:hi] += w
        b[lo:hi] += w * ub.get((k, int(p)), 0.0)
    V = 0.0
    for r in range(st.M):
        if a[r] > 0:
            s = min(1.0, 1.0 / a[r])
            Dp = 1.0 if mode == "phi" else st.dP[r]
            V += s * (a[r] * st.U[r] - b[r] * Dp)
    return V


def test_sweep_equals_bruteforce():
    st = _stream(M=4000, seed=2)
    rng = np.random.default_rng(3)
    n = 8
    anchors = rng.integers(0, st.M - 2 ** 9, n)
    ks = rng.integers(1, 10, n)
    weights = rng.uniform(0.2, 2.0, n)
    n0 = 0.7
    for mode in ("phys", "phi", "none"):
        ub = ubar_dict(st, anchors, ks, mode)
        sweep = v_b1(st, anchors, ks, weights, n0, ub, mode)
        fast = v_b1_fast(st, anchors, ks, weights, n0, ub, mode)
        brute = _brute_v_b1(st, anchors, ks, weights, n0, ub, mode)
        assert abs(sweep - brute) <= 1e-9 * max(abs(brute), 1.0)
        assert abs(fast - brute) <= 1e-9 * max(abs(brute), 1.0)


def test_one_anchor_b1_equals_unp():
    st = _stream(seed=4)
    anchors = np.array([50])
    ks = np.array([3])
    weights = np.array([1.0])
    ub = ubar_dict(st, anchors, ks, "phys")
    v1 = v_b1(st, anchors, ks, weights, 1.0, ub, "phys")
    v2, _ = v_unp(st, anchors, ks, weights, 1.0, ub, "phys")
    assert abs(v1 - v2) <= 1e-12


def test_broken_b1_skips_leverage_bound():
    st = _stream(M=40000, seed=5)
    p = 30000
    ks = np.arange(5, 15)
    anchors = np.array([p - (2 ** k - (i // 10)) for i, k in enumerate(ks)])
    weights = np.full(10, 0.1)
    n0 = 0.2
    good = leverage(st, p, anchors, ks, weights, n0, "phys", "b1", broken=False)
    bad = leverage(st, p, anchors, ks, weights, n0, "phys", "b1", broken=True)
    assert good["L_direct"] <= 1.0 + 1e-9
    assert bad["L_direct"] > 1.0 + 1e-9
