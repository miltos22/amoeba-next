"""Tests for the revision 8 readouts (absolute-scale precision, shrunk V)."""

from __future__ import annotations

import numpy as np

from step0a.far_value import Stream
from step0a.rigs.run_item3c import KMAX, Y_table, scale_v, shrunk_v


def _stream(M=60, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, M)
    dP = rng.uniform(0.01, 0.1, M)
    E = rng.uniform(0, 1, M)
    t = np.arange(M + 1) * 0.1
    return Stream(U, dP, E, t[:-1], t[1:])


def test_scale_v_formula():
    rng = np.random.default_rng(1)
    R = {}
    for k in range(1, KMAX + 1):
        r = rng.normal(0, 1, 20)
        r[rng.random(20) < 0.2] = np.nan
        R[k] = r
    sig = np.ones(KMAX + 1) * 0.7
    idx = np.array([0, 2, 5, 7])
    tau = 1.0
    total, per = scale_v(R, idx, sig, tau, n0=1.0)
    man = 0.0
    for k in range(1, KMAX + 1):
        r = R[k][idx]
        m = ~np.isnan(r)
        den = 1.0 * sig[k] ** 2 / tau ** 2 + m.sum()
        v = float(np.sum(r[m])) / den if m.any() else 0.0
        assert abs(per[k] - v) < 1e-12
        man += v
    assert abs(total - man) < 1e-12


def test_shrunk_limits():
    rng = np.random.default_rng(2)
    R = {}
    for k in range(1, KMAX + 1):
        r = rng.normal(0, 1, 20)
        r[rng.random(20) < 0.2] = np.nan
        R[k] = r
    idx = np.arange(20)
    t1, p1 = shrunk_v(R, idx, 1.0)
    t0, p0 = shrunk_v(R, idx, 1e-9)
    for k in range(1, KMAX + 1):
        r = R[k][idx]
        m = ~np.isnan(r)
        if m.any():
            assert abs(p1[k] - np.sum(r[m]) / (1.0 + m.sum())) < 1e-12
            assert abs(p0[k] - np.mean(r[m])) < 1e-6


def test_Y_table_matches_prefix():
    st = _stream(M=500)
    positions = np.array([0, 3, 40])
    Yt = Y_table(st, positions, kmax=5)
    for k in range(1, 6):
        q = 2 ** k
        half = 2 ** (k - 1)
        for i, p in enumerate(positions):
            if p <= st.M - q - 1:
                v = float(st.prefU[p + q + 1] - st.prefU[p + half + 1])
                assert abs(Yt[k][i] - v) < 1e-12
