"""Window-level oracle tests (revision 7, item 1)."""

from __future__ import annotations

import numpy as np

from step0a.far_value import (
    Stream, ubar_dict, v_b1, v_b1_windows, v_maxov_windows,
)


def _unpatched(U, dP, windows):
    U = np.asarray(U, dtype=float)
    dP = np.asarray(dP, dtype=float)
    M = len(U)
    a = np.zeros(M)
    b = np.zeros(M)
    for lo, hi, w, ub in windows:
        a[lo:hi] += w
        b[lo:hi] += w * ub
    return float(np.sum(a * U - b * dP))


def test_oracle_values():
    U = np.ones(10)
    dP = np.zeros(10)
    A = (2, 6, 0.5, 0.0)
    B = (4, 8, 0.5, 0.0)
    assert abs(v_b1_windows(U, dP, [A, B]) - 4.0) < 1e-12
    # O2
    A2 = (2, 6, 1.0, 0.0)
    B2 = (4, 8, 1.0, 0.0)
    assert abs(v_b1_windows(U, dP, [A2, B2]) - 6.0) < 1e-12
    assert abs(v_maxov_windows(U, dP, [A2, B2]) - 4.0) < 1e-12
    assert abs(_unpatched(U, dP, [A2, B2]) - 8.0) < 1e-12
    # O3
    Ur = np.arange(10, dtype=float)
    assert abs(v_b1_windows(Ur, dP, [(0, 10, 1.0, 0.0),
                                     (3, 5, 1.0, 0.0)]) - 45.0) < 1e-12
    assert abs(_unpatched(Ur, dP, [(0, 10, 1.0, 0.0),
                                   (3, 5, 1.0, 0.0)]) - 52.0) < 1e-12
    # O4
    U4 = np.zeros(10)
    U4[2:6] = 2.0
    U4[6:10] = -1.0
    assert abs(v_b1_windows(U4, dP, [A2, B2]) - 6.0) < 1e-12
    assert abs(_unpatched(U4, dP, [A2, B2]) - 10.0) < 1e-12
    # O5
    U0 = np.zeros(10)
    d1 = np.ones(10)
    win5 = [(2, 6, 1.0, 0.2), (4, 8, 1.0, 0.4)]
    assert abs(v_b1_windows(U0, d1, win5) + 1.8) < 1e-12
    # O6
    dr = np.arange(10) / 10.0
    assert abs(v_b1_windows(U0, dr, win5) + 0.89) < 1e-12
    # O7
    A7 = (2, 6, 1.0, 0.0)
    assert abs(v_b1_windows(U, dP, [A7, A7]) - 4.0) < 1e-12
    assert abs(_unpatched(U, dP, [A7, A7]) - 8.0) < 1e-12


def test_anchor_level_equals_window_level():
    rng = np.random.default_rng(0)
    for _ in range(50):
        M = 500
        U = rng.normal(0, 1, M)
        dP = rng.uniform(0.01, 0.2, M)
        E = rng.uniform(0, 1, M)
        t = np.arange(M + 1) * 0.1
        st = Stream(U, dP, E, t[:-1], t[1:])
        n = int(rng.integers(1, 8))
        anchors = rng.integers(0, M - 2 ** 7, n)
        ks = rng.integers(1, 8, n)
        e = rng.uniform(0.1, 2.0, n)
        n0 = float(rng.choice([0.2, 1.0, 5.0]))
        ub = ubar_dict(st, anchors, ks, "phys")
        lbl = v_b1(st, anchors, ks, e, n0, ub, "phys")
        W = {}
        for ej, k in zip(e, ks):
            W[k] = W.get(k, 0.0) + ej
        windows = []
        for p, k, ej in zip(anchors, ks, e):
            if p > M - 2 ** k - 1:
                continue
            w = ej / (n0 + W[k])
            windows.append((p + 1 + 2 ** (k - 1), p + 2 ** k + 1, w,
                            ub.get((int(k), int(p)), 0.0)))
        orc = v_b1_windows(U, dP, windows)
        assert abs(lbl - orc) <= 1e-9 * max(abs(orc), 1.0)
