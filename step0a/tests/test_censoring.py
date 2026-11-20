"""Tests for H6 interval censoring (revision 3, items 2 and 3)."""

from __future__ import annotations

import numpy as np

from step0a.censoring import band_bounds, read_atoms
from step0a.experience_time import organ_stream
from step0a.pacemaker import Pacemaker
from step0a.world_signals import (
    H_W, UtilitySource, look_indices, n_grid,
)

S = 1.0
TAU = 0.5 * S


def test_h6a_bounds_cover_every_shock_position():
    # one atom spanning Phi [100, 105] with one shock of integral -S
    ends = np.array([105.0])
    ip = np.array([0.0])
    im = np.array([S])
    a, b = 102.0, 104.0
    lo, hi, _, _ = band_bounds(a, b, ends, ip, im)
    assert lo <= -S + 1e-12 and hi >= 0.0 - 1e-12
    positions = np.linspace(100.01, 104.99, 101)
    inside = 0
    for p in positions:
        true = -S if (a < p <= b) else 0.0
        if lo - 1e-12 <= true <= hi + 1e-12:
            inside += 1
    assert inside == 101
    # broken: linear interpolation of the atom integral by Phi
    broken = -S * (b - a) / (ends[0] - 100.0)
    wrong = 0
    for p in positions:
        true = -S if (a < p <= b) else 0.0
        if abs(broken - true) > 0.1 * S:
            wrong += 1
    assert wrong > 0


def test_h6b_coalescing_and_broken_anchor_per_look():
    life = 200.0
    n = n_grid(life)
    pm = Pacemaker(noisy=False, seed=0)
    organ = organ_stream(pm, [], n)
    idx = look_indices("50", n)
    u = UtilitySource([(50.0, 50.2, -5.0), (120.0, 120.5, 2.0)]).grid(n)
    cp, cm = UtilitySource.cumulative_integrals(u)
    ip, im = UtilitySource.interval_integrals(cp, cm, idx)

    normal = read_atoms(organ, idx, ip, im)
    assert normal["anchors_at_m0"] == 0
    assert normal["n_m0_looks"] > 0
    # all anchors have m >= 1
    assert np.all(normal["m"] >= 1)
    # coalesced sums equal the direct physical integrals over the same time
    direct_p = float(cp[idx[-1]])
    direct_m = float(cm[idx[-1]])
    coal_p = float(normal["I_plus"].sum() + normal["pending_I_plus"])
    coal_m = float(normal["I_minus"].sum() + normal["pending_I_minus"])
    assert abs(coal_p - direct_p) <= 1e-9 * max(abs(direct_p), 1.0)
    assert abs(coal_m - direct_m) <= 1e-9 * max(abs(direct_m), 1.0)

    broken = read_atoms(organ, idx, ip, im, anchor_every_look=True)
    assert broken["anchors_at_m0"] > 0
    assert broken["anchors_at_m0"] == broken["n_m0_looks"]
