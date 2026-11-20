"""Tests for the RECEPTOR mode (revision 2)."""

from __future__ import annotations

import numpy as np

from step0a.experience_time import (
    SIGMA_MIN, GridGroup, SensoryGroup, organ_stream, read_organ, simulate,
)
from step0a.pacemaker import Pacemaker
from step0a.receptors import Receptor, calibrate_sigma_hat
from step0a.world_signals import (
    H_W, Source, irregular_look_indices, look_indices, n_grid,
)


def _ramp_case(life=50.0, seed=0):
    n = n_grid(life)
    pm = Pacemaker(noisy=False, seed=seed)
    src = Source("ramp", seed=seed)
    x = Receptor(src, 0.2).filtered(n)
    sig = calibrate_sigma_hat(src)
    gg = GridGroup("ramp", x[None, :], np.array([sig]), np.array([SIGMA_MIN]))
    sg = SensoryGroup("ramp", x[None, :], np.array([sig]), np.array([SIGMA_MIN]))
    return n, pm, gg, sg, x, sig


def test_receptor_equals_chord_at_grid_rate():
    n, pm, gg, sg, x, sig = _ramp_case()
    organ = organ_stream(pm, [gg], n)
    cum = np.cumsum(organ["m_grid"], dtype=np.int64)
    all_idx = np.arange(n, dtype=np.int64)
    ch = simulate(pm, [sg], all_idx, n, mode="chord",
                  store_trajectory=True, drop_zero_groups=False)
    assert np.array_equal(ch["phi_traj"], cum.astype(np.float64))
    assert int(ch["phi"]) == organ["phi"]
    for sched in ("2", "5", "15", "50", "irregular"):
        idx = look_indices(sched, n, seed=7)
        ro = read_organ(organ, idx)
        assert np.array_equal(ro["phi_look"], cum[idx])


def test_receptor_phi_count_schedule_independent():
    n, pm, gg, sg, x, sig = _ramp_case()
    organ = organ_stream(pm, [gg], n)
    cum = np.cumsum(organ["m_grid"], dtype=np.int64)
    schedules = [look_indices(s, n) for s in ("2", "5", "15", "50")]
    schedules.append(irregular_look_indices(n, seed=11))
    t_common = min(int(idx[-1]) for idx in schedules)
    for idx in schedules:
        idx2 = idx[idx <= t_common]
        if idx2[-1] != t_common:
            idx2 = np.append(idx2, t_common)
        ro = read_organ(organ, idx2)
        assert int(ro["phi_look"][-1]) == int(cum[t_common])


def test_receptor_silent_group_adds_zero():
    life = 20.0
    n = n_grid(life)
    pm = Pacemaker(noisy=False, seed=0)
    zero = np.zeros((1, n), dtype=np.float64)
    def run(k):
        groups = [GridGroup(f"s{i}", zero, np.array([0.0]),
                            np.array([SIGMA_MIN])) for i in range(k)]
        return organ_stream(pm, groups, n, drop_zero_groups=False)
    a = run(0)
    b = run(1)
    c = run(10)
    assert np.array_equal(a["m_grid"], b["m_grid"])
    assert np.array_equal(a["m_grid"], c["m_grid"])
    assert a["phi"] == b["phi"] == c["phi"]


def test_receptor_declared_copy_changes_nothing():
    n, pm, gg, sg, x, sig = _ramp_case(life=20.0)
    copy = GridGroup("ramp_copy", x[None, :], np.array([sig]),
                     np.array([SIGMA_MIN]), copies_of="ramp")
    a = organ_stream(pm, [gg], n, drop_zero_groups=False)
    b = organ_stream(pm, [gg, copy], n, drop_zero_groups=False)
    assert np.array_equal(a["m_grid"], b["m_grid"])
    assert a["phi"] == b["phi"]
