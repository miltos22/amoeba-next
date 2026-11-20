"""Unit tests for section 2 of the design (exact properties)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from step0a.experience_time import (
    DELTA, SIGMA_HAT_P, SIGMA_MIN, Z_REF, SensoryGroup, assert_non_freezing,
    basal_coords, simulate,
)
from step0a.pacemaker import Pacemaker
from step0a.receptors import Receptor, calibrate_sigma_hat
from step0a.pacemaker import PERIODS
from step0a.world_signals import (
    H_W, Source, UtilitySource, irregular_look_indices, look_indices, n_grid,
)


# ---------------------------------------------------------------------------
# 2.1 receptor identity across look schedules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["steps", "bursts", "ramp"])
def test_receptor_identity_across_schedules(kind):
    life = 200.0
    n = n_grid(life)
    src = Source(kind=kind, seed=3)
    x_all = Receptor(src, 0.2).filtered(n)
    ref_idx = look_indices("50", n)
    ref = x_all[ref_idx]
    schedules = [look_indices(s, n) for s in ["2", "5", "15"]]
    schedules.append(irregular_look_indices(n, seed=17))
    for idx in schedules:
        common = np.intersect1d(idx, ref_idx)
        assert len(common) > 10
        pos_s = np.searchsorted(idx, common)
        pos_r = np.searchsorted(ref_idx, common)
        assert np.max(np.abs(x_all[idx[pos_s]] - ref[pos_r])) <= 1e-12


# ---------------------------------------------------------------------------
# 2.2 utility integrals across look schedules
# ---------------------------------------------------------------------------

def test_utility_integrals_across_schedules():
    life = 200.0
    n = n_grid(life)
    u = UtilitySource([(10.0, 40.0, 0.7), (60.0, 61.0, -2.0),
                       (120.0, 180.0, 0.3)]).grid(n)
    cp, cm = UtilitySource.cumulative_integrals(u)
    schedules = {}
    for sched in ["2", "5", "15", "50"]:
        schedules[sched] = look_indices(sched, n)
    schedules["irregular"] = irregular_look_indices(n, seed=23)
    t_common = min(int(idx[-1]) for idx in schedules.values())
    totals = {}
    for name, idx in schedules.items():
        idx2 = idx[idx <= t_common]
        if idx2[-1] != t_common:
            idx2 = np.append(idx2, t_common)
        ip, im = UtilitySource.interval_integrals(cp, cm, idx2)
        totals[name] = (float(ip.sum()), float(im.sum()))
    ref_p, ref_m = totals["50"]
    for name, (p, m) in totals.items():
        assert abs(p - ref_p) <= 1e-9 * max(abs(ref_p), 1.0)
        assert abs(m - ref_m) <= 1e-9 * max(abs(ref_m), 1.0)


# ---------------------------------------------------------------------------
# 2.3 silent group invariance (and the grow_Z failure)
# ---------------------------------------------------------------------------

def _quiet_run(n_silent, mode, n, idx, seed=0):
    pm = Pacemaker(noisy=False, seed=seed)
    L = len(idx)
    groups = [
        SensoryGroup(f"silent{i}", np.zeros((1, L)), np.array([0.0]),
                     np.array([SIGMA_MIN]))
        for i in range(n_silent)
    ]
    return simulate(pm, groups, idx, n, mode=mode, store_trajectory=True,
                    drop_zero_groups=False)


@pytest.mark.parametrize("mode", ["chord", "arc"])
def test_silent_groups_bit_identical(mode):
    life = 300.0
    n = n_grid(life)
    idx = look_indices("50", n)
    base = _quiet_run(0, mode, n, idx)
    one = _quiet_run(1, mode, n, idx)
    ten = _quiet_run(10, mode, n, idx)
    for other in (one, ten):
        assert np.array_equal(base["d_traj"], other["d_traj"])
        assert np.array_equal(base["phi_traj"], other["phi_traj"])


def test_grow_Z_changes_phi():
    life = 300.0
    n = n_grid(life)
    idx = look_indices("50", n)
    one = _quiet_run(1, "grow_Z", n, idx)
    ten = _quiet_run(10, "grow_Z", n, idx)
    r1 = one["phi"] / (idx[-1] * H_W)
    r10 = ten["phi"] / (idx[-1] * H_W)
    rel = abs(r10 - r1) / r1
    assert rel > 0.05, f"grow_Z failed to change Phi: rel={rel}"


# ---------------------------------------------------------------------------
# 2.4 declared duplicate
# ---------------------------------------------------------------------------

def test_declared_copy_bit_identical():
    life = 300.0
    n = n_grid(life)
    idx = look_indices("15", n)
    L = len(idx)
    src = Source("ramp", seed=4)
    x = Receptor(src, 0.2).sample(idx, n)
    sig = calibrate_sigma_hat(src)
    pm = Pacemaker(noisy=False, seed=0)
    g = SensoryGroup("ramp", x[None, :], np.array([sig]), np.array([1e-3]))
    copy = SensoryGroup("ramp_copy", x[None, :], np.array([sig]),
                        np.array([1e-3]), copies_of="ramp")
    a = simulate(pm, [g], idx, n, mode="chord", store_trajectory=True)
    b = simulate(pm, [g, copy], idx, n, mode="chord", store_trajectory=True)
    assert np.array_equal(a["phi_traj"], b["phi_traj"])
    assert np.array_equal(a["d_traj"], b["d_traj"])


# ---------------------------------------------------------------------------
# 2.5 unit rescaling
# ---------------------------------------------------------------------------

def test_unit_rescaling():
    life = 1000.0
    n = n_grid(life)
    idx = look_indices("15", n)
    L = len(idx)
    pm = Pacemaker(noisy=False, seed=0)
    src = Source("ramp", seed=5)
    x = Receptor(src, 0.2).sample(idx, n)
    sig = calibrate_sigma_hat(src)
    # scale the rendered signal (and its noise) by 1000
    x_scaled = x * 1000.0
    sig_scaled = calibrate_sigma_hat(src) * 1000.0
    g = SensoryGroup("ramp", x[None, :], np.array([sig]), np.array([1e-3]))
    gs = SensoryGroup("ramp", x_scaled[None, :], np.array([sig_scaled]),
                      np.array([1.0]))
    a = simulate(pm, [g], idx, n, mode="chord")
    b = simulate(pm, [gs], idx, n, mode="chord")
    diff = abs(a["phi"] - b["phi"])
    print(f"unit rescaling max Phi difference: {diff}")
    assert diff <= 1.0


# ---------------------------------------------------------------------------
# 2.6 coalescing and conservation
# ---------------------------------------------------------------------------

def test_coalescing_and_conservation():
    life = 400.0
    n = n_grid(life)
    idx = look_indices("50", n)
    pm = Pacemaker(noisy=False, seed=0)
    u = UtilitySource([(50.0, 52.0, -5.0), (200.0, 201.0, 0.4)]).grid(n)
    cp, cm = UtilitySource.cumulative_integrals(u)
    ip, im = UtilitySource.interval_integrals(cp, cm, idx)
    dth = Pacemaker.dtheta(pm.phases(n), idx)
    dp = Pacemaker.p_raw_increments(dth)
    out = simulate(pm, [], idx, n, mode="chord", utility_ip=ip, utility_im=im,
                   dp_raw=dp, record_anchors=True, store_trajectory=True)
    m = out["m_traj"]
    n_zero = int(np.sum(m == 0))
    assert n_zero > 0
    anchors = out["anchors"]
    assert len(anchors["m"]) == int(np.sum(m >= 1))
    assert np.all(anchors["m"] >= 1)
    total_p = out["total_I_plus"]
    total_m = out["total_I_minus"]
    total_dp = out["total_dP_raw"]
    cons_p = anchors["I_plus"].sum() + out["pending_I_plus"]
    cons_m = anchors["I_minus"].sum() + out["pending_I_minus"]
    cons_dp = anchors["dP_raw"].sum() + out["pending_dP_raw"]
    assert abs(cons_p - total_p) <= 1e-9 * max(abs(total_p), 1.0)
    assert abs(cons_m - total_m) <= 1e-9 * max(abs(total_m), 1.0)
    assert abs(cons_dp - total_dp) <= 1e-9 * max(abs(total_dp), 1.0)
    # no anchor at an m = 0 look: all anchor look indices have m >= 1
    assert np.all(m[anchors["look_index"]] >= 1)


# ---------------------------------------------------------------------------
# 2.7 non-freezing guarantee
# ---------------------------------------------------------------------------

def test_non_freezing_assertion():
    assert_non_freezing(0.25)
    with pytest.raises(AssertionError):
        assert_non_freezing(1.5)


# ---------------------------------------------------------------------------
# 2.8 P_raw
# ---------------------------------------------------------------------------

def test_p_raw_positive_and_schedule_agreement():
    life = 1000.0
    n = n_grid(life)
    pm = Pacemaker(noisy=False, seed=0)
    ph = pm.phases(n)
    schedules = {s: look_indices(s, n) for s in ["2", "5", "15", "50"]}
    schedules["irregular"] = irregular_look_indices(n, seed=29)
    praw_end = {}
    for name, idx in schedules.items():
        dth = Pacemaker.dtheta(ph, idx)
        pm.assert_positive(dth)
        assert np.all(dth[:, 1:] > 0.0)
        inc = Pacemaker.p_raw_increments(dth)
        praw_end[name] = float(Pacemaker.p_raw(dth)[-1])
    # compare at a common covered time
    t_common = min(int(idx[-1]) for idx in schedules.values())
    expected = t_common * H_W * sum(1.0 / p for p in PERIODS)
    for name, idx in schedules.items():
        idx2 = idx[idx <= t_common]
        if idx2[-1] != t_common:
            idx2 = np.append(idx2, t_common)
        dth = Pacemaker.dtheta(ph, idx2)
        p = float(Pacemaker.p_raw(dth)[-1])
        assert abs(p - expected) <= 1e-6, (name, p, expected)


# ---------------------------------------------------------------------------
# an earlier revision: output path is inside the project root
# ---------------------------------------------------------------------------

def test_output_folder_inside_project():
    import os
    from step0a.rigs import run_item1 as rig
    assert os.path.commonpath([rig.OUT, rig.ROOT]) == rig.ROOT
    assert rig.OUT.endswith(os.path.join("runs", "out"))
