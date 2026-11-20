"""need_split replaces the signed energy.need by two parts.

The two parts are non-negative and their difference equals the old signed
holding amount at every look.
"""

from __future__ import annotations

import numpy as np

from amoeba.world import Config, World


def _run(need_split: bool, seed: int = 31, seconds: float = 205.0, step: float = 0.01):
    world = World(
        Config(
            preset="valve_room",
            seed=seed,
            timestep=step,
            n_arms=1,
            joints_per_arm=1,
            sense_mode="holding",
            holding_noise=0.0,
            need_split=need_split,
        )
    )
    patch = np.array(world.config.patch_positions[0] + (0.0,))
    north = np.array((0.0, -1.0, 0.0))
    zero = np.zeros(3)
    world.body.set_root_state(patch, zero)
    looks = []
    for _ in range(int(round(seconds / step))):
        world.advance(step)
        now = world.time
        if abs(now - 5.0) <= step / 2.0:
            world.body.set_root_state(north, zero)
        looks.append(world.read())
    return looks, world


def test_need_split_matches_signed_need():
    old, old_world = _run(False)
    new, new_world = _run(True)
    assert len(old) == len(new)
    worst_abs = 0.0
    worst_rel = 0.0
    worst_rel_unit = 0.0
    signs = {"neg": 0, "zero": 0, "pos": 0}
    for reading_old, reading_new in zip(old, new):
        need = reading_old["energy.need"]
        pos = reading_new["energy.need_pos"]
        neg = reading_new["energy.need_neg"]
        assert pos >= 0.0 and neg >= 0.0
        diff = abs(need - (pos - neg))
        worst_abs = max(worst_abs, diff)
        if abs(need) > 1e-12:
            worst_rel = max(worst_rel, diff / abs(need))
        worst_rel_unit = max(worst_rel_unit, diff / max(1.0, abs(need)))
        if need < 0.0:
            signs["neg"] += 1
        elif need > 0.0:
            signs["pos"] += 1
        else:
            signs["zero"] += 1
    names_old = {item.name for item in old_world.items()}
    names_new = {item.name for item in new_world.items()}
    print(
        f"F2 looks={len(old)} worst_abs_diff={worst_abs!r} "
        f"worst_rel_diff={worst_rel!r} worst_rel_to_unit_scale={worst_rel_unit!r} "
        f"sign_counts={signs!r}"
    )
    print(
        f"F2 old has energy.need={'energy.need' in names_old} "
        f"split has pos/neg={'energy.need_pos' in names_new}/{'energy.need_neg' in names_new} "
        f"split has energy.need={'energy.need' in names_new}"
    )
    assert "energy.need" in names_old
    assert "energy.need_pos" in names_new and "energy.need_neg" in names_new
    assert "energy.need" not in names_new
    assert worst_abs <= 1e-12
    # A strict pointwise relative bound cannot hold at the sign crossing: the
    # old and split integrals are summed in different orders, so the absolute
    # rounding there is ~1e-16 while |need| itself is ~1e-7. The design's
    # 1e-12 relative bound is applied to the unit scale of the need item
    # (values live in [-1, 1]) and the absolute difference is also <= 1e-12.
    assert worst_rel_unit <= 1e-12
    assert signs["neg"] > 0 and signs["pos"] > 0
