"""World-level checks for section 3.2 that the numbered tests do not cover."""

import math

import numpy as np

from amoeba.world import Config, World


def test_advance_rounds_and_carries():
    w = World(Config(seed=0, n_patches=0, timestep=0.002))
    w.advance(0.003)
    t1 = w.time
    w.advance(0.001)
    t2 = w.time
    w.advance(0.001)
    t3 = w.time
    w.advance(0.001)
    t4 = w.time
    print(f"advance(0.003) -> t={t1!r}; then 0.001 -> {t2!r}, 0.001 -> {t3!r}, 0.001 -> {t4!r}")
    assert abs(t1 - 0.004) < 1e-12
    assert abs(t2 - 0.004) < 1e-12
    assert abs(t3 - 0.006) < 1e-12
    assert abs(t4 - 0.006) < 1e-12


def test_time_after_many_calls():
    w = World(Config(seed=0, n_patches=0, timestep=0.002))
    for _ in range(1000):
        w.advance(0.01)
    print(f"1000 x advance(0.01): time={w.time!r}")
    assert abs(w.time - 10.0) < 1e-9


def test_food_regrows_toward_capacity():
    w = World(Config(seed=1, n_patches=2, particles_per_patch=5, regrowth_tau_s=10.0))
    before = w.food.totals().copy()
    w.food.amounts[:] = 0.0
    w.advance(10.0)
    after = w.food.totals().copy()
    cap = w.config.patch_capacity
    print(f"food totals before={before!r} after 10 s from zero={after!r} capacity={cap!r}")
    assert np.all(after < cap)
    assert np.all(after > 0.5 * cap)


def test_smell_falls_with_distance():
    w = World(Config(seed=2, n_patches=1, particles_per_patch=1))
    w.food.pin_patch(0, (0.0, 0.0))
    near = float(w.food.smell([[0.1, 0.0]])[0])
    far = float(w.food.smell([[0.6, 0.0]])[0])
    print(f"smell at 0.1 m={near!r}, at 0.6 m={far!r}")
    assert near > far > 0.0


def test_eating_adds_energy_and_removes_food():
    w = World(Config(seed=3, n_patches=1, particles_per_patch=5))
    w.food.pin_patch(0, w.body.centre())
    w.energy = 0.5
    w.hold_food_at_mouth(True)
    before = w.food.totals()[0]
    w.advance(1.0)
    print(f"energy 0.5 -> {w.energy!r}, food_eaten={w.food_eaten!r}")
    assert w.energy > 0.5
    assert w.food_eaten > 0.0
    assert w.food.totals()[0] <= before + 1e-9


def test_body_cannot_leave_the_dish():
    w = World(Config(seed=4, n_patches=0))
    w.body.data.qpos[:3] = [1.2, 0.0, 0.0]
    w.body.data.qvel[:3] = [8.0, 3.0, 0.0]
    max_r = 0.0
    for _ in range(300):
        w.advance(0.02)
        x, y = w.body.centre()
        max_r = max(max_r, math.hypot(x, y))
    final_r = math.hypot(*w.body.centre())
    w.body.data.qvel[:3] = [0.0, 3.0, 0.0]
    x1, y1 = w.body.centre()
    w.advance(2.0)
    x2, y2 = w.body.centre()
    slide = math.hypot(x2 - x1, y2 - y1)
    print(
        f"fired at wall: max body-centre radius={max_r!r}, final radius={final_r!r}, "
        f"slide along wall={slide!r}, dish radius={w.config.dish_radius!r}"
    )
    assert max_r <= w.config.dish_radius
    assert slide > 0.02
    assert all(np.isfinite(list(w.read().values())))


def test_same_seed_same_patch_layout():
    a = World(Config(seed=42, n_patches=6))
    b = World(Config(seed=42, n_patches=6))
    print(f"same seed centres equal={np.array_equal(a.food.centres, b.food.centres)}")
    assert np.array_equal(a.food.centres, b.food.centres)
    assert np.array_equal(a.food.amounts, b.food.amounts)


def test_drifting_patches_move_and_fixed_patches_do_not():
    w = World(Config(seed=20, n_patches=6, fixed_fraction=0.5, drift_interval_s=1.0))
    before = w.food.centres.copy()
    fixed_before = before[w.food.fixed].copy()
    w.advance(2.5)
    after = w.food.centres.copy()
    moved = np.linalg.norm(after[~w.food.fixed] - before[~w.food.fixed], axis=1)
    print(
        f"fixed patches moved by {np.linalg.norm(after[w.food.fixed] - fixed_before, axis=1)!r}; "
        f"drifting patches moved by {moved!r}"
    )
    assert np.allclose(after[w.food.fixed], fixed_before)
    assert np.all(moved > 0.0)


def test_feeling_cannot_be_written():
    w = World(Config(seed=21, n_patches=0))
    try:
        w.write({"energy.need": 1.0})
    except KeyError:
        print("write({'energy.need': 1.0}) raised KeyError: the body owns the feeling")
    else:
        raise AssertionError("energy.need was writable")
