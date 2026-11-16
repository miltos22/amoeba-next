"""Tests T1-T8: food kinds whose effect arrives late.

Every test prints its numbers so that ``pytest -s`` shows them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from amoeba.world import Config, FoodKind, World

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from make_regression_default import run_regression  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def _zero_targets(world: World) -> None:
    world.write({item.name: 0.0 for item in world.items() if item.kind == "actuator"})


def _delay_life(delay: float, seconds: float = 120.0):
    world = World(
        Config(seed=0, n_patches=1, food_kinds=(FoodKind("late", delay, 1.0, 1.0),))
    )
    _zero_targets(world)
    world.hold_food_at_mouth(True)
    log = []
    for _ in range(int(round(seconds / 0.01))):
        world.advance(0.01)
        log.append(
            {"t": world.time, "eaten": world.food_eaten, "delivered": world.food_delivered}
        )
    return world, log


def assert_delay(world_log, delay: float) -> None:
    """T2's check: delivered total lags the eaten total by exactly ``delay``."""
    times = [row["t"] for row in world_log]
    step = times[1] - times[0]
    shift = delay / step
    if abs(shift - round(shift)) > 1e-9:
        raise AssertionError(f"delay {delay!r} is not a whole number of food steps")
    shift = int(round(shift))
    first_eat = next((i for i, row in enumerate(world_log) if row["eaten"] > 0.0), None)
    if first_eat is None:
        raise AssertionError("nothing was ever eaten")
    for i, row in enumerate(world_log):
        if i < first_eat + shift:
            if row["delivered"] != 0.0:
                raise AssertionError(
                    f"t={row['t']!r}: delivered={row['delivered']!r} before delay elapsed"
                )
        expected = world_log[i - shift]["eaten"] if i >= shift else 0.0
        if abs(row["delivered"] - expected) > 1e-12:
            raise AssertionError(
                f"t={row['t']!r}: delivered={row['delivered']!r} != eaten(t-{delay})={expected!r}"
            )


def test_t1_default_regression_is_bit_for_bit():
    for filename, seed, preset, n_patches in (
        ("regression_default.npz", 3, "lean", 12),
        ("regression_rich.npz", 4, "rich", 6),
    ):
        stored = np.load(DATA / filename, allow_pickle=False)
        names, readings, energy = run_regression(seed, preset, n_patches)
        same_names = list(stored["names"]) == names
        same_readings = np.array_equal(stored["readings"], readings)
        same_energy = np.array_equal(stored["energy"], energy)
        print(
            f"T1 {filename}: readings={readings.shape} names_equal={same_names} "
            f"readings_equal={same_readings} energy_equal={same_energy}"
        )
        assert same_names
        assert same_readings
        assert same_energy


def test_t2_delay_is_exact():
    world, log = _delay_life(30.0)
    assert_delay(log, 30.0)
    first_eat = next(row["t"] for row in log if row["eaten"] > 0.0)
    first_delivery = next(row["t"] for row in log if row["delivered"] != 0.0)
    print(
        f"T2 first eating t={first_eat!r}, first delivery t={first_delivery!r}, "
        f"difference={first_delivery - first_eat!r}, "
        f"delivered at end={world.food_delivered!r}, eaten at end={world.food_eaten!r}"
    )
    assert first_delivery > first_eat
    assert abs((first_delivery - first_eat) - 30.0) < 1e-9


def test_t3_delay_checker_can_fail():
    _world, log = _delay_life(0.0)
    print(f"T3 running the 30 s check on a delay-0 life of {len(log)} rows")
    with pytest.raises(AssertionError):
        assert_delay(log, 30.0)
    print("T3 check failed as required")


def test_t4_harm_is_hand_kept():
    world = World(
        Config(seed=0, n_patches=1, food_kinds=(FoodKind("bad", 20.0, -1.0, 1.0),))
    )
    _zero_targets(world)
    world.advance(60.0)
    world.hold_food_at_mouth(True)
    world.advance(0.01)
    start_time = world.time
    e0 = world.energy
    d0 = world.food_delivered_amount
    de0 = world.food_delivered
    dtc = world.config.timestep * world._food_every
    drain_sum = 0.0
    first_delivery = None
    for _ in range(4000):
        world.advance(0.01)
        delivered = world.food_delivered_amount - d0
        drain = dtc * (world.preset.basal_rate + world.preset.effort_rate * world._effort_sample)
        drain_sum += drain
        expected = e0 + (-1.0) * delivered - drain_sum
        assert 0.0 < world.energy < 1.0
        assert abs(world.energy - expected) <= 1e-12
        if first_delivery is None and world.food_delivered_amount != d0:
            first_delivery = world.time
    print(
        f"T4 window t={start_time!r}..{world.time!r}, e0={e0!r}, "
        f"first delivery t={first_delivery!r}, end energy={world.energy!r}, "
        f"delivered amount={world.food_delivered_amount - d0!r}, "
        f"delivered energy={world.food_delivered - de0!r}, hand-kept drain={drain_sum!r}"
    )
    assert first_delivery is not None
    assert first_delivery - start_time > 19.99


def test_t5_gut_limits_intake():
    world = World(
        Config(seed=0, n_patches=1, food_kinds=(FoodKind("late", 300.0, 1.0, 1.0),))
    )
    _zero_targets(world)
    world.hold_food_at_mouth(True)
    max_sum = 0.0
    max_gut = 0.0
    for _ in range(60000):
        world.advance(0.01)
        total = world.energy + world.gut_load
        max_sum = max(max_sum, total)
        max_gut = max(max_gut, world.gut_load)
        assert total <= 1.0 + 1e-9
    print(
        f"T5 over 600 s: max(energy+gut_load)={max_sum!r}, "
        f"max gut_load={max_gut!r}, end energy={world.energy!r}"
    )
    assert max_gut > 0.0


def _smell_readings(world: World) -> dict[str, float]:
    return world.read()


def test_t6_smell_is_per_kind():
    kinds = (FoodKind("A", 0.0, 1.0, 0.5), FoodKind("B", 0.0, 1.0, 0.5))
    world = World(Config(seed=1, n_patches=2, food_kinds=kinds))
    patch_a = int(np.flatnonzero(world.food.kind == 0)[0])
    patch_b = int(np.flatnonzero(world.food.kind == 1)[0])
    world.food.pin_patch(patch_a, (0.0, 0.0))
    world.food.pin_patch(patch_b, (0.7, 0.3))
    before = _smell_readings(world)
    names_b = ["centre.smell.B"] + [f"arm{k}.smell.B" for k in range(world.config.n_arms)]
    names_a = ["centre.smell.A"] + [f"arm{k}.smell.A" for k in range(world.config.n_arms)]
    print(f"T6 centre.smell.A={before['centre.smell.A']!r} centre.smell.B={before['centre.smell.B']!r}")
    assert before["centre.smell.A"] > 0.0

    world.food.amounts[world.food.kind == 0] = 0.0
    world.food._rebuild()
    after_a_gone = _smell_readings(world)
    print(
        f"T6 A emptied: centre.smell.A={after_a_gone['centre.smell.A']!r}, "
        f"B unchanged={all(after_a_gone[n] == before[n] for n in names_b)}"
    )
    assert after_a_gone["centre.smell.A"] == 0.0
    for name in names_b:
        assert after_a_gone[name] == before[name]

    world.food.pin_patch(patch_a, (0.0, 0.0))
    world.food.pin_patch(patch_b, (0.7, 0.3))
    both_full = _smell_readings(world)
    world.food.amounts[world.food.kind == 1] = 0.0
    world.food._rebuild()
    after_b_gone = _smell_readings(world)
    print(
        f"T6 B emptied: centre.smell.B={after_b_gone['centre.smell.B']!r}, "
        f"A unchanged={all(after_b_gone[n] == both_full[n] for n in names_a)}"
    )
    assert after_b_gone["centre.smell.B"] == 0.0
    for name in names_a:
        assert after_b_gone[name] == both_full[name]


def _counts(n_patches: int, shares) -> list[int]:
    kinds = tuple(FoodKind(f"k{i}", 0.0, 1.0, share) for i, share in enumerate(shares))
    world = World(Config(seed=2, n_patches=n_patches, food_kinds=kinds))
    return np.bincount(world.food.kind, minlength=len(shares)).tolist()


def test_t7_patch_counts_use_largest_remainder():
    cases = [
        (12, (0.5, 0.5), [6, 6]),
        (12, (0.34, 0.66), [4, 8]),
        (12, (1 / 3, 1 / 3, 1 / 3), [4, 4, 4]),
        (5, (0.5, 0.5), [3, 2]),
    ]
    for n_patches, shares, expected in cases:
        got = _counts(n_patches, shares)
        print(f"T7 n={n_patches} shares={shares} -> {got} (expected {expected})")
        assert got == expected
    print(
        "T7 5 patches 0.5/0.5 -> 3 and 2: both quotas are 2.5 with equal remainder, "
        "the tie is broken by kind order, so the earlier kind takes the extra patch"
    )


def test_t8_gut_sense_reading():
    plain = World(Config(seed=3, n_patches=1))
    assert "gut" not in plain.read()
    kinds = (FoodKind("late", 5.0, 1.0, 1.0),)
    world = World(Config(seed=3, n_patches=1, food_kinds=kinds, gut_sense=True))
    _zero_targets(world)
    world.hold_food_at_mouth(True)
    world.advance(3.0)
    reading = world.read()
    print(
        f"T8 gut reading={reading['gut']!r} gut_load={world.gut_load!r} "
        f"gut items={sum(1 for i in world.items() if i.name == 'gut')}"
    )
    assert "gut" in reading
    assert reading["gut"] == world.gut_load
    assert world.gut_load > 0.0
