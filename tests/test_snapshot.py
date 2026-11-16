"""Tests S1-S4: snapshot and restore of the world, bit for bit."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from amoeba.world import Config, FoodKind, World

KINDS = (
    FoodKind("plain", 0.0, 1.0, 0.5),
    FoodKind("late", 40.0, 1.0, 0.5),
)


def _config() -> Config:
    return Config(seed=5, preset="lean", n_patches=12, food_kinds=KINDS)


def _schedule(actuators, t: float) -> dict[str, float]:
    out = {}
    for j, item in enumerate(actuators):
        value = 2.0 * math.sin(0.7 * t + 0.9 * j)
        value = item.low if value < item.low else (item.high if value > item.high else value)
        out[item.name] = value
    return out


def _run(world: World, actuators, names, seconds: float):
    readings = []
    energy = []
    centres = []
    gut_lengths = []
    for _ in range(int(round(seconds / 0.1))):
        world.write(_schedule(actuators, world.time))
        world.advance(0.1)
        reading = world.read()
        readings.append([reading[name] for name in names])
        energy.append(world.energy)
        centres.append(world.food.centres.copy())
        gut_lengths.append(len(world.gut))
    return (
        np.array(readings),
        np.array(energy),
        np.array(centres),
        np.array(gut_lengths),
    )


def _prepare():
    world = World(_config())
    actuators = [item for item in world.items() if item.kind == "actuator"]
    names = list(world.read())
    _run(world, actuators, names, 50.0)
    return world, actuators, names


def _jump_times(centres: np.ndarray, fixed: np.ndarray, t0: float) -> list[float]:
    times = []
    for i in range(1, len(centres)):
        moved = np.any(centres[i][~fixed] != centres[i - 1][~fixed])
        if moved:
            times.append(t0 + i * 0.1)
    return times


def test_s1_snapshot_restores_bit_for_bit():
    world, actuators, names = _prepare()
    snap = world.snapshot()
    snap_centres = snap["food"]["centres"]
    assert snap["gut"], "S1 snapshot must hold gut entries in flight"
    print(
        f"S1 at t={world.time!r}: gut entries={len(snap['gut'])} "
        f"gut_load={snap['gut_load']!r} energy={world.energy!r}"
    )
    run1 = _run(world, actuators, names, 100.0)
    readings1, energy1, centres1, gut1 = run1
    print(
        f"S1 run 1 t={world.time!r}: max gut entries in window={int(gut1.max())}, "
        f"end energy={world.energy!r}"
    )
    assert gut1.max() > 0, "S1 window must contain gut entries in flight"

    fixed = snap["food"]["fixed"]
    all_centres = np.concatenate([snap_centres[None, :, :], centres1], axis=0)
    jumps = _jump_times(all_centres, fixed, t0=50.0)
    print(f"S1 patch jumps at t={jumps!r}")
    assert any(59.5 < t < 60.5 for t in jumps), "no patch jump near 60 s"
    assert any(119.5 < t < 120.5 for t in jumps), "no patch jump near 120 s"

    world.restore(snap)
    run2 = _run(world, actuators, names, 100.0)
    readings2, energy2, centres2, gut2 = run2
    same_readings = np.array_equal(readings1, readings2)
    same_energy = np.array_equal(energy1, energy2)
    same_centres = np.array_equal(centres1, centres2)
    same_gut = np.array_equal(gut1, gut2)
    print(
        f"S1 restored run 2: readings_equal={same_readings} energy_equal={same_energy} "
        f"centres_equal={same_centres} gut_lengths_equal={same_gut}"
    )
    assert same_readings
    assert same_energy
    assert same_centres
    assert same_gut


def test_s2_snapshot_copy_with_other_rng_or_empty_gut_differs():
    world, actuators, names = _prepare()
    snap = world.snapshot()
    readings1, energy1, _centres, _gut = _run(world, actuators, names, 100.0)

    bad_rng = copy.deepcopy(snap)
    bad_rng["rng_state"] = np.random.default_rng(123456).bit_generator.state
    world.restore(bad_rng)
    readings2, energy2, _c, _g = _run(world, actuators, names, 100.0)
    same = np.array_equal(readings1, readings2) and np.array_equal(energy1, energy2)
    print(f"S2 fresh rng state: runs identical={same}")
    assert not same

    bad_gut = copy.deepcopy(snap)
    bad_gut["gut"] = []
    bad_gut["gut_load"] = 0.0
    world.restore(bad_gut)
    readings3, energy3, _c, _g = _run(world, actuators, names, 100.0)
    same = np.array_equal(readings1, readings3) and np.array_equal(energy1, energy3)
    print(f"S2 emptied gut: runs identical={same}")
    assert not same


def test_s3_save_load_into_a_new_world():
    world, actuators, names = _prepare()
    path = Path(__file__).resolve().parent / "_snapshot_s3.pkl"
    try:
        world.save_snapshot(path)
        readings1, energy1, centres1, gut1 = _run(world, actuators, names, 100.0)
        other = World(_config())
        other.load_snapshot(path)
        other_actuators = [item for item in other.items() if item.kind == "actuator"]
        readings2, energy2, centres2, gut2 = _run(other, other_actuators, names, 100.0)
    finally:
        if path.exists():
            path.unlink()
    same_readings = np.array_equal(readings1, readings2)
    same_energy = np.array_equal(energy1, energy2)
    same_centres = np.array_equal(centres1, centres2)
    same_gut = np.array_equal(gut1, gut2)
    print(
        f"S3 new world after load: readings_equal={same_readings} energy_equal={same_energy} "
        f"centres_equal={same_centres} gut_lengths_equal={same_gut}"
    )
    assert same_readings
    assert same_energy
    assert same_centres
    assert same_gut


def test_s4_restore_rejects_a_changed_body_shape():
    world, _actuators, _names = _prepare()
    snap = world.snapshot()
    world.change_body(4, 2)
    with pytest.raises(ValueError) as error:
        world.restore(snap)
    print(f"S4 restore after change_body(4, 2) raised ValueError: {error.value}")
