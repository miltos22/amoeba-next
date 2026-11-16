"""Test 4.4: energy's resting point, its decay rate and life at zero."""

import math

import numpy as np

from amoeba.feelings import PRESETS
from amoeba.world import Config, World


def _steady(preset: str):
    w = World(Config(seed=10, preset=preset, n_patches=1))
    w.hold_food_at_mouth(True)
    w.advance(120.0)
    samples = []
    for _ in range(600):
        w.advance(1.0)
        samples.append(w.energy)
    samples = np.array(samples)
    expected = 1.0 - PRESETS[preset].basal_rate / PRESETS[preset].intake_rate
    spread = float(samples.max() - samples.min())
    print(
        f"preset={preset}: equilibrium={samples[-1]!r} expected={expected!r} "
        f"spread over 600 s={spread!r} max={samples.max()!r}"
    )
    assert samples.max() < 1.0
    assert samples[-1] > 0.8
    assert spread < 1e-3
    return samples[-1]


def test_energy_settles_below_one_with_food_at_mouth():
    _steady("lean")
    _steady("rich")


def _decay(preset: str):
    w = World(Config(seed=11, preset=preset, n_patches=0))
    rate = PRESETS[preset].basal_rate
    w.advance(100.0)
    measured = (1.0 - w.energy) / 100.0
    zero_time = None
    guard = 0
    while w.energy > 0.0 and guard < 100000:
        w.advance(1.0)
        guard += 1
    zero_time = w.time
    error = abs(measured - rate) / rate
    print(
        f"preset={preset}: decay over first 100 s={measured!r} preset rate={rate!r} "
        f"relative error={error!r}; reached zero at t={zero_time!r} s "
        f"(1/basal={1.0 / rate!r} s)"
    )
    assert error < 0.01
    assert abs(zero_time - 1.0 / rate) / (1.0 / rate) < 0.01


def test_energy_falls_to_zero_no_food_no_movement():
    _decay("lean")
    _decay("rich")


def test_zero_energy_body_can_still_move_its_joints():
    w = World(Config(seed=12, preset="lean", n_patches=0))
    w.energy = 0.0
    w.body.set_energy(0.0)
    before = w.read()["arm0.j1.angle"]
    w.write({"arm0.j1.target": 0.5})
    w.advance(1.0)
    after = w.read()["arm0.j1.angle"]
    gain = float(w.body.model.actuator_gainprm[0, 0])
    limit = float(w.body.model.actuator_forcerange[0, 1])
    print(
        f"at energy=0: arm0.j1.angle {before!r} -> {after!r}, "
        f"gain={gain!r}, force_limit={limit!r}"
    )
    assert after - before > 0.05
    assert abs(gain - w.config.kp * PRESETS["lean"].strength_floor) < 1e-12
    assert abs(limit - w.config.force_limit * PRESETS["lean"].strength_floor) < 1e-12


def test_feeling_is_computed_by_the_body_and_weakness_flag():
    w = World(Config(seed=13, preset="lean", n_patches=0))
    w.energy = 0.55
    need = w.read()["energy.need"]
    print(f"energy=0.55 -> energy.need={need!r} (set point {PRESETS['lean'].energy_set_point!r})")
    assert abs(need - (0.8 - 0.55)) < 1e-12
    strong = World(Config(seed=14, preset="lean", n_patches=0, weakness=False))
    strong.energy = 0.0
    strong.body.set_energy(0.0)
    gain = float(strong.body.model.actuator_gainprm[0, 0])
    print(f"weakness=False at energy=0: gain={gain!r} (constant {strong.config.kp!r})")
    assert abs(gain - strong.config.kp) < 1e-12
