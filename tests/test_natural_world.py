"""Tests A1-A6, B1-B4, C1-C3, D1-D3: the natural late consequences.

Every test prints its numbers so that ``pytest -s`` shows them.

The known-answer tables are read verbatim from ``tests/data/regrowth_table.txt``.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from amoeba.feelings import fatigue_step, strength
from amoeba.world import Config, FoodField, FoodKind, World

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from make_regression_default import run_regression  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


FORMULA_CHECK = DATA / "regrowth_table.txt"

FAR = (100.0, 100.0)


# -- shared helpers ------------------------------------------------------------
def _actuators(world: World):
    return [item for item in world.items() if item.kind == "actuator"]


def _schedule(actuators, t: float) -> dict[str, float]:
    """The regression test's target schedule, clipped to the item limits."""
    out = {}
    for j, item in enumerate(actuators):
        value = 2.0 * math.sin(0.7 * t + 0.9 * j)
        value = item.low if value < item.low else (item.high if value > item.high else value)
        out[item.name] = value
    return out


def _zero_targets(world: World) -> None:
    world.write({item.name: 0.0 for item in world.items() if item.kind == "actuator"})


def _regrowth_table() -> list[tuple[float, float, float, float, float, float]]:
    """(tau, sigma, x0, x1, closed-form time, stepped time) rows, verbatim."""
    rows = []
    inside = False
    for line in FORMULA_CHECK.read_text().splitlines():
        if line.startswith("REGROWTH TIMES"):
            inside = True
            continue
        if not inside:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            break
        rows.append(tuple(float(part) for part in parts))
    if not rows:
        raise AssertionError(f"no REGROWTH TIMES rows in {FORMULA_CHECK}")
    return rows


def _fatigue_table() -> list[tuple[float, ...]]:
    """(F0, effort, threshold, gain, tau, t, exact, RK4) rows, verbatim."""
    rows = []
    inside = False
    for line in FORMULA_CHECK.read_text().splitlines():
        if line.startswith("FATIGUE"):
            inside = True
            continue
        if not inside:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        rows.append(tuple(float(part) for part in parts))
    if not rows:
        raise AssertionError(f"no FATIGUE rows in {FORMULA_CHECK}")
    return rows


def _food_field(tau: float, sigma: float, x0: float, mode: str = "logistic") -> FoodField:
    cfg = Config(
        seed=0,
        n_patches=1,
        regrowth_mode=mode,
        regrowth_seed=sigma,
        regrowth_tau_s=tau,
        drift_interval_s=1e9,
    )
    food = FoodField(cfg, np.random.default_rng(0))
    food.amounts[:] = x0 * food.cap_per_particle
    food._rebuild()
    return food


def _first_time_all_at_least(food: FoodField, x1: float, limit_s: float = 20000.0) -> float:
    cap = food.cap_per_particle
    while food.time <= limit_s:
        food.step(0.01, FAR, food.config.preset_constants.intake_rate, 1.0, 0.0)
        if float(food.amt.min()) / cap >= x1:
            return food.time
    raise AssertionError(f"x1={x1!r} not reached within {limit_s!r} s")


# -- Part A: logistic regrowth -------------------------------------------------
def test_a1_default_regression_is_bit_for_bit():
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
            f"A1 {filename}: readings={readings.shape} names_equal={same_names} "
            f"readings_equal={same_readings} energy_equal={same_energy}"
        )
        assert same_names
        assert same_readings
        assert same_energy


def test_a2_logistic_regrowth_times_match_the_table():
    table = _regrowth_table()
    worst = 0.0
    for tau, sigma, x0, x1, closed_time, stepped_time in table:
        food = _food_field(tau, sigma, x0, "logistic")
        found = _first_time_all_at_least(food, x1)
        err = abs(found - closed_time)
        err_stepped = abs(found - stepped_time)
        worst = max(worst, err)
        print(
            f"A2 tau={tau:g} sigma={sigma:g} x0={x0:g} x1={x1:g} "
            f"found={found!r} closed={closed_time!r} stepped={stepped_time!r} "
            f"err={err!r} err_stepped={err_stepped!r}"
        )
        assert err <= 0.02
        assert err_stepped <= 0.02
    print(f"A2 worst closed-form error over {len(table)} rows: {worst!r}")


def test_a3_regrowth_checker_fails_on_exp_mode():
    table = _regrowth_table()
    failures = []
    with pytest.raises(AssertionError):
        for tau, sigma, x0, x1, closed_time, _stepped_time in table:
            food = _food_field(tau, sigma, x0, "exp")
            found = _first_time_all_at_least(food, x1)
            failures.append((tau, sigma, x0, x1, found, closed_time))
            assert abs(found - closed_time) <= 0.02
    print(f"A3 the exp-mode checker raised AssertionError; rows tried: {len(failures)}")


def test_a4_logistic_step_is_independent_of_dt():
    x0 = 0.05
    fine = _food_field(120.0, 1e-3, x0)
    coarse = _food_field(120.0, 1e-3, x0)
    for _ in range(30000):
        fine.step(0.01, FAR, 0.05, 1.0, 0.0)
    for _ in range(6000):
        coarse.step(0.05, FAR, 0.05, 1.0, 0.0)
    fine_total = float(fine.amt.sum())
    coarse_total = float(coarse.amt.sum())
    relative = abs(fine_total - coarse_total) / abs(fine_total)
    print(
        f"A4 after 300 s from x0={x0!r}: dt=0.01 total={fine_total!r}, "
        f"dt=0.05 total={coarse_total!r}, relative difference={relative!r}"
    )
    assert relative <= 1e-9


def test_a5_bare_particle_is_not_a_dead_end():
    tau, sigma = 120.0, 1e-6
    food = _food_field(tau, sigma, 0.0)
    closed_half = tau / (1.0 + sigma) * math.log(
        ((0.5 + sigma) * (1.0 - 0.0)) / ((1.0 - 0.5) * (0.0 + sigma))
    )
    limit = 2.0 * closed_half
    capacity = float(food.cap.sum())
    while food.time <= limit:
        food.step(0.01, FAR, 0.05, 1.0, 0.0)
        if float(food.amt.sum()) > 0.5 * capacity:
            break
    stock = float(food.amt.sum())
    print(
        f"A5 from all zero: closed-form 0->0.5 time={closed_half!r} s, limit={limit!r} s; "
        f"found stock={stock!r} of capacity={capacity!r} at t={food.time!r} s"
    )
    assert stock > 0.5 * capacity
    assert food.time < limit


def test_a6_eating_still_works_in_logistic_mode():
    world = World(
        Config(
            seed=0,
            preset="lean",
            n_patches=1,
            regrowth_mode="logistic",
            regrowth_tau_s=120.0,
            regrowth_seed=1e-3,
            initial_energy=0.9,
        )
    )
    _zero_targets(world)
    world.advance(120.0)
    after_wait = float(world.energy)
    eaten_during_wait = float(world.food_eaten)
    world.hold_food_at_mouth(True)
    world.advance(80.0)
    print(
        f"A6 t=0 energy=0.9; at t={120.0!r} after the wait energy={after_wait!r} "
        f"(eaten={eaten_during_wait!r}); hold from t=120.0 to {world.time!r}: "
        f"energy={world.energy!r}, food_eaten={world.food_eaten!r}"
    )
    assert after_wait < 0.9
    assert world.energy > 0.9
    assert world.food_eaten > 0.0


# -- Part B: tiredness ---------------------------------------------------------
def test_b1_fatigue_step_known_answers():
    # (a) effort below threshold: pure decay from 0.8, tau 600.
    f = 0.8
    for _ in range(60000):
        f = fatigue_step(f, 0.2, 0.01, 0.5, 0.02, 600.0)
    expected_a = 0.8 * math.exp(-1.0)
    print(f"B1a effort below threshold: F={f!r}, 0.8*exp(-1)={expected_a!r}")
    assert abs(f - expected_a) <= 1e-9

    # (b) effort 1.0, threshold 0.5, gain 0.02, tau 100: F_inf = 1.
    f = 0.0
    for _ in range(10000):
        f = fatigue_step(f, 1.0, 0.01, 0.5, 0.02, 100.0)
    expected_b = 1.0 - math.exp(-1.0)
    print(f"B1b F0=0, 100 s at effort 1.0: F={f!r}, 1-exp(-1)={expected_b!r}")
    assert abs(f - expected_b) <= 1e-9

    # (c) one 100 s step equals 10,000 steps of 0.01 s.
    one = fatigue_step(0.0, 1.0, 100.0, 0.5, 0.02, 100.0)
    many = 0.0
    for _ in range(10000):
        many = fatigue_step(many, 1.0, 0.01, 0.5, 0.02, 100.0)
    print(f"B1c one 100 s step={one!r}, 10,000 x 0.01 s={many!r}, difference={one - many!r}")
    assert abs(one - many) <= 1e-9

    # (d) the FATIGUE table of check.
    table = _fatigue_table()
    worst = 0.0
    for f0, effort, threshold, gain, tau, t, exact, rk4 in table:
        value = fatigue_step(f0, effort, t, threshold, gain, tau)
        worst = max(worst, abs(value - exact), abs(value - rk4))
        print(
            f"B1d F0={f0:g} effort={effort:g} threshold={threshold:g} gain={gain:g} "
            f"tau={tau:g} t={t:g}: fatigue_step={value!r} table={exact!r} rk4={rk4!r}"
        )
        assert abs(value - exact) <= 1e-6
        assert abs(value - rk4) <= 1e-6
    print(f"B1d worst table error over {len(table)} rows: {worst!r}")


def _check_fatigue_in_world(weight: float, fatigue_weight: float) -> dict:
    """B2's checker: the model gain must carry the asserted fatigue weight.

    The world rewrites the model strength only when it changed by more than
    1e-12, so the gain in force at a food step is built from the energy and F
    of the food step at which the model was *last* rewritten.  The checker
    replays that rule with the world's own weight and compares the gain with
    the asserted weight's multiplier on the last-rewritten pair.
    """
    cfg = Config(
        seed=3,
        preset="lean",
        n_patches=12,
        fatigue_weight=fatigue_weight,
        fatigue_threshold=0.0,
        fatigue_gain_per_s=0.02,
        fatigue_tau_s=100.0,
    )
    world = World(cfg)
    kp = cfg.kp
    model_mult = 1.0
    last_energy, last_fatigue = 1.0, 0.0
    worst = 0.0
    drop = 0.0
    efforts = []
    for _ in range(12000):
        world.write(_schedule(_actuators(world), world.time))
        energy_used = float(world.energy)
        fatigue_used = float(world.fatigue)
        world.advance(0.01)
        efforts.append(float(world.effort_sample))
        candidate = strength(energy_used, world.preset, cfg.weakness) * (
            1.0 - fatigue_weight * fatigue_used
        )
        if abs(candidate - model_mult) > 1e-12:
            model_mult = candidate
            last_energy, last_fatigue = energy_used, fatigue_used
        gain = float(world.body.model.actuator_gainprm[0, 0]) / kp
        expected = strength(last_energy, world.preset, cfg.weakness) * (
            1.0 - weight * last_fatigue
        )
        worst = max(worst, abs(gain - expected))
        drop = max(drop, abs(strength(energy_used, world.preset, cfg.weakness) - gain))
    mean_effort = float(np.mean(efforts))
    print(
        f"B2 weight={weight!r} world fatigue_weight={fatigue_weight!r}: "
        f"end F={world.fatigue!r}, mean effort={mean_effort!r}, "
        f"worst |gain - expected|={worst!r}, max |strength(energy) - gain|={drop!r}, "
        f"last rewritten pair energy={last_energy!r} F={last_fatigue!r}"
    )
    assert worst <= 1e-12
    assert world.fatigue > 0.05
    assert drop > 1e-3
    return {"fatigue": world.fatigue, "mean_effort": mean_effort, "worst": worst}


def test_b2_world_fatigue_scales_the_actuators():
    values = _check_fatigue_in_world(0.5, 0.5)
    print(f"B2 F={values['fatigue']!r} mean effort={values['mean_effort']!r}")


def test_b3_fatigue_checker_fails_on_a_fatigue_free_life():
    with pytest.raises(AssertionError):
        _check_fatigue_in_world(0.5, 0.0)
    print("B3 the weight-0.5 checker raised AssertionError on a fatigue_weight=0.0 life")


def test_b4_tired_sense():
    plain = World(Config(seed=3, n_patches=1))
    plain_read = plain.read()
    plain_items = {item.name: item for item in plain.items()}
    print(
        f"B4 default: tired in readings={'tired' in plain_read}, "
        f"tired in items={'tired' in plain_items}"
    )
    assert "tired" not in plain_read
    assert "tired" not in plain_items

    sensed = World(
        Config(
            seed=3,
            n_patches=1,
            fatigue_weight=0.5,
            fatigue_threshold=0.0,
            fatigue_gain_per_s=0.02,
            fatigue_tau_s=100.0,
            fatigue_sense=True,
        )
    )
    for _ in range(2000):
        sensed.write(_schedule(_actuators(sensed), sensed.time))
        sensed.advance(0.01)
    reading = sensed.read()
    items = {item.name: item for item in sensed.items()}
    print(
        f"B4 sensed: tired={reading['tired']!r} world.F={sensed.fatigue!r} "
        f"item=({items['tired'].kind!r}, {items['tired'].low!r}, {items['tired'].high!r})"
    )
    assert "tired" in reading
    assert reading["tired"] == sensed.fatigue
    assert items["tired"].kind == "sense"
    assert items["tired"].low == 0.0 and items["tired"].high == 1.0
    assert reading["tired"] > 0.0


# -- Part C: pulse patches -----------------------------------------------------
def _pulse_inside(t: float, period: float, on: float, phase: float) -> bool:
    if t < phase:
        return False
    k = math.floor((t - phase) / period)
    return (t - (phase + k * period)) < on


def _pulse_log(period: float, on: float, phase: float, seconds: float = 350.0):
    cfg = Config(
        seed=0,
        n_patches=0,
        pulse_patches=1,
        pulse_period_s=period,
        pulse_on_s=on,
        pulse_phase_s=phase,
        drift_interval_s=1e9,
    )
    food = FoodField(cfg, np.random.default_rng(0))
    log = []
    steps = int(round(seconds / 0.01)) + 1
    for _ in range(steps):
        food.step(0.01, FAR, 0.05, 1.0, 0.0)
        log.append((food.time, float(food.totals()[0])))
    return log, float(cfg.patch_capacity)


def _check_pulse_timing(log, capacity, period, on, phase) -> list[float]:
    for t, total in log:
        if _pulse_inside(t, period, on, phase):
            assert abs(total - capacity) <= 1e-9, f"t={t!r}: on but total={total!r}"
        else:
            assert total == 0.0, f"t={t!r}: off but total={total!r}"
    edges = []
    previous = None
    for t, total in log:
        state = total > 0.0
        if previous is not None and state != previous:
            edges.append(t)
        previous = state
    expected = []
    k = 0
    while True:
        start = phase + k * period
        if start > log[-1][0] + 0.01:
            break
        expected.extend([start, start + on])
        k += 1
    expected = [e for e in expected if e <= log[-1][0] + 0.01]
    assert len(edges) == len(expected), f"edges={edges!r} expected={expected!r}"
    for found, want in zip(edges, expected):
        assert abs(found - want) <= 0.01 + 1e-9, f"edge {found!r} != {want!r}"
    return edges


def test_c1_pulse_windows_switch_on_and_off():
    log, capacity = _pulse_log(100.0, 20.0, 30.0)
    edges = _check_pulse_timing(log, capacity, 100.0, 20.0, 30.0)
    print(
        f"C1 period=100 on=20 phase=30: capacity={capacity!r}, "
        f"switch times={edges!r}; expected 30, 50, 130, 150, 230, 250, 330, 350 "
        f"within one food step"
    )


def test_c2_pulse_checker_fails_on_a_wrong_period():
    log, capacity = _pulse_log(90.0, 20.0, 30.0)
    with pytest.raises(AssertionError):
        _check_pulse_timing(log, capacity, 100.0, 20.0, 30.0)
    print("C2 the period-100 checker raised AssertionError on a period-90 world")


def test_c3_pulse_patches_do_not_disturb_the_ordinary_patches():
    common = dict(
        seed=4,
        n_patches=12,
        regrowth_mode="logistic",
        regrowth_seed=1e-3,
        regrowth_tau_s=120.0,
    )
    plain = FoodField(Config(**common), np.random.default_rng(4))
    mixed_cfg = Config(**common, pulse_patches=2, pulse_period_s=600, pulse_on_s=60)
    mixed = FoodField(mixed_cfg, np.random.default_rng(4))
    for step in range(20000):
        plain.step(0.01, FAR, 0.05, 1.0, 0.0)
        mixed.step(0.01, FAR, 0.05, 1.0, 0.0)
        if not np.array_equal(plain.amounts, mixed.amounts[:12]):
            raise AssertionError(f"ordinary amounts differ at step {step}")
    print(
        f"C3 after 200 s: 12 ordinary patches bit for bit equal with and without "
        f"2 pulse patches; plain total={float(plain.amounts.sum())!r}, "
        f"mixed ordinary total={float(mixed.amounts[:12].sum())!r}"
    )


# -- Part D: snapshot, restore, change_body ------------------------------------
D1_KINDS = (
    FoodKind("plain", 0.0, 1.0, 0.5),
    FoodKind("late", 40.0, 1.0, 0.5),
)


def _d1_config() -> Config:
    return Config(
        seed=5,
        preset="lean",
        n_patches=12,
        regrowth_mode="logistic",
        regrowth_seed=1e-3,
        fatigue_weight=0.5,
        fatigue_threshold=0.0,
        fatigue_tau_s=100.0,
        fatigue_sense=True,
        pulse_patches=2,
        pulse_period_s=80,
        pulse_on_s=20,
        food_kinds=D1_KINDS,
    )


def _scheduled_run(world: World, seconds: float, names: list[str] | None = None):
    actuators = _actuators(world)
    readings = []
    energy = []
    pulse = []
    for _ in range(int(round(seconds / 0.1))):
        world.write(_schedule(actuators, world.time))
        world.advance(0.1)
        reading = world.read()
        if names is None:
            names = list(reading)
        if list(reading) != names:
            raise AssertionError("reading names changed during the run")
        readings.append([reading[name] for name in names])
        energy.append(float(world.energy))
        pulse.append(float(world.food.amounts[world.config.n_patches:].sum()))
    return np.array(readings), np.array(energy), np.array(pulse), names


def test_d1_snapshot_covers_fatigue_pulses_and_logistic_regrowth():
    world = World(_d1_config())
    names = None
    _, _, _, names = _scheduled_run(world, 50.0, names)
    snap = world.snapshot()
    snapshot_f = float(world.fatigue)
    snapshot_pulse = (float(world.food.pulse_time), bool(world.food.pulse_on))
    print(
        f"D1 at t={world.time!r}: F={snapshot_f!r}, "
        f"pulse clock={snapshot_pulse[0]!r} on={snapshot_pulse[1]!r}"
    )
    path = DATA / "_d1_snapshot.pkl"
    world.save_snapshot(path)
    try:
        run1, energy1, pulse1, names = _scheduled_run(world, 100.0, names)
        on_switch = any(p > 0.0 for p in pulse1)
        first_on = next((i for i, p in enumerate(pulse1) if p > 0.0), None)
        off_switch = (
            first_on is not None
            and any(p == 0.0 for p in pulse1[first_on:])
        )
        print(
            f"D1 run 1 t={world.time!r}: pulse on-switch={on_switch} "
            f"off-switch={off_switch} max pulse={float(pulse1.max())!r} "
            f"F={world.fatigue!r}"
        )
        assert on_switch and off_switch
        assert world.fatigue > 0.0

        world.restore(snap)
        run2, energy2, pulse2, names2 = _scheduled_run(world, 100.0, names)
        same = (
            np.array_equal(run1, run2)
            and np.array_equal(energy1, energy2)
            and np.array_equal(pulse1, pulse2)
        )
        print(f"D1 restore in place: readings_equal={np.array_equal(run1, run2)} "
              f"energy_equal={np.array_equal(energy1, energy2)} "
              f"pulse_equal={np.array_equal(pulse1, pulse2)}")

        other = World(_d1_config())
        other.load_snapshot(path)
        run3, energy3, pulse3, _ = _scheduled_run(other, 100.0, names)
        same_file = (
            np.array_equal(run1, run3)
            and np.array_equal(energy1, energy3)
            and np.array_equal(pulse1, pulse3)
        )
        print(f"D1 file load into a new world: readings_equal={np.array_equal(run1, run3)} "
              f"energy_equal={np.array_equal(energy1, energy3)} "
              f"pulse_equal={np.array_equal(pulse1, pulse3)}")
        assert same
        assert same_file
    finally:
        if path.exists():
            path.unlink()


def test_d2_wrong_fatigue_or_pulse_state_changes_the_run():
    world = World(_d1_config())
    names = None
    _, _, _, names = _scheduled_run(world, 50.0, names)
    snap = world.snapshot()
    run1, energy1, _, names = _scheduled_run(world, 100.0, names)

    bad_fatigue = copy.deepcopy(snap)
    bad_fatigue["fatigue"] = 0.0
    world.restore(bad_fatigue)
    run2, energy2, _, _ = _scheduled_run(world, 100.0, names)
    differs_f = not (np.array_equal(run1, run2) and np.array_equal(energy1, energy2))

    fresh = World(_d1_config())
    bad_pulse = copy.deepcopy(snap)
    bad_pulse["food"]["pulse"] = {
        "time": float(fresh.food.pulse_time),
        "on": bool(fresh.food.pulse_on),
    }
    world.restore(bad_pulse)
    run3, energy3, _, _ = _scheduled_run(world, 100.0, names)
    differs_p = not (np.array_equal(run1, run3) and np.array_equal(energy1, energy3))

    print(
        f"D2 F set to 0.0 (snapshot F={snap['fatigue']!r}): run differs={differs_f}; "
        f"fresh pulse state (clock={fresh.food.pulse_time!r}, on={fresh.food.pulse_on!r}) "
        f"against snapshot clock={snap['food']['pulse']['time']!r}: run differs={differs_p}"
    )
    assert differs_f
    assert differs_p


def test_d3_change_body_keeps_fatigue_and_pulses():
    world = World(_d1_config())
    _scheduled_run(world, 50.0)
    snap = world.snapshot()
    world.restore(snap)
    _scheduled_run(world, 100.0)
    fatigue_before = float(world.fatigue)
    world.change_body(4, 2)
    assert world.fatigue == fatigue_before
    actuators = _actuators(world)
    assert len(actuators) == 8
    pulse_seen = 0.0
    for _ in range(20):
        world.write(_schedule(actuators, world.time))
        world.advance(1.0)
        reading = world.read()
        assert all(np.isfinite(list(reading.values())))
        assert "tired" in reading
        assert reading["tired"] == world.fatigue
        pulse_seen = max(pulse_seen, float(world.food.amounts[world.config.n_patches:].sum()))
    print(
        f"D3 after change_body(4, 2) at t={world.time!r}: F={world.fatigue!r} "
        f"(before change {fatigue_before!r}), actuators=8, tired present, "
        f"max pulse stock in the 20 s window={pulse_seen!r}"
    )
    assert pulse_seen > 0.0
    assert 0.0 <= world.fatigue <= 1.0
