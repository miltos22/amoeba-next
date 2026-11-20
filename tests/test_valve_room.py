"""Checks C1-C8: the two-valve room and holding senses.

Every check prints its measured numbers (run pytest with -s).  Each check
has a passing case and a must-fail twin: the twin asserts that the same
comparison does fail when it should.
"""

from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from amoeba.world import Config, World

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
DATA = HERE / "data"
C1_DIR = DATA / "valve_room_refs"


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _targets(names, t: float, scale: float = 1.0) -> dict[str, float]:
    """A fixed scripted held-target schedule: four held stages."""
    stage = int(t // 5.0)
    values = []
    for j, _name in enumerate(names):
        if stage == 0:
            value = 0.0
        elif stage == 1:
            value = 0.9 * math.sin(0.5 * j + 0.3)
        elif stage == 2:
            value = -0.8 * math.sin(0.3 * j)
        else:
            value = 0.7 * math.cos(0.8 * j)
        values.append(scale * value)
    return dict(zip(names, values))


def _regular_looks(look: float, seconds: float) -> list[float]:
    n = int(round(seconds / look))
    return [round(k * look, 9) for k in range(1, n + 1)]


def _irregular_looks(seconds: float, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    times: list[float] = []
    when = 0.0
    while True:
        when += float(rng.exponential(0.5))
        if when >= seconds:
            break
        times.append(round(when, 9))
    times.append(float(seconds))
    return sorted(set(times))


def _merge_events(looks, seconds: float):
    events: dict[float, set[str]] = {}
    for when in (0.0, 5.0, 10.0, 15.0):
        events.setdefault(round(when, 9), set()).add("write")
    for when in looks:
        events.setdefault(round(when, 9), set()).add("look")
    for when in np.arange(0.5, seconds + 1e-9, 0.5):
        events.setdefault(round(float(when), 9), set()).add("check")
    return events


def _run_schedule(looks, seconds: float, seed: int, mode: str, scale: float = 1.0):
    """One world under one look schedule; the target writes are identical.

    Returns the look sums (integral in holding mode, value x gap in instant
    mode), the 0.5 s checkpoints, and the world at the end.
    """
    world = World(Config(preset="valve_room", seed=seed, sense_mode=mode))
    names = [item.name for item in world.items() if item.kind == "actuator"]
    events = _merge_events(looks, seconds)
    sums: dict[str, float] = {}
    last_look_time = 0.0
    checks = []
    for when in sorted(events):
        gap = when - world.time
        if gap > 0.0:
            world.advance(gap)
        labels = events[when]
        if "write" in labels:
            world.write(_targets(names, when, scale))
        if "look" in labels:
            reading = world.read()
            if mode == "holding":
                increment = reading
            else:
                span = world.time - last_look_time
                increment = {name: value * span for name, value in reading.items()}
                last_look_time = world.time
            for name, value in increment.items():
                sums[name] = sums.get(name, 0.0) + value
        if "check" in labels:
            checks.append(
                (
                    world.time,
                    world.body.data.qpos.copy(),
                    world.food.amounts.copy(),
                    world.valve_states(),
                    world.valve_toggles(0),
                    world.energy,
                )
            )
    return sums, checks, world


def _max_relative(a: dict[str, float], b: dict[str, float]) -> tuple[float, str, float]:
    scale = max(max(abs(v) for v in a.values()), max(abs(v) for v in b.values()))
    worst = 0.0
    name = ""
    for key in a:
        diff = abs(a[key] - b[key])
        if diff > worst:
            worst = diff
            name = key
    return worst / scale, name, scale


def _scripted_valve_run(
    delay: float,
    valve_index: int,
    seconds: float,
    seed: int = 17,
    step: float = 0.01,
    effective: int = 0,
):
    """C2/C3/C6 script: deplete at 90 s, touch the valve at 100 and 200 s."""
    world = World(
        Config(
            preset="valve_room",
            seed=seed,
            valve_delay_s=float(delay),
            valve_effective=effective,
        )
    )
    position = np.array(world.config.valve_positions[valve_index] + (0.0,))
    zero = np.zeros(3)
    samples = []
    for _ in range(int(round(seconds / step))):
        world.advance(step)
        now = world.time
        if abs(now - 90.0) <= step / 2.0:
            world.food.amounts[:] = 0.1
        if abs(now - 100.0) <= step / 2.0:
            world.body.set_root_state(position, zero)
        if abs(now - 101.0) <= step / 2.0:
            world.body.set_root_state(zero, zero)
        if abs(now - 200.0) <= step / 2.0:
            world.body.set_root_state(position, zero)
        if abs(now - 201.0) <= step / 2.0:
            world.body.set_root_state(zero, zero)
        samples.append((now, world.regrowth_allowed, float(world.food.totals()[0])))
    return world, samples


def _intervals(samples, flag: bool) -> list[tuple[float, float]]:
    out = []
    start = None
    for when, allowed, _total in samples:
        if allowed == flag and start is None:
            start = when
        elif allowed != flag and start is not None:
            out.append((start, when))
            start = None
    if start is not None:
        out.append((start, samples[-1][0]))
    return out


def _first_transition(samples, after: float, flag: bool) -> float | None:
    for when, allowed, _total in samples:
        if when >= after and allowed == flag:
            return when
    return None


# --------------------------------------------------------------------------
# C1: defaults unchanged
# --------------------------------------------------------------------------
def test_c1_defaults_unchanged():
    script = DATA / "valve_room_lean60.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORKTREE)
    outputs = {}
    for label, out in (
        ("stored", C1_DIR / "lean60_master.npz"),
        ("fresh", C1_DIR / "_fresh_default.npz"),
    ):
        if label == "fresh":
            subprocess.run(
                [sys.executable, str(script), str(out), "3", "60"],
                cwd=str(WORKTREE),
                env=env,
                check=True,
            )
        outputs[label] = dict(np.load(out))
    checks = ("qpos", "qvel", "energy", "amounts", "centres")
    equal = {
        key: bool(np.array_equal(outputs["fresh"][key], outputs["stored"][key]))
        for key in checks
    }
    print(
        "C1 60 s default lean against the stored reference: "
        + ", ".join(f"{key}={equal[key]}" for key in checks)
        + f" (checkpoints={outputs['fresh']['qpos'].shape[0]})"
    )
    assert all(equal.values()), equal

    # A valve-enabled run must not perturb the physics either.
    out = C1_DIR / "_fresh_valves.npz"
    subprocess.run(
        [sys.executable, str(script), str(out), "3", "60", "valves"],
        cwd=str(WORKTREE),
        env=env,
        check=True,
    )
    valves = dict(np.load(out))
    stored_valves = dict(np.load(C1_DIR / "lean60_valves.npz"))
    same = {key: bool(np.array_equal(valves[key], stored_valves[key])) for key in checks}
    print("C1 valves-enabled physics: " + ", ".join(f"{key}={same[key]}" for key in checks))
    assert all(same.values()), same

    # Must-fail twin: the same comparison on a different seed must differ.
    out = C1_DIR / "_fresh_seed4.npz"
    subprocess.run(
        [sys.executable, str(script), str(out), "4", "60"],
        cwd=str(WORKTREE),
        env=env,
        check=True,
    )
    other = dict(np.load(out))
    differs = {
        key: bool(not np.array_equal(other[key], outputs["stored"][key])) for key in checks
    }
    print("C1 must-fail twin (seed 4): " + ", ".join(f"{key}={differs[key]}" for key in checks))
    assert all(differs.values()), differs
    for leftover in ("_fresh_default.npz", "_fresh_valves.npz", "_fresh_seed4.npz"):
        (C1_DIR / leftover).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# C2: effective valve, scripted
# --------------------------------------------------------------------------
def test_c2_effective_valve_scripted():
    delay = 30.0
    seconds = 300.0
    world, samples = _scripted_valve_run(delay, 0, seconds)
    closures = _intervals(samples, False)
    stop = _first_transition(samples, 100.0, False)
    assert stop is not None
    restart = _first_transition(samples, stop + 1e-6, True)
    print(f"C2 effective toggles={world.valve_toggles(0)!r} states={world.valve_states()!r}")
    print(f"C2 regrowth-closed intervals={closures!r}")
    print(f"C2 measured stop={stop!r} expected={100.0 + delay!r}")
    print(f"C2 measured restart={restart!r} expected={200.0 + delay!r}")
    assert stop is not None and abs(stop - (100.0 + delay)) <= 0.0100001
    assert restart is not None and abs(restart - (200.0 + delay)) <= 0.0100001
    assert len(closures) == 1, closures
    frozen_start = closures[0][0] + 0.05
    frozen_end = closures[0][1] - 0.05
    totals = {round(when, 6): total for when, _flag, total in samples}
    frozen = [total for when, _flag, total in samples if frozen_start <= when <= frozen_end]
    print(
        f"C2 frozen patch total min={min(frozen)!r} max={max(frozen)!r} "
        f"steps={len(frozen)}"
    )
    assert min(frozen) == max(frozen)

    # Must-fail twin: the same script on the decoy valve never closes regrowth.
    decoy_world, decoy_samples = _scripted_valve_run(delay, 1, seconds)
    decoy_open = _intervals(decoy_samples, False)
    print(
        f"C2 decoy toggles={decoy_world.valve_toggles(1)!r} "
        f"states={decoy_world.valve_states()!r} closed_intervals={decoy_open!r}"
    )
    assert decoy_open == [], decoy_open


# --------------------------------------------------------------------------
# C3: delay bracket
# --------------------------------------------------------------------------
def test_c3_delay_bracket():
    results = {}
    for delay in (300.0, 3000.0):
        seconds = 100.0 + delay + 300.0
        world, samples = _scripted_valve_run(delay, 0, seconds)
        stop = _first_transition(samples, 100.0, False)
        assert stop is not None
        restart = _first_transition(samples, stop + 1e-6, True)
        results[delay] = (stop, restart, world.valve_toggles(0))
        print(
            f"C3 delay={delay!r}: stop={stop!r} (expected {100.0 + delay!r}), "
            f"restart={restart!r} (expected {200.0 + delay!r}), "
            f"toggles={world.valve_toggles(0)!r}"
        )
        assert stop is not None and abs(stop - (100.0 + delay)) <= 0.0100001
        assert restart is not None and abs(restart - (200.0 + delay)) <= 0.0100001

    # Must-fail twin: the 300 s script on the decoy never closes regrowth.
    _world, decoy_samples = _scripted_valve_run(300.0, 1, 800.0)
    decoy_closed = _intervals(decoy_samples, False)
    print(f"C3 decoy delay=300 closed_intervals={decoy_closed!r}")
    assert decoy_closed == []


# --------------------------------------------------------------------------
# C4: holding senses are batch-free
# --------------------------------------------------------------------------
def test_c4_holding_batch_free():
    seconds = 20.0
    seed = 29
    schedules = {
        "0.05": _regular_looks(0.05, seconds),
        "0.5": _regular_looks(0.5, seconds),
        "2.0": _regular_looks(2.0, seconds),
        "irregular": _irregular_looks(seconds, seed=7),
    }
    results = {}
    for label, looks in schedules.items():
        sums, _checks, world = _run_schedule(looks, seconds, seed, "holding")
        integral = world.holding_integral()
        results[label] = (sums, looks, integral)
        rel, name, scale = _max_relative(sums, integral)
        print(
            f"C4 holding look={label}: looks={len(looks)} lines={len(sums)} "
            f"max|sum-integral|/max={rel:.3e} (worst {name}, scale={scale!r})"
        )
        assert rel <= 1e-9, (label, rel, name, scale)
    reference = results["0.05"][0]
    for label in ("0.5", "2.0", "irregular"):
        rel, name, scale = _max_relative(results[label][0], reference)
        print(
            f"C4 schedule {label} vs 0.05: max difference/max={rel:.3e} "
            f"(worst {name}, scale={scale!r})"
        )
        assert rel <= 1e-9, (label, rel, name, scale)

    # Must-fail twin: instant-mode value x gap sums depend on the schedule.
    instant = {}
    for label, looks in schedules.items():
        sums, _checks, _world = _run_schedule(looks, seconds, seed, "instant")
        instant[label] = sums
    differences = {}
    for label in ("0.5", "2.0", "irregular"):
        rel, name, scale = _max_relative(instant[label], instant["0.05"])
        differences[label] = rel
        print(
            f"C4 must-fail instant {label} vs 0.05: diff/max={rel:.3e} "
            f"(worst {name}, scale={scale!r})"
        )
    print(f"C4 must-fail instant differences={differences!r}")
    assert max(differences.values()) > 1e-3, differences


# --------------------------------------------------------------------------
# C5: the world is the same under every look schedule
# --------------------------------------------------------------------------
def test_c5_world_same_under_schedules():
    seconds = 20.0
    seed = 29
    schedules = {
        "0.05": _regular_looks(0.05, seconds),
        "0.5": _regular_looks(0.5, seconds),
        "2.0": _regular_looks(2.0, seconds),
        "irregular": _irregular_looks(seconds, seed=7),
    }
    runs = {
        label: _run_schedule(looks, seconds, seed, "holding")[1]
        for label, looks in schedules.items()
    }
    base = runs["0.05"]
    print(f"C5 checkpoints={len(base)}")
    for label in ("0.5", "2.0", "irregular"):
        other = runs[label]
        assert len(base) == len(other)
        worst = 0.0
        for first, second in zip(base, other):
            assert first[0] == second[0] or abs(first[0] - second[0]) < 1e-12
            worst = max(
                worst,
                float(np.max(np.abs(first[1] - second[1]))),
                float(np.max(np.abs(first[2] - second[2]))),
            )
            assert first[3] == second[3]
            assert first[4] == second[4]
            assert first[5] == second[5]
        print(f"C5 {label} vs 0.05: max |position/food difference|={worst!r}")
        assert worst == 0.0

    # Must-fail twin: different held targets give a different trajectory.
    other_checks = _run_schedule(
        schedules["0.05"], seconds, seed, "holding", scale=1.2
    )[1]
    differs = any(
        not np.array_equal(first[1], second[1])
        for first, second in zip(base, other_checks)
    )
    print(f"C5 must-fail twin (targets x 1.2): positions differ={differs}")
    assert differs


# --------------------------------------------------------------------------
# C6: snapshot / restore
# --------------------------------------------------------------------------
def test_c6_snapshot_restore():
    seconds = 300.0
    delay = 100.0
    seed = 23
    step = 0.01


    def build():
        return World(
            Config(
                preset="valve_room",
                seed=seed,
                valve_delay_s=delay,
                sense_mode="holding",
            )
        )

    world = build()
    names = [item.name for item in world.items() if item.kind == "actuator"]
    world.write(_targets(names, 0.0))
    position = np.array(world.config.valve_positions[0] + (0.0,))
    zero = np.zeros(3)
    readings_a = {}
    snapshot = None
    for _ in range(int(round(seconds / step))):
        world.advance(step)
        now = world.time
        if abs(now - 90.0) <= step / 2.0:
            world.food.amounts[:] = 0.1
        if abs(now - 100.0) <= step / 2.0:
            world.body.set_root_state(position, zero)
        if abs(now - 101.0) <= step / 2.0:
            world.body.set_root_state(zero, zero)
        if abs(now - round(now)) <= step / 2.0:
            reading = world.read()
            when = round(now, 9)
            if when == 150.0:
                snapshot = world.snapshot()
            readings_a[when] = reading
    print(
        f"C6 run A t={world.time!r} states={world.valve_states()!r} "
        f"toggles={world.valve_toggles(0)!r} snapshot={'yes' if snapshot else 'no'}"
    )
    assert snapshot is not None
    valve_part = snapshot["valves"]
    print(
        f"C6 snapshot valves={valve_part['open']!r} toggles={valve_part['toggle_times']!r} "
        f"holding_acc_lines={len(snapshot['holding']['acc'])} "
        f"holding_prev_lines={len(snapshot['holding']['prev'])}"
    )
    assert valve_part["open"] == [False, True]
    assert len(valve_part["toggle_times"][0]) == 1

    restored = build()
    restored.restore(snapshot)
    for _ in range(int(round((seconds - 150.0) / step))):
        restored.advance(step)
        now = restored.time
        if abs(now - round(now)) <= step / 2.0:
            when = round(now, 9)
            reading_b = restored.read()
            reading_a = readings_a[when]
            assert set(reading_a) == set(reading_b)
            for name in reading_a:
                assert reading_a[name] == reading_b[name], (when, name)
    same_qpos = np.array_equal(world.body.data.qpos, restored.body.data.qpos)
    same_qvel = np.array_equal(world.body.data.qvel, restored.body.data.qvel)
    same_food = np.array_equal(world.food.amounts, restored.food.amounts)
    same_energy = world.energy == restored.energy
    same_valves = world.valve_states() == restored.valve_states()
    same_toggles = world.valve_toggles(0) == restored.valve_toggles(0)
    same_acc = world.holding_integral() == restored.holding_integral()
    print(
        f"C6 restored to 300 s: qpos={same_qpos} qvel={same_qvel} food={same_food} "
        f"energy={same_energy} valves={same_valves} toggles={same_toggles} "
        f"holding_acc={same_acc}"
    )
    assert all(
        (same_qpos, same_qvel, same_food, same_energy, same_valves, same_toggles, same_acc)
    )

    # Must-fail twin: drop the valve and holding sections and the restore diverges.
    broken = copy.deepcopy(snapshot)
    broken.pop("valves")
    broken.pop("holding")
    broken_world = build()
    broken_world.restore(broken)
    differing = 0
    for _ in range(int(round((seconds - 150.0) / step))):
        broken_world.advance(step)
        now = broken_world.time
        if abs(now - round(now)) <= step / 2.0:
            when = round(now, 9)
            reading_b = broken_world.read()
            if any(reading_b[name] != readings_a[when][name] for name in reading_b):
                differing += 1
    print(f"C6 must-fail twin (no valve/holding sections): differing looks={differing}")
    assert differing > 0


# --------------------------------------------------------------------------
# C7: noise variance scales with world time
# --------------------------------------------------------------------------
def test_c7_noise_variance():
    seconds = 600.0
    seeds = range(20)
    noise = 0.1
    line = "energy.need"

    def run(seed: int, look: float) -> float:
        world = World(
            Config(
                preset="valve_room",
                seed=seed,
                timestep=0.01,
                n_arms=1,
                joints_per_arm=1,
                sense_mode="holding",
                holding_noise=noise,
            )
        )
        names = [item.name for item in world.items() if item.kind == "actuator"]
        world.write(_targets(names, 0.0))
        when = look
        total = 0.0
        looks = 0
        while world.time < seconds - 1e-9:
            gap = when - world.time
            if gap > 0.0:
                world.advance(gap)
            if world.time >= when - 1e-9:
                world.read()
                total += world.last_holding_noise[line]
                looks += 1
                when += look
        return total

    sums = {0.05: [], 2.0: []}
    for look in sums:
        for seed in seeds:
            sums[look].append(run(seed, look))
    variances = {look: float(np.var(values, ddof=1)) for look, values in sums.items()}
    expected = noise * noise * seconds
    ratio = variances[0.05] / variances[2.0]
    low, high = 19 * expected / 32.852, 19 * expected / 8.907
    print(
        f"C7 looks=0.05 variance={variances[0.05]!r} looks=2.0 "
        f"variance={variances[2.0]!r} ratio={ratio!r}"
    )
    print(f"C7 expected variance={expected!r} 95% chi-square interval=[{low!r}, {high!r}]")
    print(f"C7 per-seed sums 0.05={sums[0.05]!r}")
    print(f"C7 per-seed sums 2.0={sums[2.0]!r}")
    assert low <= variances[0.05] <= high
    assert low <= variances[2.0] <= high
    assert 0.396 <= ratio <= 2.526

    # Must-fail twin: noise per look with sd=noise ignores the gaps, so the
    # fine schedule's summed variance grows with the number of looks.
    rng = np.random.default_rng(12345)
    bad = {}
    for look in sums:
        count = int(round(seconds / look))
        values = [float(rng.normal(0.0, noise, size=count).sum()) for _ in seeds]
        bad[look] = float(np.var(values, ddof=1))
    print(
        "C7 must-fail twin (per-look sd=noise, synthetic): "
        f"var(0.05)={bad[0.05]!r} var(2.0)={bad[2.0]!r} "
        f"ratio={bad[0.05] / bad[2.0]!r} "
        f"(the wrong rule draws {int(seconds / 0.05)} vs {int(seconds / 2.0)} shots)"
    )
    assert bad[0.05] / bad[2.0] > 10.0


# --------------------------------------------------------------------------
# C8: cost
# --------------------------------------------------------------------------
def test_c8_cost():
    def rate(seconds: float, holding: bool) -> float:
        world = World(
            Config(
                preset="valve_room",
                seed=5,
                sense_mode="holding" if holding else "instant",
            )
        )
        names = [item.name for item in world.items() if item.kind == "actuator"]
        world.write(_targets(names, 0.0))
        when = 0.05
        old = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(old)})
        try:
            started = time.perf_counter()
            while world.time < seconds - 1e-9:
                gap = when - world.time
                if gap > 0.0:
                    world.advance(gap)
                if world.time >= when - 1e-9:
                    world.read()
                    when += 0.05
            wall = time.perf_counter() - started
        finally:
            os.sched_setaffinity(0, old)
        return seconds / wall

    holding = rate(20.0, True)
    instant = rate(20.0, False)
    print(
        f"C8 valve_room holding 0.05 s looks, one core: "
        f"world seconds per wall second={holding!r} "
        f"(instant at the same looks={instant!r})"
    )
    # Must-fail twin: the instant run is measurably cheaper than holding.
    assert holding > 0.0
    assert holding < instant


# --------------------------------------------------------------------------
# supporting check for build item 6: no clock line reaches the brain
# --------------------------------------------------------------------------
def test_no_clock_line_reaches_the_brain():
    instant = World(Config(preset="valve_room", seed=11))
    holding = World(Config(preset="valve_room", seed=11, sense_mode="holding"))
    instant.advance(3.0)
    holding.advance(3.0)
    reading_i = instant.read()
    reading_h = holding.read()
    print(f"lines instant={sorted(reading_i)!r}")
    print(f"lines holding={sorted(reading_h)!r}")
    assert sorted(reading_i) == sorted(reading_h)
    targets = sorted(item.name for item in instant.items() if item.kind == "actuator")
    forbidden = ("time", "clock", "stamp", "step", "dt", "interval", "elapsed", "age")
    bad = [
        name
        for name in list(reading_i) + targets
        if any(word in name.lower() for word in forbidden)
    ]
    print(f"delivered lines={len(reading_i) + len(targets)} clock-like={bad!r}")
    print(f"target lines={targets!r}")
    assert bad == []
