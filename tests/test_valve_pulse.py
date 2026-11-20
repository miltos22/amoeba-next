"""Pulse valve mode, its decoy twin and its snapshot.

Every check prints its measured numbers (run pytest with -s).  The pulse
field is OFF by default; the toggle tests in tests/test_valve_room.py keep
covering the old mode unchanged.
"""

from __future__ import annotations

import numpy as np

from amoeba.world import Config, World


def _build(
    *,
    delay: float = 30.0,
    pulse: float = 50.0,
    mode: str = "pulse",
    seed: int = 17,
    holding: bool = False,
    effect: str = "stop",
) -> World:
    return World(
        Config(
            preset="valve_room",
            seed=seed,
            timestep=0.01,
            valve_mode=mode,
            valve_delay_s=delay,
            valve_pulse_s=pulse,
            valve_pulse_effect=effect,
            sense_mode="holding" if holding else "instant",
        )
    )


def _scripted_run(
    delay: float,
    pulse: float,
    mode: str,
    valve_index: int,
    seconds: float,
    seed: int = 17,
    step: float = 0.01,
    second_touch: bool = True,
    effect: str = "stop",
    deplete: bool = True,
):
    """Optionally deplete at 90 s, touch one valve at 100 s (and at 200 s)."""
    world = _build(delay=delay, pulse=pulse, mode=mode, seed=seed, effect=effect)
    position = np.array(world.config.valve_positions[valve_index] + (0.0,))
    zero = np.zeros(3)
    samples = []
    for _ in range(int(round(seconds / step))):
        world.advance(step)
        now = world.time
        if deplete and abs(now - 90.0) <= step / 2.0:
            world.food.amounts[:] = 0.1
        if abs(now - 100.0) <= step / 2.0:
            world.body.set_root_state(position, zero)
        if abs(now - 101.0) <= step / 2.0:
            world.body.set_root_state(zero, zero)
        if second_touch and abs(now - 200.0) <= step / 2.0:
            world.body.set_root_state(position, zero)
        if second_touch and abs(now - 201.0) <= step / 2.0:
            world.body.set_root_state(zero, zero)
        samples.append(
            (now, bool(world.regrowth_allowed), float(world.food.totals()[0]))
        )
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
# 1.3 passing case: one touch stops regrowth exactly during [130, 180]
# --------------------------------------------------------------------------
def test_pulse_stop_window():
    delay = 30.0
    pulse = 50.0
    world, samples = _scripted_run(
        delay, pulse, "pulse", 0, 300.0, second_touch=False
    )
    touch = world.valve_toggles(0)[0]
    stop = _first_transition(samples, touch + delay - 1.0, False)
    restart = _first_transition(samples, stop + 1e-6, True)
    closures = _intervals(samples, False)
    print(
        f"1.3 pulse touches={world.valve_toggles(0)!r} "
        f"states={world.valve_states()!r} closed_intervals={closures!r}"
    )
    print(
        f"1.3 pulse measured stop={stop!r} expected={touch + delay!r} "
        f"restart={restart!r} expected={touch + delay + pulse!r}"
    )
    assert stop is not None
    assert abs(stop - (touch + delay)) <= 0.0100001
    assert restart is not None
    assert abs(restart - (touch + delay + pulse)) <= 0.0200001
    assert len(closures) == 1, closures

    frozen = [
        total
        for when, _allowed, total in samples
        if stop + 0.05 <= when <= restart - 0.05
    ]
    print(
        f"1.3 pulse patch total in window min={min(frozen)!r} "
        f"max={max(frozen)!r} steps={len(frozen)}"
    )
    assert min(frozen) == max(frozen)

    # Overlapping windows merge: touches at 100 and 120 give [130, 180] and
    # [150, 200], so regrowth stays stopped from 130 to 200 in one stretch.
    merged_world = _build(delay=delay, pulse=pulse, mode="pulse", seed=17)
    position = np.array(merged_world.config.valve_positions[0] + (0.0,))
    zero = np.zeros(3)
    merged = []
    for _ in range(int(round(300.0 / 0.01))):
        merged_world.advance(0.01)
        now = merged_world.time
        if abs(now - 100.0) <= 0.005 or abs(now - 120.0) <= 0.005:
            merged_world.body.set_root_state(position, zero)
        if abs(now - 101.0) <= 0.005 or abs(now - 121.0) <= 0.005:
            merged_world.body.set_root_state(zero, zero)
        merged.append(
            (now, bool(merged_world.regrowth_allowed), float(merged_world.food.totals()[0]))
        )
    merged_closures = _intervals(merged, False)
    print(
        f"1.3 pulse merged touches={merged_world.valve_toggles(0)!r} "
        f"closed_intervals={merged_closures!r}"
    )
    assert len(merged_closures) == 1, merged_closures
    first = merged_closures[0]
    assert abs(first[0] - (merged_world.valve_toggles(0)[0] + delay)) <= 0.0100001
    assert abs(first[1] - (merged_world.valve_toggles(0)[1] + delay + pulse)) <= 0.0100001


# --------------------------------------------------------------------------
# 1.3 must-fail twin: the decoy touch stops nothing
# --------------------------------------------------------------------------
def test_pulse_decoy_stops_nothing():
    delay = 30.0
    pulse = 50.0
    world, samples = _scripted_run(delay, pulse, "pulse", 1, 300.0)
    closures = _intervals(samples, False)
    print(
        f"1.3 decoy touches={world.valve_toggles(1)!r} "
        f"states={world.valve_states()!r} closed_intervals={closures!r}"
    )
    assert closures == [], closures
    # The effective valve was never touched, so it also never stopped.
    assert world.valve_toggles(0) == ()


# --------------------------------------------------------------------------
# 1.3 toggle mode is still the default and still latches (existing suite
# tests/test_valve_room.py is run separately for the full comparison)
# --------------------------------------------------------------------------
def test_toggle_mode_unchanged_smoke():
    assert Config().valve_mode == "toggle"
    assert Config().valve_pulse_s == 300.0
    delay = 30.0
    world, samples = _scripted_run(delay, 50.0, "toggle", 0, 300.0)
    closures = _intervals(samples, False)
    print(
        f"1.3 toggle touches={world.valve_toggles(0)!r} "
        f"closed_intervals={closures!r} default_mode={Config().valve_mode!r}"
    )
    assert len(closures) == 1, closures
    # In toggle mode the state stays closed from the first delayed touch to
    # the second (no pulse end): [130, 230] as in test C2.
    assert abs(closures[0][0] - (world.valve_toggles(0)[0] + delay)) <= 0.0100001
    assert abs(closures[0][1] - (world.valve_toggles(0)[1] + delay)) <= 0.0100001


# --------------------------------------------------------------------------
# 1.3 snapshot at 150 s and restore: bit-identical continuation
# --------------------------------------------------------------------------
def test_pulse_snapshot_restore():
    delay = 30.0
    pulse = 50.0
    seconds = 300.0
    seed = 23
    step = 0.01

    world = _build(delay=delay, pulse=pulse, mode="pulse", seed=seed, holding=True)
    position = np.array(world.config.valve_positions[0] + (0.0,))
    zero = np.zeros(3)

    def advance_one(instance: World, now: float) -> None:
        if abs(now - 90.0) <= step / 2.0:
            instance.food.amounts[:] = 0.1
        if abs(now - 100.0) <= step / 2.0:
            instance.body.set_root_state(position, zero)
        if abs(now - 101.0) <= step / 2.0:
            instance.body.set_root_state(zero, zero)

    snapshot = None
    readings_a = {}
    flags_a = {}
    for _ in range(int(round(seconds / step))):
        world.advance(step)
        now = world.time
        advance_one(world, now)
        if abs(now - round(now * 2.0) / 2.0) <= step / 2.0:
            when = round(now, 9)
            readings_a[when] = world.read()
            flags_a[when] = (bool(world.regrowth_allowed), world.food.totals().copy())
            if when == 150.0:
                snapshot = world.snapshot()
    assert snapshot is not None
    print(
        f"1.3 snapshot t=150 states={world.valve_states()!r} "
        f"toggles={world.valve_toggles(0)!r} valves_section={sorted(snapshot['valves'])!r}"
    )

    restored = _build(delay=delay, pulse=pulse, mode="pulse", seed=seed, holding=True)
    restored.restore(snapshot)
    differing = 0
    for _ in range(int(round((seconds - 150.0) / step))):
        restored.advance(step)
        now = restored.time
        if abs(now - round(now * 2.0) / 2.0) <= step / 2.0:
            when = round(now, 9)
            reading_a = readings_a[when]
            reading_b = restored.read()
            if set(reading_a) != set(reading_b):
                differing += 1
                continue
            if any(reading_a[name] != reading_b[name] for name in reading_a):
                differing += 1
            flag_a, totals_a = flags_a[when]
            if (
                bool(restored.regrowth_allowed) != flag_a
                or not np.array_equal(restored.food.totals(), totals_a)
            ):
                differing += 1
    same_vals = (
        world.valve_states() == restored.valve_states()
        and world.valve_toggles(0) == restored.valve_toggles(0)
        and np.array_equal(world.body.data.qpos, restored.body.data.qpos)
        and np.array_equal(world.body.data.qvel, restored.body.data.qvel)
        and np.array_equal(world.food.amounts, restored.food.amounts)
        and world.energy == restored.energy
    )
    print(
        f"1.3 snapshot/restore to {seconds:.0f} s: differing look/flag checks="
        f"{differing} end_state_equal={same_vals}"
    )
    assert differing == 0
    assert same_vals


# --------------------------------------------------------------------------
# Follow-up 1: valve_pulse_effect="empty" empties every patch at the start
# of each stop window and keeps regrowth stopped through it.
# --------------------------------------------------------------------------
def test_pulse_empty_window():
    delay = 30.0
    pulse = 50.0
    world, samples = _scripted_run(
        delay,
        pulse,
        "pulse",
        0,
        300.0,
        second_touch=False,
        effect="empty",
        deplete=False,
    )
    touch = world.valve_toggles(0)[0]
    start = touch + delay
    end = touch + delay + pulse
    before = [total for when, _a, total in samples if start - 1.0 <= when < start - 0.02]
    during = [total for when, _a, total in samples if start + 0.02 <= when <= end - 0.02]
    after = [total for when, _a, total in samples if end + 1.0 <= when <= end + 20.0]
    flags_during = [allowed for when, allowed, _t in samples if start + 0.02 <= when <= end - 0.02]
    print(
        f"F1 empty touches={world.valve_toggles(0)!r} window=({start!r}, {end!r}) "
        f"before min/max={min(before)!r}/{max(before)!r} "
        f"during min/max={min(during)!r}/{max(during)!r} "
        f"after min/max={min(after)!r}/{max(after)!r} "
        f"regrowth_allowed_during={sorted(set(flags_during))!r}"
    )
    assert max(before) > 0.0
    assert min(during) == 0.0 and max(during) == 0.0
    assert min(after) > 0.0
    assert set(flags_during) == {False}


def test_pulse_empty_decoy():
    delay = 30.0
    pulse = 50.0
    world, samples = _scripted_run(
        delay,
        pulse,
        "pulse",
        1,
        300.0,
        second_touch=False,
        effect="empty",
        deplete=False,
    )
    touch = world.valve_toggles(1)[0]
    start = touch + delay
    end = touch + delay + pulse
    during = [total for when, _a, total in samples if start + 0.02 <= when <= end - 0.02]
    print(
        f"F1 empty decoy touches={world.valve_toggles(1)!r} "
        f"effective_touches={world.valve_toggles(0)!r} "
        f"during min/max={min(during)!r}/{max(during)!r}"
    )
    assert world.valve_toggles(0) == ()
    assert min(during) > 0.0


def test_pulse_empty_snapshot_restore():
    delay = 30.0
    pulse = 50.0
    seconds = 300.0
    seed = 23
    step = 0.01

    world = _build(
        delay=delay, pulse=pulse, mode="pulse", seed=seed, holding=True, effect="empty"
    )
    position = np.array(world.config.valve_positions[0] + (0.0,))
    zero = np.zeros(3)
    snapshot = None
    readings_a = {}
    food_a = {}
    flags_a = {}
    for _ in range(int(round(seconds / step))):
        world.advance(step)
        now = world.time
        if abs(now - 100.0) <= step / 2.0:
            world.body.set_root_state(position, zero)
        if abs(now - 101.0) <= step / 2.0:
            world.body.set_root_state(zero, zero)
        if abs(now - round(now * 2.0) / 2.0) <= step / 2.0:
            when = round(now, 9)
            readings_a[when] = world.read()
            food_a[when] = world.food.amounts.copy()
            flags_a[when] = bool(world.regrowth_allowed)
            if when == 150.0:
                snapshot = world.snapshot()
    assert snapshot is not None
    snap_food = float(np.asarray(snapshot["food"]["amounts"]).sum())
    print(
        f"F1 empty snapshot t=150 pulse_stop_active="
        f"{snapshot['valves']['pulse_stop_active']!r} "
        f"snapshot_food_total={snap_food!r} food_at_end={float(world.food.totals()[0])!r}"
    )
    assert snapshot["valves"]["pulse_stop_active"] is True
    assert snap_food == 0.0

    restored = _build(
        delay=delay, pulse=pulse, mode="pulse", seed=seed, holding=True, effect="empty"
    )
    restored.restore(snapshot)
    differing = 0
    for _ in range(int(round((seconds - 150.0) / step))):
        restored.advance(step)
        now = restored.time
        if abs(now - round(now * 2.0) / 2.0) <= step / 2.0:
            when = round(now, 9)
            reading_a = readings_a[when]
            reading_b = restored.read()
            if any(reading_a[name] != reading_b[name] for name in reading_a):
                differing += 1
            if not np.array_equal(restored.food.amounts, food_a[when]):
                differing += 1
            if bool(restored.regrowth_allowed) != flags_a[when]:
                differing += 1
    same_vals = (
        world.valve_states() == restored.valve_states()
        and world.valve_toggles(0) == restored.valve_toggles(0)
        and np.array_equal(world.body.data.qpos, restored.body.data.qpos)
        and np.array_equal(world.body.data.qvel, restored.body.data.qvel)
        and np.array_equal(world.food.amounts, restored.food.amounts)
        and world.energy == restored.energy
        and world._pulse_stop_active == restored._pulse_stop_active
    )
    print(
        f"F1 empty snapshot/restore to {seconds:.0f} s: differing checks="
        f"{differing} end_state_equal={same_vals}"
    )
    assert differing == 0
    assert same_vals
