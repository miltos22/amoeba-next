"""Known-answer checks for the two blind reference arms (W16).

  (a) BlindSwimmer gives the same targets in two worlds with different
      food layouts at the same seed: it reads nothing.
  (b) A spy on read() counts zero calls for BlindSwimmer; for
      BlindGatedSwimmer the targets depend on the energy reading and on
      nothing else (two layouts, same energy -> same targets; same
      layout, different energy -> different targets).
  (c) BlindSwimmer swims: the body centre covers at least 3 m of path in
      300 s.

Run from the Amobea root:
    PYTHONPATH=. .venv/bin/python -m pytest experiments/upper_reference/test_blind.py -s -q
"""

from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from amoeba.world import Config, World  # noqa: E402
from experiments.upper_reference.gait import stop_targets  # noqa: E402
from experiments.upper_reference.policies import (  # noqa: E402
    BlindGatedSwimmer,
    BlindSwimmer,
)

GAIT_S = 0.02
CONTROL_S = 0.1


class ReadSpy:
    """A counting stand-in for world.read()."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.world.read()


class FakeEnergy:
    """A read() that returns only the energy feeling, settable at will."""

    def __init__(self, energy: float, set_point: float = 0.8) -> None:
        self.energy = float(energy)
        self.set_point = float(set_point)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return {"energy.need": self.set_point - self.energy}


def _world(seed: int, preset: str, n_patches: int) -> World:
    return World(
        Config(
            seed=seed,
            preset=preset,
            n_patches=n_patches,
            patch_radius=0.25,
            dish_radius=1.5,
        )
    )


def _drive_blind(world: World, policy: BlindSwimmer, seconds: float):
    """Run the forager_life cycle: decide every 0.1 s, write every 0.02 s."""
    targets_seen = []
    t = 0.0
    tick = 0
    control_ticks = int(round(CONTROL_S / GAIT_S))
    n_ticks = int(round(seconds / GAIT_S))
    for tick in range(n_ticks):
        if tick % control_ticks == 0:
            policy.update()
        targets = policy.targets(float(t))
        targets_seen.append(targets)
        world.write(targets)
        world.advance(GAIT_S)
        t += GAIT_S
    return targets_seen


def test_blind_swimmer_same_targets_in_different_worlds():
    w_a = _world(seed=1, preset="lean", n_patches=3)
    w_b = _world(seed=2, preset="rich", n_patches=12)
    assert w_a.food.centres_list() != w_b.food.centres_list(), "the two layouts must differ"

    policy_a = BlindSwimmer(hold_s=2.0, seed=11, control_s=CONTROL_S)
    policy_b = BlindSwimmer(hold_s=2.0, seed=11, control_s=CONTROL_S)
    targets_a = _drive_blind(w_a, policy_a, seconds=30.0)
    targets_b = _drive_blind(w_b, policy_b, seconds=30.0)
    assert targets_a == targets_b
    print(f"(a) targets identical for two layouts: n_target_rows={len(targets_a)}")
    print(f"(a) first direction={policy_a.direction!r} hold_left_s={policy_a.hold_left_s!r}")


def test_blind_swimmer_never_reads_and_gated_reads_only_energy():
    world_blind = _world(seed=3, preset="lean", n_patches=5)
    spy_blind = ReadSpy(world_blind)
    blind = BlindSwimmer(hold_s=2.0, seed=5, control_s=CONTROL_S)
    for _ in range(100):
        blind.update()
        blind.targets(0.0)
    assert spy_blind.calls == 0, f"BlindSwimmer called read() {spy_blind.calls} times"

    w_lean = _world(seed=4, preset="lean", n_patches=3)
    w_rich = _world(seed=8, preset="rich", n_patches=9)
    assert w_lean.food.centres_list() != w_rich.food.centres_list()
    w_lean.energy = 0.70
    w_rich.energy = 0.70
    spy_lean = ReadSpy(w_lean)
    spy_rich = ReadSpy(w_rich)
    gated_lean = BlindGatedSwimmer(spy_lean, set_point=0.8, hold_s=2.0, seed=21, control_s=CONTROL_S)
    gated_rich = BlindGatedSwimmer(spy_rich, set_point=0.8, hold_s=2.0, seed=21, control_s=CONTROL_S)
    same_energy_targets_lean = []
    same_energy_targets_rich = []
    for step in range(100):
        gated_lean.update()
        gated_rich.update()
        same_energy_targets_lean.append(gated_lean.targets(step * CONTROL_S))
        same_energy_targets_rich.append(gated_rich.targets(step * CONTROL_S))
    assert same_energy_targets_lean == same_energy_targets_rich
    assert spy_lean.calls == 100 and spy_rich.calls == 100
    assert gated_lean.hungry and gated_rich.hungry

    w_full = _world(seed=4, preset="lean", n_patches=3)
    w_full.energy = 0.95
    spy_full = ReadSpy(w_full)
    gated_full = BlindGatedSwimmer(spy_full, set_point=0.8, hold_s=2.0, seed=21, control_s=CONTROL_S)
    full_targets = []
    for step in range(100):
        gated_full.update()
        full_targets.append(gated_full.targets(step * CONTROL_S))
    assert not gated_full.hungry
    assert all(row == stop_targets() for row in full_targets)
    assert same_energy_targets_lean[0] != full_targets[0]
    print(f"(b) blind read calls={spy_blind.calls} gated read calls={spy_lean.calls}")
    print(f"(b) gated hungry first targets == full first targets: "
          f"{same_energy_targets_lean[0] == full_targets[0]}")


def test_blind_gated_hysteresis_uses_both_levels():
    read = FakeEnergy(0.70)
    gated = BlindGatedSwimmer(read, set_point=0.8, hold_s=2.0, seed=1, control_s=CONTROL_S)
    gated.update()
    assert gated.hungry
    read.energy = 0.80
    gated.update()
    assert gated.hungry, "between the levels the hungry state must be kept"
    read.energy = 0.83
    gated.update()
    assert not gated.hungry
    read.energy = 0.80
    gated.update()
    assert not gated.hungry, "between the levels the full state must be kept"
    read.energy = 0.77
    gated.update()
    assert gated.hungry
    print(f"(b) hysteresis holds at eat_below=0.78 leave_above=0.82, read calls={read.calls}")


def test_blind_swimmer_swims_three_metres_in_300_s():
    world = _world(seed=0, preset="lean", n_patches=12)
    policy = BlindSwimmer(hold_s=10.0, seed=0, control_s=CONTROL_S)
    prev = world.body.centre()
    path = 0.0
    n_ticks = int(round(300.0 / GAIT_S))
    control_ticks = int(round(CONTROL_S / GAIT_S))
    for tick in range(n_ticks):
        if tick % control_ticks == 0:
            policy.update()
        world.write(policy.targets(tick * GAIT_S))
        world.advance(GAIT_S)
        here = world.body.centre()
        path += math.hypot(here[0] - prev[0], here[1] - prev[1])
        prev = here
    print(f"(c) path_m={path!r} in 300 s (threshold 3.0)")
    assert path >= 3.0
