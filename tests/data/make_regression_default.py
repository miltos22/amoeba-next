"""Make the bit-for-bit regression data for the default food world (T1).

Run from the worktree root with PYTHONPATH pointing at it:

    .venv/bin/python tests/data/make_regression_default.py

The run reproduced here is fixed by the design: every 0.1 s write
the targets ``2.0 * sin(0.7 * t + 0.9 * j)`` (target index j, clipped to
the item limits) and advance 0.1 s, for 300 s.  Every reading at every
step and the energy are stored as float64.  The script is kept beside the
data so the same function generated the baseline and is used by the test.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from amoeba.world import Config, World

HERE = Path(__file__).resolve().parent


def run_regression(seed: int, preset: str, n_patches: int):
    """Run the fixed target schedule and return (names, readings, energy)."""
    world = World(Config(seed=seed, preset=preset, n_patches=n_patches))
    actuators = [item for item in world.items() if item.kind == "actuator"]
    names: list[str] | None = None
    readings: list[list[float]] = []
    energy: list[float] = []
    for _ in range(3000):
        t = world.time
        targets = {}
        for j, item in enumerate(actuators):
            value = 2.0 * math.sin(0.7 * t + 0.9 * j)
            value = item.low if value < item.low else (item.high if value > item.high else value)
            targets[item.name] = value
        world.write(targets)
        world.advance(0.1)
        reading = world.read()
        if names is None:
            names = list(reading)
        elif list(reading) != names:
            raise AssertionError("reading names changed during the run")
        readings.append([reading[name] for name in names])
        energy.append(world.energy)
    return names, np.array(readings, dtype=np.float64), np.array(energy, dtype=np.float64)


def make(path: Path, seed: int, preset: str, n_patches: int) -> None:
    names, readings, energy = run_regression(seed, preset, n_patches)
    np.savez_compressed(
        path,
        names=np.array(names, dtype=str),
        readings=readings,
        energy=energy,
    )
    print(f"wrote {path}: readings={readings.shape} energy={energy.shape}")


def main() -> int:
    make(HERE / "regression_default.npz", seed=3, preset="lean", n_patches=12)
    make(HERE / "regression_rich.npz", seed=4, preset="rich", n_patches=6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
