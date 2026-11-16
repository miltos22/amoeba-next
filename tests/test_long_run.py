"""Test 4.9: one long run, not part of the default suite.

Run with AMOEBA_LONG=1.  One million physics steps of random targets,
preset lean; prints steps per second, any non-finite value, min and max
energy.
"""

import os
import time

import numpy as np
import pytest

from amoeba.world import Config, World

pytestmark = pytest.mark.slow

STEPS = 1_000_000
CHUNK = 250


def test_one_million_steps():
    if os.environ.get("AMOEBA_LONG") != "1":
        pytest.skip("set AMOEBA_LONG=1 to run the 1,000,000-step life")
    w = World(Config(seed=21, preset="lean", n_patches=12))
    rng = np.random.default_rng(2)
    names = [i.name for i in w.items() if i.kind == "actuator"]
    chunks = STEPS // CHUNK
    non_finite = 0
    energy_min = 1.0
    energy_max = 0.0
    started = time.perf_counter()
    for i in range(chunks):
        w.write({n: float(rng.uniform(-1.5, 1.5)) for n in names})
        w.advance(CHUNK * w.config.timestep)
        energy_min = min(energy_min, w.energy)
        energy_max = max(energy_max, w.energy)
        values = list(w.read().values())
        q = w.body.data.qpos
        v = w.body.data.qvel
        if not (np.isfinite(values).all() and np.isfinite(q).all() and np.isfinite(v).all()):
            non_finite += 1
        if i % 1000 == 0:
            print(
                f"  step_chunk={i}/{chunks} t={w.time!r} energy={w.energy!r}",
                flush=True,
            )
    wall = time.perf_counter() - started
    print(
        f"steps={STEPS} world_seconds={w.time!r} wall_seconds={wall!r} "
        f"steps_per_second={STEPS / wall!r} non_finite_chunks={non_finite} "
        f"energy_min={energy_min!r} energy_max={energy_max!r} "
        f"food_eaten={w.food_eaten!r}"
    )
    assert non_finite == 0
