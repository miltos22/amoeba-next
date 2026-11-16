"""Test 4.2: same held-target sequence at three timesteps; print the differences.

No tolerance is chosen here: the test only asserts finiteness and prints
the differences, as the brief demands.
"""

import numpy as np

from amoeba.world import Config, World


def _life(timestep):
    w = World(Config(seed=5, timestep=timestep, n_patches=0, dish_radius=1.5))
    rng = np.random.default_rng(11)
    names = [it.name for it in w.items() if it.kind == "actuator"]
    for _ in range(40):
        w.write({n: float(rng.uniform(-1.5, 1.5)) for n in names})
        w.advance(0.5)
    return np.array(w.body.centre()), float(w.energy)


def test_timestep_changes_the_same_life_slowly():
    results = {dt: _life(dt) for dt in (0.001, 0.002, 0.004)}
    for dt, (pos, energy) in results.items():
        print(f"timestep={dt}: centre=({pos[0]!r}, {pos[1]!r}) energy={energy!r}")
        assert np.all(np.isfinite(pos)) and np.isfinite(energy)
    base_pos, base_e = results[0.001]
    for dt, (pos, energy) in results.items():
        if dt == 0.001:
            continue
        dist = float(np.linalg.norm(pos - base_pos))
        de = float(abs(energy - base_e))
        print(f"difference 0.001 vs {dt}: centre_distance={dist!r} energy_difference={de!r}")
        assert np.isfinite(dist) and np.isfinite(de)
