"""Test 4.5: a random life, both presets, three seeds, dense food."""

import math

import numpy as np

from amoeba.world import Config, World


def _life(preset: str, seed: int):
    cfg = Config(seed=seed, preset=preset, n_patches=12, patch_radius=0.25, dish_radius=1.5)
    coverage = cfg.n_patches * cfg.patch_radius**2 / cfg.dish_radius**2
    w = World(cfg)
    rng = np.random.default_rng(1000 + seed)
    names = [i.name for i in w.items() if i.kind == "actuator"]
    energies: list[float] = []
    max_distance = 0.0
    finite = True
    for _ in range(1200):
        w.write({n: float(rng.uniform(-1.5, 1.5)) for n in names})
        w.advance(0.5)
        energies.append(w.energy)
        x, y = w.body.centre()
        max_distance = max(max_distance, math.hypot(x, y))
        finite = finite and bool(np.isfinite(list(w.read().values())).all())
    energies_arr = np.array(energies)
    print(
        f"preset={preset} seed={seed}: food_eaten={w.food_eaten!r} "
        f"mean_energy={energies_arr.mean()!r} min_energy={energies_arr.min()!r} "
        f"max_distance={max_distance!r} finite={finite} coverage={coverage!r}"
    )
    return w.food_eaten, finite, max_distance


def test_random_lives():
    ate = {"lean": False, "rich": False}
    for preset in ("lean", "rich"):
        for seed in (0, 1, 2):
            eaten, finite, max_distance = _life(preset, seed)
            assert finite, f"non-finite reading in preset={preset} seed={seed}"
            assert max_distance <= 1.5, f"body outside the dish in preset={preset} seed={seed}"
            if eaten > 0.0:
                ate[preset] = True
    print(f"at least one run ate something: {ate}")
    assert ate["lean"] and ate["rich"]
