"""Test 4.8: same seed and same calls give identical records."""

import numpy as np

from amoeba.record import read_record
from amoeba.world import Config, World


def _life(seed: int, path):
    w = World(Config(seed=seed, n_patches=4))
    rec = w.enable_recording(path)
    rng = np.random.default_rng(77)
    names = [i.name for i in w.items() if i.kind == "actuator"]
    for _ in range(40):
        w.write({n: float(rng.uniform(-1.5, 1.5)) for n in names})
        w.advance(0.5)
    rec.close()
    return read_record(path)


def test_two_lives_with_the_same_seed_are_identical(tmp_path):
    a = _life(7, tmp_path / "a.jsonl")
    b = _life(7, tmp_path / "b.jsonl")
    shared = sorted(set(a) & set(b))
    print(f"compared {len(shared)} arrays over {len(a['time'])} rows")
    for key in shared:
        assert np.array_equal(a[key], b[key], equal_nan=True), f"array {key} differs"
    assert a["time"].tolist() == b["time"].tolist()
    print("all shared arrays identical")
