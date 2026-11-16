"""Test 4.6: body change mid-life keeps place, time, energy and old names."""

import numpy as np

from amoeba.world import Config, World


def _counts(w):
    acts = [i for i in w.items() if i.kind == "actuator"]
    angles = [i for i in w.items() if i.name.endswith(".angle")]
    return len(acts), len(angles)


def _random_targets(w, rng):
    names = [i.name for i in w.items() if i.kind == "actuator"]
    return {n: float(rng.uniform(-1.5, 1.5)) for n in names}


def test_body_change_6x2_to_8x2_to_4x3():
    w = World(Config(seed=8, n_patches=4))
    rng = np.random.default_rng(21)
    for _ in range(10):
        w.write(_random_targets(w, rng))
        w.advance(0.5)

    old_names = {i.name for i in w.items()}
    old_pos = np.array(w.body.centre())
    old_heading = w.body.heading()
    old_time = w.time
    old_energy = w.energy

    w.change_body(8, 2)
    assert w.time == old_time
    assert w.energy == old_energy
    assert np.array_equal(np.array(w.body.centre()), old_pos)
    assert w.body.heading() == old_heading
    new_names = {i.name for i in w.items()}
    print(
        f"6x2 -> 8x2: time={w.time!r} energy={w.energy!r} "
        f"centre=({w.body.centre()[0]!r}, {w.body.centre()[1]!r}) counts={_counts(w)}"
    )
    assert old_names <= new_names
    assert _counts(w) == (16, 16)

    for _ in range(10):
        w.write(_random_targets(w, rng))
        w.advance(0.5)
        assert all(np.isfinite(list(w.read().values())))

    old_names = new_names
    old_pos = np.array(w.body.centre())
    old_heading = w.body.heading()
    old_time = w.time
    old_energy = w.energy
    w.change_body(4, 3)
    assert w.time == old_time
    assert w.energy == old_energy
    assert np.array_equal(np.array(w.body.centre()), old_pos)
    assert w.body.heading() == old_heading
    new_names = {i.name for i in w.items()}
    survivors = {n for n in old_names if n in new_names}
    print(
        f"8x2 -> 4x3: time={w.time!r} energy={w.energy!r} counts={_counts(w)} "
        f"survivors={len(survivors)} of {len(old_names)}"
    )
    assert _counts(w) == (12, 12)
    for k in range(4):
        for m in range(1, 3):
            base = f"arm{k}.j{m}"
            assert f"{base}.angle" in new_names and f"{base}.target" in new_names
    assert "arm0.j3.angle" in new_names
    assert not any(n.startswith("arm6.") for n in new_names)

    for _ in range(120):
        w.write(_random_targets(w, rng))
        w.advance(0.5)
        vals = w.read()
        assert all(np.isfinite(list(vals.values()))), "non-finite reading after body change"
    print(f"after 60 s more: time={w.time!r} energy={w.energy!r} finite=True")
