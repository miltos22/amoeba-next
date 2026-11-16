"""Test 4.1: set-and-hold and partial writes."""

import numpy as np

from amoeba.world import Config, World


def test_set_and_hold_survives_100_advances():
    w = World(Config(seed=1, n_patches=0))
    name = "arm0.j1.target"
    w.write({name: 0.7})
    for _ in range(100):
        w.advance(0.01)
    assert w.targets()[name] == 0.7
    angle = w.read()["arm0.j1.angle"]
    print(f"after 100 advance calls: target={w.targets()[name]!r} angle={angle!r}")
    assert angle > 0.3


def test_partial_write_leaves_the_rest():
    w = World(Config(seed=2, n_patches=0))
    w.write({"arm0.j1.target": 0.7})
    w.advance(0.2)
    w.write({"arm1.j1.target": -0.5})
    w.advance(0.2)
    t = w.targets()
    print(f"partial write: arm0.j1.target={t['arm0.j1.target']!r} arm1.j1.target={t['arm1.j1.target']!r}")
    assert t["arm0.j1.target"] == 0.7
    assert t["arm1.j1.target"] == -0.5
    assert t["arm2.j1.target"] == 0.0


def test_targets_are_clipped_to_item_range():
    w = World(Config(seed=3, n_patches=0))
    item = next(i for i in w.items() if i.name == "arm0.j1.target")
    w.write({"arm0.j1.target": 99.0})
    high = w.targets()["arm0.j1.target"]
    w.write({"arm0.j1.target": -99.0})
    low = w.targets()["arm0.j1.target"]
    print(f"clip: item range [{item.low!r}, {item.high!r}] high={high!r} low={low!r}")
    assert high == item.high and low == item.low


def test_item_names_unique_and_read_matches_item_kinds():
    w = World(Config(seed=4, n_patches=0))
    items = w.items()
    names = [i.name for i in items]
    assert len(names) == len(set(names))
    read = w.read()
    expected = {i.name for i in items if i.kind in ("sense", "feeling")}
    print(f"items={len(items)} read_keys={len(read)} expected_read_keys={len(expected)}")
    assert set(read) == expected
    assert all(np.isfinite(list(read.values())))
