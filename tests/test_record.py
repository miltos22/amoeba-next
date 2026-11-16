"""Test 4.7: the life record is written as the life goes and reads back exactly."""

import json

import numpy as np

from amoeba.record import read_header, read_record
from amoeba.world import Config, World


def test_record_round_trip_and_header(tmp_path):
    cfg = Config(seed=17, n_patches=6)
    path = tmp_path / "life.jsonl"
    w = World(cfg)
    rec = w.enable_recording(path)
    rng = np.random.default_rng(3)
    names = [i.name for i in w.items() if i.kind == "actuator"]
    captured = []
    for _ in range(30):
        w.write({n: float(rng.uniform(-1.0, 1.0)) for n in names})
        w.advance(2.0)
        captured.append(
            (
                w.time,
                w.read(),
                w.targets(),
                w.body.centre(),
                w.body.heading(),
                w.energy,
            )
        )
    lines_before_close = sum(1 for _ in open(path))
    print(f"lines in the still-open record: {lines_before_close} (header + 30 rows expected)")
    assert lines_before_close == 31
    rec.close()

    data = read_record(path)
    header = read_header(path)
    print(f"header keys: {sorted(header)}")
    print(f"seed={header['seed']!r} git_commit={header['git_commit']!r} versions={header['versions']}")
    assert header["seed"] == 17
    assert header["config"]["n_patches"] == 6
    assert header["config"]["preset"] == "lean"
    assert "basal_rate" in header["config"]["preset_constants"]
    assert len(header["git_commit"]) > 0
    assert header["versions"]["mujoco"] and header["versions"]["numpy"]

    assert len(data["time"]) == 30
    for i, (t, reading, targets, (x, y), heading, energy) in enumerate(captured):
        assert data["time"][i] == t
        assert data["pos.x"][i] == x
        assert data["pos.y"][i] == y
        assert data["heading"][i] == heading
        assert data["energy"][i] == energy
        for name, value in reading.items():
            assert data[name][i] == value, f"reading {name} differs at row {i}"
        for name, value in targets.items():
            assert data[name][i] == value, f"target {name} differs at row {i}"
    print(f"round trip: 30 rows, {len(captured[0][1])} item readings and {len(captured[0][2])} targets equal")
    assert np.isfinite(data["time"]).all()


def test_record_time_range_and_food_rows(tmp_path):
    path = tmp_path / "life2.jsonl"
    w = World(Config(seed=18, n_patches=3, food_record_interval_s=1.0))
    rec = w.enable_recording(path)
    for _ in range(10):
        w.advance(1.0)
    rec.close()
    data = read_record(path, t0=3.0, t1=6.0)
    print(f"time window 3..6 s -> {data['time'].tolist()}")
    print(f"food rows present: {int(np.isfinite(data['food.amounts'][:, 0]).sum())} of {len(data['time'])}")
    assert data["time"][0] == 3.0 and data["time"][-1] == 6.0
    assert np.isfinite(data["food.amounts"]).any()
    assert data["food.amounts"].shape[1] == 3
    assert data["food.centres"].shape[2] == 2
