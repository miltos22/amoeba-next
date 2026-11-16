"""Test 4.10: steps per second for two bodies, record on and off."""

import time

import numpy as np

from amoeba.world import Config, World

STEPS = 20000
CHUNK = 250  # 0.5 s


def _measure(n_arms, joints_per_arm, recording, path=None):
    w = World(Config(seed=31, n_arms=n_arms, joints_per_arm=joints_per_arm, n_patches=12))
    if recording:
        w.enable_recording(path)
    rng = np.random.default_rng(5)
    names = [i.name for i in w.items() if i.kind == "actuator"]
    chunks = STEPS // CHUNK
    started = time.perf_counter()
    for _ in range(chunks):
        w.write({n: float(rng.uniform(-1.5, 1.5)) for n in names})
        w.advance(0.5)
    wall = time.perf_counter() - started
    if recording:
        w.disable_recording()
    return STEPS / wall


def test_steps_per_second(tmp_path):
    results = {}
    for body in ((6, 2), (12, 3)):
        for recording in (False, True):
            path = tmp_path / f"speed_{body[0]}x{body[1]}_{recording}.jsonl"
            rate = _measure(body[0], body[1], recording, path)
            results[(body, recording)] = rate
            print(
                f"body={body[0]}x{body[1]} record={'on' if recording else 'off'}: "
                f"steps_per_second={rate!r}"
            )
    for key, rate in results.items():
        assert np.isfinite(rate) and rate > 0.0
