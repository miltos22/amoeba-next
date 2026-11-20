"""Test 4.3: physics, known answer both ways.

Under tests/ only (no controller lives in amoeba/): a travelling wave
along the arms and a single-joint back-and-forth flap, same amplitude
and frequency, 20 world seconds each.  A wave along a chain in a fluid
is non-reciprocal and must move the body; a reciprocal flap must not
move it much. The design asks for a factor of at least 3.
"""

import math

import numpy as np

from amoeba.world import Config, World

AMP = 1.0
FREQ = 0.75
SECONDS = 20.0
STEP = 0.02


def _run(kind: str):
    w = World(Config(seed=6, n_patches=0))
    n_arms = w.config.n_arms
    jpa = w.config.joints_per_arm
    names = [it.name for it in w.items() if it.kind == "actuator"]
    targets = {n: 0.0 for n in names}
    start = np.array(w.body.centre())
    t = 0.0
    while t < SECONDS - 1e-12:
        for k in range(n_arms):
            for j in range(jpa):
                if kind == "wave":
                    phase = 2.0 * math.pi * FREQ * t - j * math.pi / 2.0 - k * math.pi / 2.0
                    value = AMP * math.sin(phase)
                elif kind == "flap" and k == 0 and j == jpa - 1:
                    value = AMP * math.sin(2.0 * math.pi * FREQ * t)
                else:
                    value = 0.0
                targets[f"arm{k}.j{j + 1}.target"] = value
        w.write(targets)
        w.advance(STEP)
        t += STEP
    end = np.array(w.body.centre())
    displacement = float(np.linalg.norm(end - start))
    print(
        f"{kind}: start=({start[0]!r}, {start[1]!r}) end=({end[0]!r}, {end[1]!r}) "
        f"displacement={displacement!r}"
    )
    return displacement


def test_wave_swims_and_flap_does_not():
    wave = _run("wave")
    flap = _run("flap")
    ratio = wave / flap if flap > 0.0 else math.inf
    print(f"wave={wave!r} flap={flap!r} ratio={ratio!r}")
    assert np.isfinite(wave) and np.isfinite(flap)
    assert wave >= 3.0 * flap
