"""Measure the steerable gait: six 30 s swims, one per commanded direction.

For each direction j * 60 degrees (body frame) the body starts at rest at
the dish centre on an empty lean dish (no food, no smell) and swims for
30 s.  Prints one line per direction: net displacement, the direction of
travel in the body frame at the start, mean speed (net and path), and the
yaw the body picked up.  Also runs the reciprocal in-phase flap, which
must not move the body.

Run from the Amobea root:
    taskset -c 0-7 nice -n 19 .venv/bin/python experiments/upper_reference/measure_gait.py
"""

from __future__ import annotations

import math

import numpy as np

from amoeba.world import Config, World

from experiments.upper_reference.gait import (
    in_phase_targets,
    swim_targets,
    _wrap,
)

SECONDS = 30.0
STEP = 0.02


def _run(kind: str, direction_body: float = 0.0, seed: int = 0) -> dict:
    world = World(Config(seed=seed, preset="lean", n_patches=0))
    start = np.array(world.body.centre())
    heading0 = world.body.heading()
    prev = start.copy()
    path = 0.0
    t = 0.0
    while t < SECONDS - 1e-12:
        targets = in_phase_targets(t) if kind == "in_phase" else swim_targets(direction_body, t)
        world.write(targets)
        world.advance(STEP)
        centre = np.array(world.body.centre())
        path += float(np.linalg.norm(centre - prev))
        prev = centre
        t += STEP
    end = np.array(world.body.centre())
    heading1 = world.body.heading()
    delta = end - start
    net = float(np.linalg.norm(delta))
    world_angle = math.atan2(float(delta[1]), float(delta[0]))
    body_angle = _wrap(world_angle - heading0)
    return {
        "kind": kind,
        "direction_body": direction_body,
        "start": (float(start[0]), float(start[1])),
        "end": (float(end[0]), float(end[1])),
        "delta": (float(delta[0]), float(delta[1])),
        "displacement_m": net,
        "path_m": path,
        "direction_body_deg": math.degrees(body_angle),
        "net_speed_cm_s": 100.0 * net / SECONDS,
        "path_speed_cm_s": 100.0 * path / SECONDS,
        "yaw_deg": math.degrees(_wrap(heading1 - heading0)),
        "heading0": heading0,
        "heading1": heading1,
    }


def main() -> int:
    print(f"gait measurement: {SECONDS!r} s swims, seed 0, lean, n_patches=0")
    for j in range(6):
        commanded = j * math.pi / 3.0
        r = _run("swim", commanded)
        error = _wrap(math.radians(r["direction_body_deg"]) - commanded)
        print(
            f"direction_deg={j * 60} "
            f"displacement_m={r['displacement_m']!r} "
            f"delta_xy=({r['delta'][0]!r}, {r['delta'][1]!r}) "
            f"direction_body_deg={r['direction_body_deg']!r} "
            f"direction_error_deg={math.degrees(error)!r} "
            f"net_speed_cm_s={r['net_speed_cm_s']!r} "
            f"path_speed_cm_s={r['path_speed_cm_s']!r} "
            f"path_m={r['path_m']!r} "
            f"yaw_deg={r['yaw_deg']!r}"
        )
    flap = _run("in_phase")
    print(
        f"in_phase displacement_m={flap['displacement_m']!r} "
        f"path_speed_cm_s={flap['path_speed_cm_s']!r} yaw_deg={flap['yaw_deg']!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
