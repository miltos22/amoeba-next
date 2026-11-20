"""Helper: one default lean run, checkpoints every 1 s.

Run from a project root with PYTHONPATH pointing at it; the same file runs
the worktree code and the master code (R) so the two outputs can be
compared bit for bit.  The target schedule is the project's standard one
(tests/data/make_regression_default.py).

    python tests/data/valve_room_lean60.py OUT.npz [seed] [seconds] [valves]
"""

import math
import sys

import numpy as np

from amoeba.world import Config, World

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 60
valves = len(sys.argv) > 4 and sys.argv[4] == "valves"
options = {"valves_enabled": True} if valves else {}
world = World(Config(seed=seed, preset="lean", **options))
actuators = [item for item in world.items() if item.kind == "actuator"]
qpos = []
qvel = []
energy = []
amounts = []
centres = []
readings = []
for _ in range(seconds):
    for _ in range(10):
        t = world.time
        targets = {}
        for j, item in enumerate(actuators):
            value = 2.0 * math.sin(0.7 * t + 0.9 * j)
            value = item.low if value < item.low else (item.high if value > item.high else value)
            targets[item.name] = value
        world.write(targets)
        world.advance(0.1)
    qpos.append(world.body.data.qpos.copy())
    qvel.append(world.body.data.qvel.copy())
    energy.append(world.energy)
    amounts.append(world.food.amounts.copy())
    centres.append(world.food.centres.copy())
    reading = world.read()
    readings.append([reading[name] for name in sorted(reading)])
np.savez(
    out,
    qpos=np.array(qpos),
    qvel=np.array(qvel),
    energy=np.array(energy),
    amounts=np.array(amounts),
    centres=np.array(centres),
    readings=np.array(readings),
)
print(f"wrote {out} t={world.time!r} checkpoints={len(qpos)}")
