"""Pilot lives of the brain in the stage-0 world (design DO 8).

    seeds 0, preset lean, 12 patches
    600 s with n = 48
    600 s with n = 0
     60 s on a changed body, world.change_body(4, 3) before the brain is built,
        with n = 16
    plus the wall time of a 600 s life of the world alone, same calls, no learner

Each life is written to experiments/brain_pilot/<name>.jsonl by brain.life,
which refuses to overwrite.  The summary (decisions per wall second,
mean_abs_need, path_m, the header's constants in force) goes to
experiments/brain_pilot/pilot_summary.json and to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from amoeba.world import Config, World
from brain.learner import BrainConfig
from brain.life import SAMPLE_S, run_life

HERE = os.path.dirname(os.path.abspath(__file__))

SPECS = [
    {"name": "life_n48_seed0", "n_units": 48, "life_s": 600.0, "body": None},
    {"name": "life_n0_seed0", "n_units": 0, "life_s": 600.0, "body": None},
    {"name": "life_n16_body4x3_seed0", "n_units": 16, "life_s": 60.0, "body": (4, 3)},
]


def _world(seed=0):
    return World(
        Config(seed=seed, preset="lean", n_patches=12, patch_radius=0.25, dish_radius=1.5)
    )


def world_only(seconds=600.0, seed=0, control_s=0.1):
    """The same write / advance / read / record calls, no learner."""
    world = _world(seed)
    items = world.items()
    actuators = [item for item in items if item.kind == "actuator"]
    rng = np.random.default_rng(7)
    set_point = world.config.preset_constants.energy_set_point
    records = []
    next_sample_t = SAMPLE_S
    wall_start = time.perf_counter()
    while world.time < seconds - 1e-9:
        world.write(
            {item.name: float(rng.uniform(item.low, item.high)) for item in actuators}
        )
        world.advance(control_s)
        reading = world.read()
        while next_sample_t <= world.time + 1e-9:
            energy = float(world.energy)
            x, y = world.body.centre()
            records.append(
                {
                    "t": float(world.time),
                    "energy": energy,
                    "abs_need": abs(set_point - energy),
                    "need_pos": max(0.0, set_point - energy),
                    "need_neg": max(0.0, energy - set_point),
                    "below_half": energy < 0.5,
                    "zero": energy == 0.0,
                    "pos": (x, y),
                    "any_touch": any(
                        reading[f"arm{k}.touch"] == 1.0 for k in range(world.config.n_arms)
                    ),
                    "finite": bool(np.isfinite(list(reading.values())).all()),
                    "food_cum": float(world.food_eaten),
                }
            )
            next_sample_t += SAMPLE_S
    return time.perf_counter() - wall_start, len(records)


def main() -> int:
    summary = {"pilot": [], "world_only": None}
    for spec in SPECS:
        world = _world(0)
        if spec["body"] is not None:
            world.change_body(*spec["body"])
        config = BrainConfig(n_units=spec["n_units"], seed=0)
        out = os.path.join(HERE, spec["name"] + ".jsonl")
        result = run_life(
            world,
            config,
            spec["life_s"],
            spec["life_s"],
            0.1,
            out_path=out,
        )
        life = result["life"]
        row = {
            "name": spec["name"],
            "n_units": spec["n_units"],
            "life_s": spec["life_s"],
            "body": spec["body"],
            "decisions_per_second": life["decisions_per_second"],
            "decisions": life["decisions"],
            "wall_seconds": life["wall_seconds"],
            "mean_abs_need": life["mean_abs_need"],
            "path_m": life["path_m"],
            "mean_energy": life["mean_energy"],
            "food_eaten": life["food_eaten"],
            "frac_dead_units": life["frac_dead_units"],
            "constants_first_decision": result["header"]["constants_first_decision"],
            "n_observations": len(result["header"]["observation_items"]),
            "n_actuators": len(result["header"]["actuator_items"]),
        }
        summary["pilot"].append(row)
        print(json.dumps(row, sort_keys=True))

    wall, samples = world_only(600.0)
    summary["world_only"] = {
        "life_s": 600.0,
        "wall_seconds": wall,
        "samples": samples,
        "control_s": 0.1,
    }
    print(json.dumps({"world_only": summary["world_only"]}, sort_keys=True))
    with open(os.path.join(HERE, "pilot_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
