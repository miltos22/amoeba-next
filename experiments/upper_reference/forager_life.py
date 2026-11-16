"""Run one forager life and write the noise-floor life format (JSON lines).

The rows are produced by the same functions as
experiments/noise_floor/noise_floor.py (imported, not copied): one header
row with the full configuration in force and the git commit, one window
row per 600 s with the same fields, one life row.  summarise.py therefore
works unchanged.

The gait targets are written every 0.02 s (the interval gait_check.py
used); the policy is asked for a decision every 0.1 s.  The world is
sampled and recorded every 0.5 s, exactly as noise_floor.py does.

Run from the Amobea root with PYTHONPATH=<root>:
    .venv/bin/python experiments/upper_reference/forager_life.py life \
        --policy oracle --preset lean --n-patches 12 --seed 0 \
        --life-s 3000 --window-s 600 --out /tmp/life.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "noise_floor"))

import noise_floor as nf  # noqa: E402

from amoeba.world import Config, World  # noqa: E402
from experiments.upper_reference.policies import (  # noqa: E402
    BlindGatedSwimmer,
    BlindSwimmer,
    OracleForager,
    SmellForager,
)

SAMPLE_S = nf.SAMPLE_S
GAIT_S = 0.02


def _make_policy(args, world):
    if args.policy == "blind":
        return BlindSwimmer(hold_s=args.hold_s, seed=args.seed, control_s=args.control_s)
    if args.policy == "blind_gated":
        return BlindGatedSwimmer(
            world.read,
            world.config.preset_constants.energy_set_point,
            eat_below=args.eat_below,
            leave_above=args.leave_above,
            hold_s=args.hold_s,
            seed=args.seed,
            control_s=args.control_s,
        )
    if args.policy == "oracle":
        return OracleForager(
            world,
            eat_below=args.eat_below,
            leave_above=args.leave_above,
            reverse=args.reverse,
        )
    set_point = world.config.preset_constants.energy_set_point
    return SmellForager(
        world.read,
        set_point,
        eat_below=args.eat_below,
        leave_above=args.leave_above,
        smell_min=args.smell_min,
        seed=args.seed,
        reverse=args.reverse,
        n_arms=world.config.n_arms,
    )


def run_life(args, out_path: str | None = None):
    for name, value in (("life_s", args.life_s), ("window_s", args.window_s)):
        ratio = value / SAMPLE_S
        if abs(ratio - round(ratio)) > 1e-9:
            raise SystemExit(f"--{name} must be a whole multiple of {SAMPLE_S}")
    if abs(args.control_s / GAIT_S - round(args.control_s / GAIT_S)) > 1e-9:
        raise SystemExit(f"--control-s must be a whole multiple of {GAIT_S}")

    args.behaviour = f"forager_{args.policy}"
    if args.policy in ("blind", "blind_gated"):
        args.behaviour = args.policy
    args.redraw_s = 0.0

    n_total = int(round(args.life_s / SAMPLE_S))
    per_window = int(round(args.window_s / SAMPLE_S))
    if per_window < 1 or n_total < 1:
        raise SystemExit("life_s and window_s must each be at least one sample")

    cfg = Config(
        seed=args.seed,
        preset=args.preset,
        n_patches=args.n_patches,
        patch_radius=0.25,
        dish_radius=1.5,
    )
    fh = None
    if out_path is not None:
        fh = open(out_path, "x", encoding="utf-8")
        nf._write_line(fh, nf._header(args, cfg))

    set_point = cfg.preset_constants.energy_set_point
    wall_start = time.perf_counter()
    world = World(cfg)
    policy = _make_policy(args, world)

    sample_ticks = int(round(SAMPLE_S / GAIT_S))
    control_ticks = int(round(args.control_s / GAIT_S))
    n_ticks = int(round(args.life_s / GAIT_S))

    start_pos = world.body.centre()
    records: list[dict] = []
    windows: list[dict] = []
    current: list[dict] = []
    window_start_food = 0.0
    window_start_pos = start_pos
    window_index = 0

    def flush_window():
        nonlocal current, window_start_food, window_start_pos, window_index
        if not current:
            return None
        row = {
            "row": "window",
            "window": window_index,
            "t_start": float(current[0]["t"] - SAMPLE_S),
            "t_end": float(current[-1]["t"]),
        }
        row.update(nf._aggregate(current, window_start_pos, window_start_food))
        windows.append(row)
        if fh is not None:
            nf._write_line(fh, row)
        window_start_food = current[-1]["food_cum"]
        window_start_pos = current[-1]["pos"]
        window_index += 1
        current = []
        return row

    for tick in range(n_ticks):
        if tick % control_ticks == 0:
            policy.update()
        targets = policy.targets(float(world.time))
        if targets is not None:
            world.write(targets)
        world.advance(GAIT_S)
        if (tick + 1) % sample_ticks == 0:
            reading = world.read()
            need = reading["energy.need"]
            if abs(need - (set_point - world.energy)) > 1e-12:
                raise AssertionError(
                    f"energy.need guard failed at t={world.time!r}: "
                    f"need={need!r} set_point-energy={set_point - world.energy!r}"
                )
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
                    "any_touch": any(reading[f"arm{k}.touch"] == 1.0 for k in range(cfg.n_arms)),
                    "finite": bool(np.isfinite(list(reading.values())).all()),
                    "food_cum": float(world.food_eaten),
                }
            )
            current.append(records[-1])
            if len(current) >= per_window:
                flush_window()

    flush_window()
    wall_seconds = time.perf_counter() - wall_start
    steps = int(round(world.time / cfg.timestep))

    life_row = {
        "row": "life",
        "t_start": 0.0,
        "t_end": float(records[-1]["t"]),
    }
    life_row.update(nf._aggregate(records, start_pos, 0.0))
    life_row.update(
        {
            "wall_seconds": float(wall_seconds),
            "steps": steps,
            "steps_per_second": float(steps / wall_seconds),
        }
    )
    if fh is not None:
        nf._write_line(fh, life_row)
        fh.close()
    return {"header": nf._header(args, cfg), "windows": windows, "life": life_row, "records": records}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    life = sub.add_parser("life", help="run one forager life and write JSON lines")
    life.add_argument("--policy", choices=("oracle", "smell", "blind", "blind_gated"), required=True)
    life.add_argument("--preset", choices=("lean", "rich"), required=True)
    life.add_argument("--n-patches", type=int, required=True)
    life.add_argument("--seed", type=int, required=True)
    life.add_argument("--life-s", type=float, default=6000.0)
    life.add_argument("--window-s", type=float, default=600.0)
    life.add_argument("--eat-below", type=float, default=0.78)
    life.add_argument("--leave-above", type=float, default=0.82)
    life.add_argument("--smell-min", type=float, default=0.2)
    life.add_argument("--control-s", type=float, default=0.1)
    life.add_argument("--hold-s", type=float, default=10.0)
    life.add_argument("--reverse", action="store_true")
    life.add_argument("--out", default=None)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.out is not None and os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite existing file: {args.out}")
    result = run_life(args, out_path=args.out)
    life = result["life"]
    print(
        f"policy={args.policy} preset={args.preset} n_patches={args.n_patches} "
        f"seed={args.seed} life_s={args.life_s!r} eat_below={args.eat_below!r} "
        f"hold_s={args.hold_s!r} "
        f"leave_above={args.leave_above!r} smell_min={args.smell_min!r} "
        f"reverse={args.reverse} mean_energy={life['mean_energy']!r} "
        f"min_energy={life['min_energy']!r} mean_abs_need={life['mean_abs_need']!r} "
        f"mean_need_pos={life['mean_need_pos']!r} mean_need_neg={life['mean_need_neg']!r} "
        f"food_eaten={life['food_eaten']!r} path_m={life['path_m']!r} "
        f"wall_seconds={life['wall_seconds']!r} steps={life['steps']} "
        f"steps_per_second={life['steps_per_second']!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
