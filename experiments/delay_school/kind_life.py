"""Run one forager life with food kinds and write the noise-floor life format.

The rows are produced with the same functions as
experiments/noise_floor/noise_floor.py (imported, not copied) and the gait
timing of experiments/upper_reference/forager_life.py (imported).  The
first line is the header row holding the Config actually in force
(``as_dict``); then one window row per 600 s, then one life row.  Rows
carry the usual fields plus cumulative eaten and delivered energy per
kind and the gut load.

Policies
    informed       KindForager on the kinds named by --approach
    indiscriminate KindForager on all kinds
    random_full    uniform targets in [TARGET_LOW, TARGET_HIGH] redrawn
                   every 0.5 s, as noise_floor's "random" behaviour
    still          all targets zero, as noise_floor's "still" behaviour

Run from the Amobea root (or the worktree) with PYTHONPATH pointing at it:

    .venv/bin/python experiments/delay_school/kind_life.py \
        --preset lean --n-patches 12 --kinds "good:0:1:0.5,bad:300:-1:0.5" \
        --gut-sense 0 --policy informed --approach good --seed 0 \
        --life-s 1500 --out life.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "noise_floor"))

import noise_floor as nf  # noqa: E402

from amoeba.world import Config, FoodKind, World  # noqa: E402
from experiments.delay_school.kind_policies import KindForager  # noqa: E402
from experiments.upper_reference import forager_life as fl  # noqa: E402

SAMPLE_S = nf.SAMPLE_S
GAIT_S = fl.GAIT_S


def parse_kinds(text: str) -> tuple[FoodKind, ...]:
    kinds = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 4:
            raise SystemExit(f"bad kind {chunk!r}; expected name:delay:gain:share")
        name, delay, gain, share = parts
        kinds.append(FoodKind(name.strip(), float(delay), float(gain), float(share)))
    if not kinds:
        raise SystemExit("--kinds needs at least one kind")
    return tuple(kinds)


def _make_policy(args, world):
    if args.policy == "informed":
        approach = {name.strip() for name in args.approach.split(",") if name.strip()}
        if not approach:
            raise SystemExit("policy informed needs --approach")
    elif args.policy == "indiscriminate":
        approach = {kind.name for kind in world.config.food_kinds}
    else:
        return None
    return KindForager(
        world,
        approach,
        eat_below=args.eat_below,
        leave_above=args.leave_above,
        smell_min=args.smell_min,
        seed=args.seed,
        reverse=args.reverse,
    )


def run_life(args, out_path: str | None = None):
    for name, value in (("life_s", args.life_s), ("window_s", args.window_s)):
        ratio = value / SAMPLE_S
        if abs(ratio - round(ratio)) > 1e-9:
            raise SystemExit(f"--{name} must be a whole multiple of {SAMPLE_S}")
    if abs(args.control_s / GAIT_S - round(args.control_s / GAIT_S)) > 1e-9:
        raise SystemExit(f"--control-s must be a whole multiple of {GAIT_S}")

    kinds = parse_kinds(args.kinds)
    cfg = Config(
        seed=args.seed,
        preset=args.preset,
        n_patches=args.n_patches,
        patch_radius=0.25,
        dish_radius=1.5,
        food_kinds=kinds,
        gut_sense=bool(args.gut_sense),
    )
    kind_names = [kind.name for kind in kinds]
    fh = None
    if out_path is not None:
        fh = open(out_path, "x", encoding="utf-8")
        nf._write_line(fh, nf._header(args, cfg))

    set_point = cfg.preset_constants.energy_set_point
    wall_start = time.perf_counter()
    world = World(cfg)
    policy = _make_policy(args, world)
    actuator_names = [item.name for item in world.items() if item.kind == "actuator"]
    target_rng = (
        np.random.default_rng(1000 + args.seed) if args.policy == "random_full" else None
    )
    if args.policy in ("random_full", "still"):
        world.write({name: 0.0 for name in actuator_names})

    sample_ticks = int(round(SAMPLE_S / GAIT_S))
    control_ticks = int(round(args.control_s / GAIT_S))
    n_ticks = int(round(args.life_s / GAIT_S))

    start_pos = world.body.centre()
    records: list[dict] = []
    windows: list[dict] = []
    current: list[dict] = []
    window_start_food = 0.0
    window_start_eaten = np.zeros(len(kinds))
    window_start_delivered = np.zeros(len(kinds))
    window_start_pos = start_pos
    window_index = 0

    def flush_window():
        nonlocal current, window_start_food, window_start_pos, window_index
        nonlocal window_start_eaten, window_start_delivered
        if not current:
            return None
        row = {
            "row": "window",
            "window": window_index,
            "t_start": float(current[0]["t"] - SAMPLE_S),
            "t_end": float(current[-1]["t"]),
        }
        row.update(nf._aggregate(current, window_start_pos, window_start_food))
        row["eaten_by_kind"] = {
            name: float(current[-1]["eaten_by_kind"][i] - window_start_eaten[i])
            for i, name in enumerate(kind_names)
        }
        row["delivered_by_kind"] = {
            name: float(current[-1]["delivered_by_kind"][i] - window_start_delivered[i])
            for i, name in enumerate(kind_names)
        }
        row["gut_load"] = float(current[-1]["gut_load"])
        windows.append(row)
        if fh is not None:
            nf._write_line(fh, row)
        window_start_food = current[-1]["food_cum"]
        window_start_eaten = current[-1]["eaten_by_kind"].copy()
        window_start_delivered = current[-1]["delivered_by_kind"].copy()
        window_start_pos = current[-1]["pos"]
        window_index += 1
        current = []
        return row

    for tick in range(n_ticks):
        if target_rng is not None and tick % sample_ticks == 0:
            world.write(
                {
                    name: float(target_rng.uniform(nf.TARGET_LOW, nf.TARGET_HIGH))
                    for name in actuator_names
                }
            )
        if policy is not None and tick % control_ticks == 0:
            policy.update()
        if policy is not None:
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
                    "any_touch": any(
                        reading[f"arm{k}.touch"] == 1.0 for k in range(cfg.n_arms)
                    ),
                    "finite": bool(np.isfinite(list(reading.values())).all()),
                    "food_cum": float(world.food_eaten),
                    "eaten_by_kind": world.food_eaten_by_kind.copy(),
                    "delivered_by_kind": world.food_delivered_by_kind.copy(),
                    "gut_load": float(world.gut_load),
                }
            )
            current.append(records[-1])
            if len(current) >= int(round(args.window_s / SAMPLE_S)):
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
    life_row["eaten_by_kind"] = {
        name: float(world.food_eaten_by_kind[i]) for i, name in enumerate(kind_names)
    }
    life_row["delivered_by_kind"] = {
        name: float(world.food_delivered_by_kind[i]) for i, name in enumerate(kind_names)
    }
    life_row["gut_load"] = float(world.gut_load)
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
    return {
        "header": nf._header(args, cfg),
        "windows": windows,
        "life": life_row,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("lean", "rich"), required=True)
    parser.add_argument("--n-patches", type=int, required=True)
    parser.add_argument("--kinds", required=True, help='"name:delay:gain:share,..."')
    parser.add_argument("--gut-sense", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--policy",
        choices=("informed", "indiscriminate", "random_full", "still"),
        required=True,
    )
    parser.add_argument("--approach", default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--life-s", type=float, default=6000.0)
    parser.add_argument("--window-s", type=float, default=600.0)
    parser.add_argument("--eat-below", type=float, default=0.78)
    parser.add_argument("--leave-above", type=float, default=0.82)
    parser.add_argument("--smell-min", type=float, default=0.2)
    parser.add_argument("--control-s", type=float, default=0.1)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--out", default=None)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.out is not None and os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite existing file: {args.out}")
    result = run_life(args, out_path=args.out)
    life = result["life"]
    print(
        f"policy={args.policy} approach={args.approach!r} preset={args.preset} "
        f"n_patches={args.n_patches} kinds={args.kinds!r} gut_sense={args.gut_sense} "
        f"seed={args.seed} life_s={args.life_s!r} mean_energy={life['mean_energy']!r} "
        f"min_energy={life['min_energy']!r} mean_abs_need={life['mean_abs_need']!r} "
        f"mean_need_pos={life['mean_need_pos']!r} mean_need_neg={life['mean_need_neg']!r} "
        f"food_eaten={life['food_eaten']!r} eaten_by_kind={life['eaten_by_kind']!r} "
        f"delivered_by_kind={life['delivered_by_kind']!r} path_m={life['path_m']!r} "
        f"wall_seconds={life['wall_seconds']!r} steps={life['steps']} "
        f"steps_per_second={life['steps_per_second']!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
