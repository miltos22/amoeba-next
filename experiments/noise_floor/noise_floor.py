"""Noise-floor measurement for the stage-0 world.

One process runs one life without a brain: either random actuator targets
redrawn every ``redraw_s`` world seconds, or all targets held at zero.
The world is sampled after every 0.5 world seconds and summarised per
window and over the whole life.  Measurements are written as JSON lines.

Subcommands
    life      run one life (one per process) and write JSON lines
    selftest  known-answer checks that distinguish presets and seeds
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time

import mujoco
import numpy as np

from amoeba.world import Config, World

SAMPLE_S = 0.5
TARGET_LOW = -1.5
TARGET_HIGH = 1.5


def _git_commit(root: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _clock() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write_line(fh, obj) -> None:
    fh.write(json.dumps(obj, allow_nan=False, sort_keys=False) + "\n")
    fh.flush()


def _aggregate(records, start_pos, food_start) -> dict:
    n = len(records)
    energies = [r["energy"] for r in records]
    abs_need = [r["abs_need"] for r in records]
    need_pos = [r["need_pos"] for r in records]
    need_neg = [r["need_neg"] for r in records]
    below_half = [1.0 if r["below_half"] else 0.0 for r in records]
    zero = [1.0 if r["zero"] else 0.0 for r in records]
    touch = [1.0 if r["any_touch"] else 0.0 for r in records]
    finite = all(bool(r["finite"]) for r in records)

    path = 0.0
    radii = 0.0
    prev = start_pos
    for r in records:
        x, y = r["pos"]
        path += math.hypot(x - prev[0], y - prev[1])
        radii += math.hypot(x, y)
        prev = r["pos"]
    last = records[-1]["pos"]
    net = math.hypot(last[0] - start_pos[0], last[1] - start_pos[1])

    return {
        "n_samples": n,
        "mean_energy": float(np.mean(energies)),
        "min_energy": float(np.min(energies)),
        "max_energy": float(np.max(energies)),
        "mean_abs_need": float(np.mean(abs_need)),
        "mean_need_pos": float(np.mean(need_pos)),
        "mean_need_neg": float(np.mean(need_neg)),
        "frac_energy_below_half": float(np.mean(below_half)),
        "frac_energy_zero": float(np.mean(zero)),
        "food_eaten": float(records[-1]["food_cum"] - food_start),
        "path_m": float(path),
        "net_m": float(net),
        "mean_radius_m": float(radii / n),
        "frac_any_touch": float(np.mean(touch)),
        "finite": finite,
    }


def _header(args, cfg: Config) -> dict:
    return {
        "row": "header",
        "args": dict(vars(args)),
        "config": cfg.as_dict(),
        "git_commit": _git_commit(_project_root()),
        "python_version": sys.version,
        "python_version_info": list(sys.version_info),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "mujoco_version": mujoco.__version__,
        "start_time": _clock(),
        "sample_s": SAMPLE_S,
    }


def run_life(args, out_path: str | None = None):
    if args.behaviour == "random":
        if args.redraw_s <= 0.0:
            raise SystemExit("--redraw-s must be positive for behaviour random")
        ratio = args.redraw_s / SAMPLE_S
        if abs(ratio - round(ratio)) > 1e-9:
            raise SystemExit(f"--redraw-s must be a whole multiple of {SAMPLE_S}")
        if args.target_range <= 0.0:
            raise SystemExit("--target-range must be positive")
        redraw_samples = int(round(ratio))
    else:
        redraw_samples = 0

    for name, value in (("life_s", args.life_s), ("window_s", args.window_s)):
        ratio = value / SAMPLE_S
        if abs(ratio - round(ratio)) > 1e-9:
            raise SystemExit(f"--{name} must be a whole multiple of {SAMPLE_S}")

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
        _write_line(fh, _header(args, cfg))

    set_point = cfg.preset_constants.energy_set_point
    wall_start = time.perf_counter()
    world = World(cfg)
    names = [item.name for item in world.items() if item.kind == "actuator"]
    if args.behaviour == "random":
        target_rng = np.random.default_rng(1000 + args.seed)
    else:
        world.write({name: 0.0 for name in names})

    start_pos = world.body.centre()
    records = []
    windows = []
    current = []
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
        row.update(_aggregate(current, window_start_pos, window_start_food))
        windows.append(row)
        if fh is not None:
            _write_line(fh, row)
        window_start_food = current[-1]["food_cum"]
        window_start_pos = current[-1]["pos"]
        window_index += 1
        current = []
        return row

    for step_index in range(n_total):
        if args.behaviour == "random" and step_index % redraw_samples == 0:
            world.write({name: float(target_rng.uniform(-args.target_range, args.target_range)) for name in names})
        world.advance(SAMPLE_S)
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
    life_row.update(_aggregate(records, start_pos, 0.0))
    life_row.update(
        {
            "wall_seconds": float(wall_seconds),
            "steps": steps,
            "steps_per_second": float(steps / wall_seconds),
        }
    )
    if fh is not None:
        _write_line(fh, life_row)
        fh.close()
    return {"header": _header(args, cfg), "windows": windows, "life": life_row, "records": records}


# -- selftest ---------------------------------------------------------------
def _cli_args(**overrides):
    base = dict(
        preset="lean",
        n_patches=0,
        behaviour="still",
        redraw_s=0.5,
        target_range=1.5,
        seed=0,
        life_s=1200.0,
        window_s=600.0,
        out=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _fail(failures, message):
    failures.append(message)
    print(f"FAIL {message}")


def selftest() -> int:
    failures: list[str] = []

    print("selftest 1: still, n_patches=0, lean, life 1200 s, window 600 s")
    lean = run_life(_cli_args(preset="lean", behaviour="still", life_s=1200.0, window_s=600.0))
    reach = [r["t"] for r in lean["records"] if r["energy"] == 0.0]
    print(f"  first_zero_energy_time={reach[0]!r}" if reach else "  first_zero_energy_time=None")
    w0, w1 = lean["windows"][0], lean["windows"][1]
    print(f"  window0.food_eaten={w0['food_eaten']!r}")
    print(f"  window0.path_m={w0['path_m']!r}")
    print(f"  window1.mean_abs_need={w1['mean_abs_need']!r}")
    print(f"  window1.frac_energy_zero={w1['frac_energy_zero']!r}")
    print(f"  window1.mean_energy={w1['mean_energy']!r} min={w1['min_energy']!r} max={w1['max_energy']!r}")
    if not reach:
        _fail(failures, "lean: energy never reached 0.0")
    elif abs(reach[0] - 600.0) > 1.5:
        _fail(failures, f"lean: energy reached 0.0 at {reach[0]!r}, wanted 600 s within 1.5 s")
    if abs(w1["mean_abs_need"] - 0.8) > 1e-9:
        _fail(failures, f"lean: window1 mean_abs_need {w1['mean_abs_need']!r} != 0.8 within 1e-9")
    if not w1["frac_energy_zero"] >= 0.99:
        _fail(failures, f"lean: window1 frac_energy_zero {w1['frac_energy_zero']!r} < 0.99")
    if w0["food_eaten"] != 0.0:
        _fail(failures, f"lean: window0 food_eaten {w0['food_eaten']!r} != 0.0")
    if not w0["path_m"] < 1e-6:
        _fail(failures, f"lean: window0 path_m {w0['path_m']!r} >= 1e-6")

    print("selftest 2: still, n_patches=0, rich, life 1200 s")
    rich = run_life(_cli_args(preset="rich", behaviour="still", life_s=1200.0, window_s=600.0))
    rich_reach = [r["t"] for r in rich["records"] if r["energy"] == 0.0]
    rich_end = rich["life"]["mean_energy"]
    print(f"  first_zero_energy_time={rich_reach[0]!r}" if rich_reach else "  first_zero_energy_time=None")
    print(f"  end_energy={rich['records'][-1]['energy']!r}")
    print(f"  life.mean_energy={rich_end!r} min={rich['life']['min_energy']!r} max={rich['life']['max_energy']!r}")
    if rich_reach:
        _fail(failures, f"rich: energy reached 0.0 at {rich_reach[0]!r}; it must not within 1200 s")
    if not rich["records"][-1]["energy"] > 0.3:
        _fail(failures, f"rich: end energy {rich['records'][-1]['energy']!r} <= 0.3")

    print("selftest 3: random, lean, 12 patches, seed 3 twice and seed 4")
    common = dict(preset="lean", n_patches=12, behaviour="random", redraw_s=0.5, life_s=120.0, window_s=60.0)
    a = run_life(_cli_args(seed=3, **common))
    b = run_life(_cli_args(seed=3, **common))
    c = run_life(_cli_args(seed=4, **common))
    print(f"  seed3_run_a.windows={json.dumps(a['windows'])}")
    print(f"  seed3_run_b.windows={json.dumps(b['windows'])}")
    print(f"  seed3_a.life.mean_energy={a['life']['mean_energy']!r}")
    print(f"  seed3_b.life.mean_energy={b['life']['mean_energy']!r}")
    print(f"  seed4.life.mean_energy={c['life']['mean_energy']!r}")
    print(f"  seed3_a.life.mean_abs_need={a['life']['mean_abs_need']!r}")
    print(f"  seed4.life.mean_abs_need={c['life']['mean_abs_need']!r}")
    if a["windows"] != b["windows"]:
        _fail(failures, "identical random lives have different window rows")
    if a["life"]["mean_energy"] == c["life"]["mean_energy"]:
        _fail(failures, "seed 4 mean_energy equals seed 3 mean_energy")

    if failures:
        print(f"SELFTEST FAILED: {len(failures)} failure(s)")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("SELFTEST PASSED")
    return 0


# -- command line -----------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    life = sub.add_parser("life", help="run one life and write JSON lines")
    life.add_argument("--preset", choices=("lean", "rich"), required=True)
    life.add_argument("--n-patches", type=int, required=True)
    life.add_argument("--behaviour", choices=("random", "still"), required=True)
    life.add_argument("--redraw-s", type=float, default=0.5)
    life.add_argument("--target-range", type=float, default=1.5)
    life.add_argument("--seed", type=int, required=True)
    life.add_argument("--life-s", type=float, default=6000.0)
    life.add_argument("--window-s", type=float, default=600.0)
    life.add_argument("--out", default=None)

    sub.add_parser("selftest", help="run the known-answer checks")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "selftest":
        return selftest()
    if args.out is not None and os.path.exists(args.out):
        raise SystemExit(f"refusing to overwrite existing file: {args.out}")
    result = run_life(args, out_path=args.out)
    life = result["life"]
    print(
        f"preset={args.preset} n_patches={args.n_patches} behaviour={args.behaviour} "
        f"redraw_s={args.redraw_s!r} seed={args.seed} life_s={args.life_s!r} "
        f"mean_energy={life['mean_energy']!r} min_energy={life['min_energy']!r} "
        f"mean_abs_need={life['mean_abs_need']!r} food_eaten={life['food_eaten']!r} "
        f"path_m={life['path_m']!r} wall_seconds={life['wall_seconds']!r} "
        f"steps={life['steps']} steps_per_second={life['steps_per_second']!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
