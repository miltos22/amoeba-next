"""Small setting check for forager_oracle: four settings, seeds 5, 6, 7.

lean, 12 patches, 3000 s lives.  Prints one table row per life with
setting, seed, mean_abs_need, mean_need_pos, mean_need_neg, mean_energy,
food_eaten, path_m.  It does not pick a winner.

    taskset -c 0-7 nice -n 19 .venv/bin/python \
        experiments/upper_reference/settings_check.py

Each life is one process, one after another; existing life files in
--outdir are reused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUTDIR = os.path.join(
    _ROOT, "dispatch", "2026-11-15", "W09_upper_reference", "out", "settings"
)
PREFIX = ["taskset", "-c", "0-7", "nice", "-n", "19"]

SETTINGS = ((0.70, 0.80), (0.78, 0.82), (0.78, 0.90), (0.60, 0.80))
SEEDS = (5, 6, 7)
LIFE_S = 3000.0


def _tag(eat_below: float, leave_above: float) -> str:
    return f"eat{eat_below:.2f}_leave{leave_above:.2f}"


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    return {row["row"]: row for row in rows if row["row"] in ("header", "life")}["life"]


def _run_life(seed: int, eat_below: float, leave_above: float, outdir: str) -> dict:
    out_path = os.path.join(outdir, f"{_tag(eat_below, leave_above)}_s{seed}.jsonl")
    if os.path.exists(out_path):
        try:
            life = _load(out_path)
        except (json.JSONDecodeError, KeyError):
            life = None
        if life is not None:
            print(f"reusing {out_path}")
            return life
        os.remove(out_path)
    cmd = PREFIX + [
        sys.executable,
        os.path.join(_ROOT, "experiments", "upper_reference", "forager_life.py"),
        "life",
        "--policy", "oracle",
        "--preset", "lean",
        "--n-patches", "12",
        "--seed", str(seed),
        "--life-s", repr(LIFE_S),
        "--window-s", "600.0",
        "--eat-below", repr(eat_below),
        "--leave-above", repr(leave_above),
        "--out", out_path,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT
    log_path = out_path[:-6] + ".log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=_ROOT, env=env)
    wall = time.perf_counter() - t0
    if rc != 0:
        raise SystemExit(f"life failed (rc={rc}): {' '.join(cmd)} (see {log_path})")
    print(f"ran {os.path.basename(out_path)} in {wall!r} s")
    return _load(out_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    columns = ("setting", "seed", "mean_abs_need", "mean_need_pos", "mean_need_neg",
               "mean_energy", "food_eaten", "path_m")
    print("  ".join(columns))
    for eat_below, leave_above in SETTINGS:
        for seed in SEEDS:
            life = _run_life(seed, eat_below, leave_above, args.outdir)
            print(
                f"({eat_below:.2f}, {leave_above:.2f})  {seed}  "
                f"{life['mean_abs_need']!r}  {life['mean_need_pos']!r}  "
                f"{life['mean_need_neg']!r}  {life['mean_energy']!r}  "
                f"{life['food_eaten']!r}  {life['path_m']!r}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
