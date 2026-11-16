"""Known-answer checks for the upper reference: fires and does not fire.

One command, every number printed, exit non-zero on failure.

 1. Gait fires: each of six 30 s swims moves at least 1.0 cm/s and its
    chord direction is within 25 degrees of the commanded direction.
 2. Gait does not fire: all 12 joints in phase move the centre less than
    0.02 m in 30 s.
 3. Forager fires: lean, 12 patches, 3000 s, seeds 0 and 1: mean_abs_need
    of forager_oracle is lower than the random policy (redraw 0.5 s,
    target range 1.5) at the same seed, on both seeds.
 4. Forager does not fire: forager_oracle --reverse has a HIGHER
    mean_abs_need than forager_oracle at the same seed, on both seeds.

Each life is one process (each well under 6 minutes); the six lives run
one after another.  Existing life files in --outdir are reused, so a
failed run can be checked again without re-simulating.

    taskset -c 0-7 nice -n 19 .venv/bin/python \
        experiments/upper_reference/selftest.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from experiments.upper_reference import measure_gait  # noqa: E402
from experiments.upper_reference.gait import _wrap  # noqa: E402

PREFIX = ["taskset", "-c", "0-7", "nice", "-n", "19"]
DEFAULT_OUTDIR = os.path.join(
    _ROOT, "dispatch", "2026-11-15", "W09_upper_reference", "out", "selftest"
)


def _life_output(directory: str, name: str) -> str:
    return os.path.join(directory, name + ".jsonl")


def _run_life(cmd: list[str], out_path: str) -> dict:
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        if any(row.get("row") == "life" for row in rows):
            print(f"  reusing {out_path}")
            return {row["row"]: row for row in rows if row["row"] in ("header", "life")}["life"]
    if os.path.exists(out_path):
        os.remove(out_path)
    log_path = out_path[:-6] + ".log"
    t0 = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT
    with open(log_path, "w", encoding="utf-8") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=_ROOT, env=env)
    wall = time.perf_counter() - t0
    if rc != 0:
        raise SystemExit(f"life failed (rc={rc}): {' '.join(cmd)} (see {log_path})")
    with open(out_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    life = {row["row"]: row for row in rows if row["row"] in ("header", "life")}["life"]
    print(f"  ran {os.path.basename(out_path)} in {wall!r} s")
    return life


def _oracle_cmd(seed: int, life_s: float, out_path: str, reverse: bool) -> list[str]:
    cmd = PREFIX + [
        sys.executable,
        os.path.join(_ROOT, "experiments", "upper_reference", "forager_life.py"),
        "life",
        "--policy", "oracle",
        "--preset", "lean",
        "--n-patches", "12",
        "--seed", str(seed),
        "--life-s", repr(life_s),
        "--window-s", repr(min(600.0, life_s)),
        "--eat-below", "0.78",
        "--leave-above", "0.82",
        "--out", out_path,
    ]
    if reverse:
        cmd.append("--reverse")
    return cmd


def _random_cmd(seed: int, life_s: float, out_path: str) -> list[str]:
    return PREFIX + [
        sys.executable,
        os.path.join(_ROOT, "experiments", "noise_floor", "noise_floor.py"),
        "life",
        "--preset", "lean",
        "--n-patches", "12",
        "--behaviour", "random",
        "--redraw-s", "0.5",
        "--target-range", "1.5",
        "--seed", str(seed),
        "--life-s", repr(life_s),
        "--window-s", repr(min(600.0, life_s)),
        "--out", out_path,
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--life-s", type=float, default=3000.0)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    failures: list[str] = []

    print("selftest 1/2: gait, six 30 s swims and the in-phase flap")
    for j in range(6):
        commanded = math.radians(j * 60.0)
        r = measure_gait._run("swim", commanded)
        error = math.degrees(_wrap(math.radians(r["direction_body_deg"]) - commanded))
        print(
            f"  direction_deg={j * 60} displacement_m={r['displacement_m']!r} "
            f"direction_body_deg={r['direction_body_deg']!r} "
            f"direction_error_deg={error!r} "
            f"net_speed_cm_s={r['net_speed_cm_s']!r} "
            f"path_m={r['path_m']!r} yaw_deg={r['yaw_deg']!r}"
        )
        if not r["net_speed_cm_s"] >= 1.0:
            failures.append(f"gait direction {j * 60} deg: speed {r['net_speed_cm_s']!r} cm/s < 1.0")
        if not abs(error) <= 25.0:
            failures.append(f"gait direction {j * 60} deg: error {error!r} deg > 25")
    flap = measure_gait._run("in_phase")
    print(
        f"  in_phase displacement_m={flap['displacement_m']!r} "
        f"path_m={flap['path_m']!r} yaw_deg={flap['yaw_deg']!r}"
    )
    if not flap["displacement_m"] < 0.02:
        failures.append(f"in-phase displacement {flap['displacement_m']!r} m >= 0.02")

    print(f"selftest 3: forager fires (lean, 12 patches, {args.life_s!r} s, seeds 0 and 1)")
    results = {}
    for seed in (0, 1):
        oracle_path = _life_output(args.outdir, f"oracle_s{seed}")
        random_path = _life_output(args.outdir, f"random_s{seed}")
        print(f"  seed {seed}: oracle")
        results[("oracle", seed)] = _run_life(_oracle_cmd(seed, args.life_s, oracle_path, False), oracle_path)
        print(f"  seed {seed}: random")
        results[("random", seed)] = _run_life(_random_cmd(seed, args.life_s, random_path), random_path)
        for policy in ("oracle", "random"):
            row = results[(policy, seed)]
            print(
                f"  seed={seed} policy={policy} mean_abs_need={row['mean_abs_need']!r} "
                f"mean_energy={row['mean_energy']!r} food_eaten={row['food_eaten']!r} "
                f"path_m={row['path_m']!r} wall_seconds={row['wall_seconds']!r}"
            )
        if not results[("oracle", seed)]["mean_abs_need"] < results[("random", seed)]["mean_abs_need"]:
            failures.append(
                f"seed {seed}: oracle mean_abs_need {results[('oracle', seed)]['mean_abs_need']!r} "
                f">= random {results[('random', seed)]['mean_abs_need']!r}"
            )

    print("selftest 4: forager does not fire (oracle --reverse vs oracle)")
    for seed in (0, 1):
        reverse_path = _life_output(args.outdir, f"oracle_reverse_s{seed}")
        print(f"  seed {seed}: oracle --reverse")
        results[("reverse", seed)] = _run_life(
            _oracle_cmd(seed, args.life_s, reverse_path, True), reverse_path
        )
        row = results[("reverse", seed)]
        print(
            f"  seed={seed} policy=oracle_reverse mean_abs_need={row['mean_abs_need']!r} "
            f"mean_energy={row['mean_energy']!r} food_eaten={row['food_eaten']!r} "
            f"path_m={row['path_m']!r} wall_seconds={row['wall_seconds']!r}"
        )
        if not results[("reverse", seed)]["mean_abs_need"] > results[("oracle", seed)]["mean_abs_need"]:
            failures.append(
                f"seed {seed}: reverse mean_abs_need {results[('reverse', seed)]['mean_abs_need']!r} "
                f"<= oracle {results[('oracle', seed)]['mean_abs_need']!r}"
            )

    if failures:
        print(f"SELFTEST FAILED: {len(failures)} failure(s)")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("SELFTEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
