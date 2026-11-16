"""Power simulation for the stage-0 verdicts (CRITERIA_DRAFT.md section 9).

Input: a JSON config giving, per cell, the mean and across-life SD of
late_need for each of the five arms (BRAIN, BROKEN, STREAM-AC, BSTAR,
FORAGER-SMELL), plus ``n_seeds``, ``n_simulations``, ``rng_seed`` and
optionally ``f_a`` / ``f_c``:

    {
      "n_seeds": 8, "n_simulations": 20000, "rng_seed": 20261115,
      "f_a": 0.3333333333333333, "f_c": 0.16666666666666666,
      "cells": {
        "lean12": {
          "BRAIN": {"mean": 0.105, "sd": 0.035},
          "BROKEN": {"mean": 0.30, "sd": 0.035},
          "STREAM-AC": {"mean": 0.105, "sd": 0.035},
          "BSTAR": {"mean": 0.1472, "sd": 0.035},
          "FORAGER-SMELL": {"mean": 0.075, "sd": 0.035}
        }
      }
    }

Each simulated set draws every arm's lives as independent normal values
and hands the set to ``verdict.evaluate_cell`` itself (imported, not
reimplemented); the fraction of each verdict is printed for claims A and
C.

``--scenario`` overrides BRAIN's mean with:
    equals_broken      BRAIN = BROKEN
    equals_bstar       BRAIN = BSTAR
    equals_stream_ac   BRAIN = STREAM-AC
    on_line            BRAIN = BSTAR - f_A * G, BROKEN = BSTAR

    PYTHONPATH=. .venv/bin/python experiments/stage0_test/power.py CONFIG.json [--scenario NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import verdict  # noqa: E402

SCENARIOS = ("equals_broken", "equals_bstar", "equals_stream_ac", "on_line")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    if "cells" not in config or not isinstance(config["cells"], dict):
        raise ValueError(f"{path}: config has no cells object")
    for cell, arms in config["cells"].items():
        if not isinstance(arms, dict):
            raise ValueError(f"{path}: cell {cell!r} is not an object")
        for arm in verdict.ARMS:
            if arm not in arms:
                raise ValueError(f"{path}: cell {cell!r} is missing arm {arm!r}")
            entry = arms[arm]
            if not isinstance(entry, dict) or "mean" not in entry or "sd" not in entry:
                raise ValueError(f"{path}: cell {cell!r} arm {arm!r} needs mean and sd")
            if float(entry["sd"]) < 0.0:
                raise ValueError(f"{path}: cell {cell!r} arm {arm!r} has a negative sd")
    return config


def arm_means(cell_arms: dict, scenario: str | None, f_a: float) -> dict:
    """The means to draw from, after the scenario override."""
    means = {arm: float(cell_arms[arm]["mean"]) for arm in verdict.ARMS}
    if scenario is None:
        return means
    if scenario == "equals_broken":
        means["BRAIN"] = means["BROKEN"]
    elif scenario == "equals_bstar":
        means["BRAIN"] = means["BSTAR"]
    elif scenario == "equals_stream_ac":
        means["BRAIN"] = means["STREAM-AC"]
    elif scenario == "on_line":
        g_value = means["BSTAR"] - means["FORAGER-SMELL"]
        means["BRAIN"] = means["BSTAR"] - f_a * g_value
        means["BROKEN"] = means["BSTAR"]
    else:
        raise ValueError(f"unknown scenario {scenario!r}; known: {SCENARIOS}")
    return means


def simulate(config: dict, scenario: str | None = None, n_seeds: int | None = None,
             n_simulations: int | None = None, rng_seed: int | None = None,
             f_a: float | None = None, f_c: float | None = None) -> dict:
    """Run the power simulation; return counts and fractions per cell."""
    if scenario is not None and scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; known: {SCENARIOS}")
    n_seeds = int(config.get("n_seeds", 8) if n_seeds is None else n_seeds)
    n_simulations = int(
        config.get("n_simulations", 10000) if n_simulations is None else n_simulations
    )
    rng_seed = int(config.get("rng_seed", 0) if rng_seed is None else rng_seed)
    f_a = float(config.get("f_a", 1.0 / 3.0) if f_a is None else f_a)
    f_c = float(config.get("f_c", 1.0 / 6.0) if f_c is None else f_c)
    if n_seeds < 2:
        raise ValueError(f"need at least 2 seeds, got {n_seeds}")
    if n_simulations < 1:
        raise ValueError(f"need at least 1 simulation, got {n_simulations}")

    rng = np.random.default_rng(rng_seed)
    results = {}
    for cell in sorted(config["cells"]):
        cell_arms = config["cells"][cell]
        means = arm_means(cell_arms, scenario, f_a)
        draws = {
            arm: rng.normal(
                means[arm],
                float(cell_arms[arm]["sd"]),
                size=(n_simulations, n_seeds),
            )
            for arm in verdict.ARMS
        }
        counts_a = {name: 0 for name in verdict.CLAIM_A_VERDICTS}
        counts_c = {name: 0 for name in verdict.CLAIM_C_VERDICTS}
        u_a1_negative = 0
        for index in range(n_simulations):
            arms = {
                arm: {seed: float(draws[arm][index, seed]) for seed in range(n_seeds)}
                for arm in verdict.ARMS
            }
            result = verdict.evaluate_cell(arms, f_a=f_a, f_c=f_c)
            counts_a[result["claim_A"]] += 1
            counts_c[result["claim_C"]] += 1
            if result["D_A1"]["U"] < 0.0:
                u_a1_negative += 1
        results[cell] = {
            "scenario": scenario,
            "n_seeds": n_seeds,
            "n_simulations": n_simulations,
            "rng_seed": rng_seed,
            "f_a": f_a,
            "f_c": f_c,
            "means": means,
            "counts_A": counts_a,
            "fractions_A": {name: counts_a[name] / n_simulations for name in counts_a},
            "counts_C": counts_c,
            "fractions_C": {name: counts_c[name] / n_simulations for name in counts_c},
            "u_a1_negative": u_a1_negative,
            "fraction_u_a1_negative": u_a1_negative / n_simulations,
        }
    return results


def print_results(results: dict) -> None:
    for cell in sorted(results):
        result = results[cell]
        print(
            f"cell={cell} scenario={result['scenario']} n_seeds={result['n_seeds']} "
            f"n_simulations={result['n_simulations']} rng_seed={result['rng_seed']} "
            f"f_a={result['f_a']!r} f_c={result['f_c']!r}"
        )
        print("  means=" + json.dumps(result["means"], sort_keys=True))
        for claim in ("A", "C"):
            counts = result[f"counts_{claim}"]
            fractions = result[f"fractions_{claim}"]
            names = verdict.CLAIM_A_VERDICTS if claim == "A" else verdict.CLAIM_C_VERDICTS
            count_text = " ".join(f"{name}={counts[name]}" for name in names)
            fraction_text = " ".join(f"{name}={fractions[name]!r}" for name in names)
            print(f"  claim_{claim} counts: {count_text}")
            print(f"  claim_{claim} fractions: {fraction_text}")
        print(
            f"  P(U(D_A1) < 0)={result['fraction_u_a1_negative']!r} "
            f"({result['u_a1_negative']} of {result['n_simulations']})"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="power config JSON")
    parser.add_argument("--scenario", choices=SCENARIOS, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument("--f-a", type=float, default=None)
    parser.add_argument("--f-c", type=float, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    results = simulate(
        config,
        scenario=args.scenario,
        n_seeds=args.n_seeds,
        n_simulations=args.simulations,
        rng_seed=args.rng_seed,
        f_a=args.f_a,
        f_c=args.f_c,
    )
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
