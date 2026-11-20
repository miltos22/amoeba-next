"""Known-answer checks for late_need.py, verdict.py and power.py.

Every case is a hand-built CSV (or config) whose answer can be worked out
on paper; the verdict and power functions are then checked against it.
The Student-t quantiles of the design (section 3.6) are asserted to 4
decimals, and the power known answers of section 4.12 are checked here.

Run from the Amobea root:
    PYTHONPATH=. .venv/bin/python -m pytest experiments/stage0_test/test_verdict.py -s -q
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_STAGE0 = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_STAGE0))
for path in (_STAGE0, _ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import late_need as late_need_module  # noqa: E402
import power  # noqa: E402
import verdict  # noqa: E402

CSV_HEADER = "cell,arm,seed,late_need\n"


def _write_csv(path, cell, brain, broken, stream, bstar, smell, drop=None):
    rows = []
    for seed in range(len(brain)):
        for arm, values in (
            ("BRAIN", brain),
            ("BROKEN", broken),
            ("STREAM-AC", stream),
            ("BSTAR", bstar),
            ("FORAGER-SMELL", smell),
        ):
            if drop is not None and (arm, seed) == drop:
                continue
            rows.append((cell, arm, seed, values[seed]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(CSV_HEADER)
        for row in rows:
            fh.write(f"{row[0]},{row[1]},{row[2]},{row[3]!r}\n")
    return path


def _sim_config(brain_mean, brain_sd, brain_sd_others=1e-9, n_seeds=8):
    return {
        "n_seeds": n_seeds,
        "n_simulations": 100,
        "rng_seed": 424242,
        "f_a": 1.0 / 3.0,
        "f_c": 1.0 / 6.0,
        "cells": {
            "cell": {
                "BRAIN": {"mean": brain_mean, "sd": brain_sd},
                "BROKEN": {"mean": 0.30, "sd": brain_sd_others},
                "STREAM-AC": {"mean": 0.105, "sd": brain_sd_others},
                "BSTAR": {"mean": 0.1472, "sd": brain_sd_others},
                "FORAGER-SMELL": {"mean": 0.075, "sd": brain_sd_others},
            }
        },
    }


# -- item 9: the t quantiles of section 3.6 --------------------------------
def test_t_quantiles_match_the_brief():
    values = {4: 2.1318, 9: 1.8331, 29: 1.6991}
    for df, expected in values.items():
        got = verdict.t_quantile_95(df)
        print(f"t(0.95,{df})={got!r} expected={expected!r}")
        assert round(got, 4) == expected


# -- item 10: known-answer CSVs --------------------------------------------
def test_case_a_brain_far_below_and_broken_at_bstar_gives_A_PASS(tmp_path):
    path = tmp_path / "case_a.csv"
    brain = [0.100, 0.101, 0.099, 0.1005, 0.0995]
    bstar = [0.30, 0.30, 0.30, 0.30, 0.30]
    smell = [0.15, 0.15, 0.15, 0.15, 0.15]
    broken = [0.30, 0.30, 0.30, 0.30, 0.30]
    stream = [0.12, 0.12, 0.12, 0.12, 0.12]
    _write_csv(path, "cell", brain, broken, stream, bstar, smell)
    result = verdict.evaluate(verdict.load_csv(str(path)))["cell"]

    # Hand-computed D values, f_A = 1/3, f_C = 1/6, B* = 0.30, S = 0.15:
    #   G      = mean(0.30) - mean(0.15) = 0.15
    #   line   = B* - f_A * (B* - S) = 0.30 - (1/3) * 0.15 = 0.25
    #   D_A1   = BRAIN - line
    #          = [0.100-0.25, 0.101-0.25, 0.099-0.25, 0.1005-0.25, 0.0995-0.25]
    #          = [-0.15, -0.149, -0.151, -0.1495, -0.1505], mean -0.15
    #   D_A2   = BRAIN - BROKEN + f_A * G = BRAIN - 0.30 + 0.05
    #          = the same list, mean -0.15
    #   D_K    = BROKEN - line = 0.30 - 0.25 = 0.05 on every seed, mean 0.05
    #   D_C    = BRAIN - STREAM-AC - f_C * G = BRAIN - 0.12 - 0.025
    #          = [-0.045, -0.044, -0.046, -0.0445, -0.0455], mean -0.045
    hand = {"D_A1": -0.15, "D_A2": -0.15, "D_K": 0.05, "D_C": -0.045}
    print(f"(a) G={result['G']!r} hand=0.15")
    for name, expected in hand.items():
        got = result[name]["mean"]
        print(f"(a) {name} mean={got!r} hand={expected!r}")
        assert abs(got - expected) <= 1e-12
    assert abs(result["G"] - 0.15) <= 1e-12
    assert result["claim_A"] == "PASS"
    assert result["claim_C"] == "PASS"


def test_case_b_brain_equal_bstar_gives_A_FAIL(tmp_path):
    path = tmp_path / "case_b.csv"
    five = [0.30] * 5
    smell = [0.15] * 5
    stream = [0.12] * 5
    _write_csv(path, "cell", five, five, stream, five, smell)
    result = verdict.evaluate(verdict.load_csv(str(path)))["cell"]
    print(f"(b) D_A1 mean={result['D_A1']['mean']!r} Lo={result['D_A1']['Lo']!r}")
    assert result["claim_A"] == "FAIL"


def test_case_c_broken_as_good_as_brain_gives_A_INVALID(tmp_path):
    path = tmp_path / "case_c.csv"
    brain = [0.20] * 5
    bstar = [0.30] * 5
    smell = [0.15] * 5
    _write_csv(path, "cell", brain, brain, [0.12] * 5, bstar, smell)
    result = verdict.evaluate(verdict.load_csv(str(path)))["cell"]
    print(f"(c) D_K mean={result['D_K']['mean']!r} U={result['D_K']['U']!r}")
    assert result["claim_A"] == "INVALID"


def test_case_d_brain_around_the_line_gives_A_INCONCLUSIVE(tmp_path):
    path = tmp_path / "case_d.csv"
    brain = [0.23, 0.27, 0.25, 0.24, 0.26]
    bstar = [0.30] * 5
    smell = [0.15] * 5
    _write_csv(path, "cell", brain, [0.30] * 5, [0.12] * 5, bstar, smell)
    result = verdict.evaluate(verdict.load_csv(str(path)))["cell"]
    print(
        f"(d) D_A1 mean={result['D_A1']['mean']!r} U={result['D_A1']['U']!r} "
        f"Lo={result['D_A1']['Lo']!r} D_K U={result['D_K']['U']!r}"
    )
    assert result["claim_A"] == "INCONCLUSIVE"


def test_case_e_brain_equal_stream_ac_gives_C_PASS(tmp_path):
    path = tmp_path / "case_e.csv"
    brain = [0.120, 0.121, 0.119, 0.1205, 0.1195]
    stream = [0.120, 0.1205, 0.1195, 0.120, 0.120]
    bstar = [0.30] * 5
    smell = [0.15] * 5
    _write_csv(path, "cell", brain, [0.30] * 5, stream, bstar, smell)
    result = verdict.evaluate(verdict.load_csv(str(path)))["cell"]
    print(f"(e) D_C mean={result['D_C']['mean']!r} U={result['D_C']['U']!r}")
    assert result["claim_C"] == "PASS"


def test_case_f_brain_worse_by_two_f_c_g_gives_C_FAIL(tmp_path):
    path = tmp_path / "case_f.csv"
    stream = [0.12] * 5
    brain = [0.12 + 2.0 * (1.0 / 6.0) * 0.15] * 5
    bstar = [0.30] * 5
    smell = [0.15] * 5
    _write_csv(path, "cell", brain, [0.30] * 5, stream, bstar, smell)
    result = verdict.evaluate(verdict.load_csv(str(path)))["cell"]
    print(f"(f) D_C mean={result['D_C']['mean']!r} Lo={result['D_C']['Lo']!r}")
    assert result["claim_C"] == "FAIL"


def test_case_g_missing_seed_raises(tmp_path):
    path = tmp_path / "case_g.csv"
    bstar = [0.30] * 5
    smell = [0.15] * 5
    _write_csv(
        path,
        "cell",
        [0.10] * 5,
        [0.30] * 5,
        [0.12] * 5,
        bstar,
        smell,
        drop=("BRAIN", 4),
    )
    data = verdict.load_csv(str(path))
    with pytest.raises(ValueError) as error:
        verdict.evaluate(data)
    print(f"(g) raised: {error.value}")
    assert "BRAIN" in str(error.value)


# -- item 6: late_need selection and header recognition --------------------
def _write_record(path, header, windows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for row in windows:
            fh.write(json.dumps(row) + "\n")


def test_late_need_takes_only_windows_wholly_inside_the_last_30_percent(tmp_path):
    path = tmp_path / "noise_floor.jsonl"
    header = {
        "row": "header",
        "args": {
            "command": "life",
            "preset": "lean",
            "n_patches": 12,
            "behaviour": "still",
            "seed": 0,
            "life_s": 6000.0,
            "window_s": 600.0,
            "out": None,
        },
    }
    windows = []
    for index in range(10):
        windows.append(
            {
                "row": "window",
                "window": index,
                "t_start": float(index * 600),
                "t_end": float((index + 1) * 600),
                "mean_abs_need": float(index + 1),
                "mean_need_pos": 0.5,
                "mean_need_neg": 0.25,
            }
        )
    _write_record(path, header, windows)
    result = late_need_module.late_need(str(path))
    print(f"late_need windows={result['n_windows']} t_first={result['t_first']!r} "
          f"late_need={result['late_need']!r}")
    assert result["record_kind"] == "forager_or_noise_floor"
    assert result["n_windows"] == 3
    assert result["t_first"] == 4200.0
    assert abs(result["late_need"] - 9.0) <= 1e-12
    assert abs(result["late_need_pos"] - 0.5) <= 1e-12
    assert abs(result["late_need_neg"] - 0.25) <= 1e-12


def test_late_need_raises_on_an_unrecognised_header(tmp_path):
    path = tmp_path / "unknown.jsonl"
    _write_record(
        path,
        {"row": "header", "args": {"life_s": 10.0}},
        [{"row": "window", "t_start": 0.0, "t_end": 10.0, "mean_abs_need": 0.1}],
    )
    with pytest.raises(ValueError) as error:
        late_need_module.late_need(str(path))
    print(f"unknown header raised: {error.value}")


# -- item 12: power known answers -----------------------------------------
def test_power_all_sds_one_e_minus_nine_is_deterministic():
    config = _sim_config(brain_mean=0.105, brain_sd=1e-9)
    results = power.simulate(config, n_simulations=50)
    cell = results["cell"]
    reference = verdict.evaluate_cell(
        {
            arm: {seed: config["cells"]["cell"][arm]["mean"] for seed in range(8)}
            for arm in verdict.ARMS
        }
    )
    print(f"(12a) reference claim_A={reference['claim_A']} claim_C={reference['claim_C']}")
    print(f"(12a) fractions_A={cell['fractions_A']!r}")
    print(f"(12a) fractions_C={cell['fractions_C']!r}")
    assert cell["counts_A"][reference["claim_A"]] == cell["n_simulations"]
    assert cell["counts_C"][reference["claim_C"]] == cell["n_simulations"]
    assert sum(cell["counts_A"].values()) == cell["n_simulations"]
    assert sum(cell["counts_C"].values()) == cell["n_simulations"]


def test_power_on_line_puts_u_d_a1_negative_at_five_percent():
    config = _sim_config(brain_mean=0.105, brain_sd=0.05)
    for arm in verdict.ARMS:
        if arm != "BRAIN":
            config["cells"]["cell"][arm]["sd"] = 1e-9
    results = power.simulate(
        config, scenario="on_line", n_seeds=8, n_simulations=20000
    )
    fraction = results["cell"]["fraction_u_a1_negative"]
    print(f"(12b) P(U(D_A1) < 0)={fraction!r} over {results['cell']['n_simulations']} simulations")
    assert 0.04 <= fraction <= 0.06


def test_power_same_seed_twice_is_identical():
    config = _sim_config(brain_mean=0.105, brain_sd=0.035)
    first = power.simulate(config, scenario="on_line", n_simulations=200)
    second = power.simulate(config, scenario="on_line", n_simulations=200)
    print(f"(12c) identical={json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)}")
    assert first == second
