"""Follow-up 2 measurements: RECEPTOR mode (organ-time CHORD).

Fresh outputs go to out2/ (results.json, tables.md, part_*.json).  Reads
out/results.json only for the CHORD tau_phi = 0.2 comparison.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from step0a.experience_time import (
    DELTA, SIGMA_HAT_P, SIGMA_MIN, Z_P, Z_REF, OMEGA_P, GridGroup,
    organ_stream, read_organ, simulate,
)
from step0a.pacemaker import Pacemaker
from step0a.receptors import Receptor, calibrate_sigma_hat
from step0a.world_signals import (
    H_W, Source, look_indices, n_grid,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT2 = os.path.join(ROOT, "runs", "out2")
OUT1 = os.path.join(ROOT, "runs", "out")
SEEDS = (0, 1, 2)
SCHEDULES = ("2", "5", "15", "50", "irregular")
CASES = (("quiet", 73000.0), ("quiet_noise", 73000.0), ("active", 20000.0))
TOL_CLEAN = 0.05
TOL_NOISY = 0.10
SIGMA_N = 0.2


def _h_tag(h):
    d = H_W / h
    return f"hdiv{int(round(d))}"


def _part(name):
    return os.path.join(OUT2, f"part_{name}.json")


def _save(name, obj):
    os.makedirs(OUT2, exist_ok=True)
    with open(_part(name), "w") as f:
        json.dump(obj, f)


def _load(name):
    p = _part(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _regime_name(noisy):
    return "noisy" if noisy else "clean"


# ---------------------------------------------------------------------------
# groups on the world grid
# ---------------------------------------------------------------------------

def build_groups(case, seed, h, life):
    n = n_grid(life, h)
    groups = []
    if case == "quiet":
        for i in range(10):
            groups.append(GridGroup(f"silent{i}", None, np.array([0.0]),
                                    np.array([SIGMA_MIN])))
        return groups, n
    if case == "quiet_noise":
        for g in range(10):
            src = Source("silent", seed=seed * 100 + g, sigma_n=SIGMA_N)
            x = Receptor(src, 0.2).filtered(n, h)
            sig = calibrate_sigma_hat(src, tau=0.2, h=h)
            groups.append(GridGroup(f"noise{g}", x[None, :],
                                    np.array([sig]), np.array([SIGMA_MIN])))
        return groups, n
    if case == "active":
        for name, kind in (("steps", "steps"), ("bursts", "bursts"),
                           ("ramp", "ramp")):
            src = Source(kind, seed=seed)
            x = Receptor(src, 0.2).filtered(n, h)
            sig = calibrate_sigma_hat(src, tau=0.2, h=h)
            groups.append(GridGroup(name, x[None, :], np.array([sig]),
                                    np.array([SIGMA_MIN])))
        return groups, n
    raise ValueError(case)


def organ_rate(organ, h, schedule, seed=0):
    idx = look_indices(schedule, organ["n"], seed=seed, h=h)
    ro = read_organ(organ, idx)
    return float(ro["phi"]) / (float(idx[-1]) * h)


# ---------------------------------------------------------------------------
# section 2: schedules
# ---------------------------------------------------------------------------

def _organ_tag(case, noisy, seed):
    return f"organ_{case}_{_regime_name(noisy)}_{seed}"


def run_rates():
    rows = []
    for case, life in CASES:
        for seed in SEEDS:
            groups, n = build_groups(case, seed, H_W, life)
            for noisy in (False, True):
                tag = _organ_tag(case, noisy, seed)
                got = _load(tag)
                if got is None:
                    t0 = time.time()
                    pm = Pacemaker(noisy=noisy, seed=seed)
                    organ = organ_stream(pm, groups, n, H_W)
                    nct = organ["nc"]
                    got = {
                        "case": case, "regime": _regime_name(noisy),
                        "seed": seed, "n": int(n), "nc": int(nct),
                        "phi_total": int(organ["phi"]),
                        "rows": [
                            {"schedule": sched,
                             "rate": organ_rate(organ, H_W, sched)}
                            for sched in SCHEDULES
                        ],
                        "seconds": time.time() - t0,
                    }
                    _save(tag, got)
                    print(f"  {tag}: {got['seconds']:.0f}s nc={nct} "
                          f"phi={got['phi_total']}", file=sys.stderr)
                else:
                    print(f"  {tag}: reuse", file=sys.stderr)
                for r in got["rows"]:
                    rows.append({"case": case, "regime": got["regime"],
                                 "seed": seed, "schedule": r["schedule"],
                                 "rate": r["rate"]})
    return _aggregate_rates(rows)


def _aggregate_rates(rows):
    agg = []
    for case, _life in CASES:
        for reg in ("clean", "noisy"):
            base = {r["seed"]: r["rate"] for r in rows
                    if r["case"] == case and r["regime"] == reg
                    and r["schedule"] == "50"}
            for sched in SCHEDULES:
                sel = [r for r in rows if r["case"] == case
                       and r["regime"] == reg and r["schedule"] == sched]
                if not sel:
                    continue
                vals = [r["rate"] for r in sel]
                errs = [abs(r["rate"] - base[r["seed"]]) / base[r["seed"]]
                        for r in sel]
                tol = TOL_CLEAN if reg == "clean" else TOL_NOISY
                agg.append({"case": case, "regime": reg, "schedule": sched,
                            "rate_mean": float(np.mean(vals)),
                            "err_mean": float(np.mean(errs)),
                            "err_max": float(np.max(errs)),
                            "tolerance": tol,
                            "pass": bool(np.max(errs) <= tol)})
    rate50 = []
    for case, _life in CASES:
        for reg in ("clean", "noisy"):
            sel = [r["rate"] for r in rows if r["case"] == case
                   and r["regime"] == reg and r["schedule"] == "50"]
            rate50.append({"case": case, "regime": reg,
                           "rate_50hz": float(np.mean(sel))})
    return {"rows": rows, "aggregate": agg, "rate_50hz": rate50}


# ---------------------------------------------------------------------------
# section 3: receptor vs chord tau_phi 0.2 at 50 Hz
# ---------------------------------------------------------------------------

def run_comparison(rates):
    with open(os.path.join(OUT1, "results.json")) as f:
        old = json.load(f)

    def chord50(case, reg):
        if case == "quiet":
            for x in old["quiet"]["tables"]["chord"][reg]:
                if x["schedule"] == "50":
                    return x["rate_mean"]
        if case == "quiet_noise":
            for x in old["quiet_noise"]["aggregate"]:
                if abs(x["sigma_n"] - SIGMA_N) < 1e-12 and x["mode"] == "chord" \
                        and x["regime"] == reg and x["schedule"] == "50":
                    return x["rate_mean"]
        if case == "active":
            for x in old["active"]["aggregate"]:
                if x["mode"] == "chord" and x["regime"] == reg \
                        and x["schedule"] == "50":
                    return x["rate_mean"]
        raise KeyError((case, reg))

    rows = []
    for r in rates["rate_50hz"]:
        c = chord50(r["case"], r["regime"])
        rows.append({"case": r["case"], "regime": r["regime"],
                     "receptor_50hz": r["rate_50hz"], "chord_50hz": c,
                     "ratio_receptor_over_chord": r["rate_50hz"] / c})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# section 4: grid convergence and cost
# ---------------------------------------------------------------------------

def run_convergence(seed=0):
    cases = (("active", 20000.0), ("quiet_noise", 73000.0))
    hs = (H_W, H_W / 2.0, H_W / 4.0)
    rows = []
    for case, life in cases:
        for h in hs:
            tag = f"conv_{case}_{_h_tag(h)}"
            got = _load(tag)
            if got is None:
                t0 = time.time()
                groups, n = build_groups(case, seed, h, life)
                rates = {}
                for noisy in (False, True):
                    pm = Pacemaker(noisy=noisy, seed=seed)
                    organ = organ_stream(pm, groups, n, h)
                    rates[_regime_name(noisy)] = organ_rate(organ, h, "50")
                    del organ
                got = {"case": case, "h": h, "n": int(n), "seed": seed,
                       "rates": rates, "seconds": time.time() - t0}
                _save(tag, got)
                print(f"  {tag}: {got['seconds']:.0f}s n={n} "
                      f"rates={rates}", file=sys.stderr)
            else:
                print(f"  {tag}: reuse", file=sys.stderr)
            rows.append(got)
    # relative change against the finest grid, same case and regime
    finest = {}
    for got in rows:
        if _h_tag(got["h"]) == _h_tag(H_W / 4.0):
            finest[got["case"]] = got
    table = []
    for key in ("active", "quiet_noise"):
        fine = finest[key]
        for got in rows:
            if got["case"] != key:
                continue
            for reg in ("clean", "noisy"):
                r = got["rates"][reg]
                rf = fine["rates"][reg]
                table.append({"h": got["h"], "h_tag": _h_tag(got["h"]),
                              "case": key, "regime": reg, "rate": r,
                              "rate_finest": rf,
                              "rel_change_vs_finest": abs(r - rf) / rf})
    return {"rows": rows, "table": table}


def run_cost(seed=0, life=20000.0):
    got = _load("cost")
    if got is not None:
        print("  cost: reuse", file=sys.stderr)
        return got
    case = "active"
    n = n_grid(life, H_W)
    t0 = time.time()
    groups, n2 = build_groups(case, seed, H_W, life)
    t_build = time.time() - t0
    pm = Pacemaker(noisy=False, seed=seed)
    t0 = time.time()
    organ = organ_stream(pm, groups, n, H_W)
    t_receptor_rule = time.time() - t0
    # CHORD at 50 Hz: same grid filters, sampled looks, then the look rule
    t0 = time.time()
    idx = look_indices("50", n2, seed=0, h=H_W)
    from step0a.experience_time import SensoryGroup
    look_groups = []
    for g in groups:
        xs = np.asarray(g.xs)[:, idx]
        look_groups.append(SensoryGroup(g.name, xs, g.sigma_hat, g.sigma_min,
                                        omega=g.omega))
    t_sample = time.time() - t0
    t0 = time.time()
    chord = simulate(pm, look_groups, idx, n2, mode="chord")
    t_chord_rule = time.time() - t0
    got = {
        "case": case, "life": life, "seed": seed,
        "build_seconds": t_build,
        "receptor_rule_seconds": t_receptor_rule,
        "receptor_total_seconds": t_build + t_receptor_rule,
        "sample_seconds": t_sample,
        "chord_rule_seconds": t_chord_rule,
        "chord_total_seconds": t_build + t_sample + t_chord_rule,
        "receptor_per_1000s": (t_build + t_receptor_rule) / (life / 1000.0),
        "chord_per_1000s": (t_build + t_sample + t_chord_rule) / (life / 1000.0),
        "receptor_phi": int(organ["phi"]),
        "chord_phi": float(chord["phi"]),
    }
    _save("cost", got)
    print(f"  cost: receptor {got['receptor_total_seconds']:.1f}s "
          f"chord {got['chord_total_seconds']:.1f}s", file=sys.stderr)
    return got


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def write_tables(path, res):
    with open(path, "w") as f:
        f.write("# revision 2 tables (RECEPTOR mode)\n\n")
        f.write("Rate = Phi per world second. err = |rate(schedule) - "
                "rate(50 Hz)| / rate(50 Hz).\n\n")
        f.write("## RECEPTOR rates (2)\n\n")
        f.write("| case | regime | schedule | rate (mean) | err mean | err max | PASS |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for x in res["rates"]["aggregate"]:
            f.write(f"| {x['case']} | {x['regime']} | {x['schedule']} | "
                    f"{_fmt(x['rate_mean'],5)} | {_fmt(x['err_mean'])} | "
                    f"{_fmt(x['err_max'])} | "
                    f"{'PASS' if x['pass'] else 'FAIL'} |\n")
        f.write("\n50 Hz rates by case/regime: "
                + ", ".join(f"{x['case']}/{x['regime']} {_fmt(x['rate_50hz'],5)}"
                            for x in res["rates"]["rate_50hz"]) + ".\n")

        f.write("\n## RECEPTOR vs CHORD tau_phi=0.2 at 50 Hz (3)\n\n")
        f.write("| case | regime | RECEPTOR 50 Hz | CHORD tau 0.2 50 Hz | ratio |\n")
        f.write("|---|---|---|---|---|\n")
        for x in res["comparison"]["rows"]:
            f.write(f"| {x['case']} | {x['regime']} | "
                    f"{_fmt(x['receptor_50hz'],5)} | {_fmt(x['chord_50hz'],5)} | "
                    f"{_fmt(x['ratio_receptor_over_chord'])} |\n")

        f.write("\n## Grid convergence (2)\n\n")
        f.write("Seed 0. rel. change against h_w/4 for the same case and regime.\n\n")
        f.write("| h | case | regime | rate at 50 Hz | rel change vs h_w/4 |\n")
        f.write("|---|---|---|---|---|\n")
        for x in res["convergence"]["table"]:
            f.write(f"| {x['h']:.10f} | {x['case']} | {x['regime']} | "
                    f"{_fmt(x['rate'],5)} | {_fmt(x['rel_change_vs_finest'],6)} |\n")

        f.write("\n## Cost on ACTIVE, 20000 s, 50 Hz (2)\n\n")
        c = res["cost"]
        f.write(f"- RECEPTOR total {c['receptor_total_seconds']:.2f} s "
                f"({c['receptor_per_1000s']:.3f} s per 1000 world s); "
                f"rule {c['receptor_rule_seconds']:.2f} s.\n")
        f.write(f"- CHORD total {c['chord_total_seconds']:.2f} s "
                f"({c['chord_per_1000s']:.3f} s per 1000 world s); "
                f"rule {c['chord_rule_seconds']:.2f} s.\n")
        f.write(f"- Common source build {c['build_seconds']:.2f} s; "
                f"CHORD sampling {c['sample_seconds']:.2f} s.\n")
        f.write(f"- Phi: RECEPTOR {c['receptor_phi']}, "
                f"ATH 50 Hz {c['chord_phi']}.\n".replace("ATH", "CHORD"))


def main():
    t0 = time.time()
    os.makedirs(OUT2, exist_ok=True)
    rates = _load("rates")
    if rates is None:
        print("section 2 rates...", file=sys.stderr)
        rates = run_rates()
        _save("rates", rates)
    else:
        print("section 2 rates: reuse", file=sys.stderr)
    comparison = _load("comparison")
    if comparison is None:
        print("section 3 comparison...", file=sys.stderr)
        comparison = run_comparison(rates)
        _save("comparison", comparison)
    convergence = _load("convergence")
    if convergence is None:
        print("section 4 convergence...", file=sys.stderr)
        convergence = run_convergence(seed=0)
        _save("convergence", convergence)
    cost = run_cost(seed=0)

    res = {"meta": {"design": "revision 2 (20 Nov 2026)",
                    "h_w": H_W, "seeds": list(SEEDS),
                    "schedules": list(SCHEDULES),
                    "constants": {"delta": DELTA, "Z_p": Z_P,
                                  "omega_p": OMEGA_P, "Z_ref": Z_REF,
                                  "sigma_hat_p": SIGMA_HAT_P,
                                  "sigma_min": SIGMA_MIN},
                    "tolerances": {"clean": TOL_CLEAN, "noisy": TOL_NOISY},
                    "total_seconds": None},
           "rates": rates, "comparison": comparison,
           "convergence": convergence, "cost": cost}
    res["meta"]["total_seconds"] = time.time() - t0
    with open(os.path.join(OUT2, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_tables(os.path.join(OUT2, "tables.md"), res)
    print(f"done in {res['meta']['total_seconds']:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
