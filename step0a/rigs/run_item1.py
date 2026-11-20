"""Section 3 measurements (Step 0A item 1 fixture readout).

Writes results.json and tables.md into the runs folder. Run from the
project root with PYTHONPATH=R, OMP_NUM_THREADS=1, taskset -c 0-7 nice -n 19.
"""

from __future__ import annotations

import json
import math
import os
import resource
import sys
import time

import numpy as np

from step0a.experience_time import (
    DELTA, SIGMA_HAT_P, SIGMA_MIN, Z_REF, Z_P, OMEGA_P, SensoryGroup, simulate,
)
from step0a.pacemaker import Pacemaker
from step0a.receptors import Receptor, calibrate_sigma_hat
from step0a.traces import LAMBDAS, run_traces
from step0a.world_signals import (
    H_W, Source, UtilitySource, irregular_look_indices, look_indices, n_grid,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "runs", "out")
SEEDS = (0, 1, 2)
SCHEDULES = ("2", "5", "15", "50", "irregular")
MODES = ("chord", "arc")
TOL_CLEAN = 0.05
TOL_NOISY = 0.10


def _rate(phi: float, idx: np.ndarray, h: float = H_W) -> float:
    return float(phi) / (float(idx[-1]) * h)


def _run(pm, groups, idx, n, mode, **kw):
    return simulate(pm, groups, idx, n, mode=mode, **kw)


def _rate_table(rows_by_seed, regime, mode, tolerance=None):
    """Build table rows (mean rate, mean/max relative error vs 50 Hz)."""
    out = []
    for sched in SCHEDULES:
        rates = [rows_by_seed[s][sched] for s in SEEDS]
        errs = []
        for s in SEEDS:
            r50 = rows_by_seed[s]["50"]
            errs.append(abs(rows_by_seed[s][sched] - r50) / r50)
        row = {
            "mode": mode,
            "regime": regime,
            "schedule": sched,
            "seed_rates": [float(x) for x in rates],
            "rate_mean": float(np.mean(rates)),
            "err_mean": float(np.mean(errs)),
            "err_max": float(np.max(errs)),
        }
        if tolerance is not None:
            row["tolerance"] = tolerance
            row["pass"] = bool(row["err_max"] <= tolerance)
        out.append(row)
    return out


def _silent_specs(ngroups):
    return [
        {"name": f"silent{i}", "xs": None, "sigma_hat": 0.0, "sigma_min": SIGMA_MIN}
        for i in range(ngroups)
    ]


def _groups_for_schedule(specs, idx, samples, si):
    groups = []
    for gi, sp in enumerate(specs):
        if sp["xs"] is None:
            xs = np.zeros(len(idx), dtype=np.float64)
        else:
            xs = samples[(si, gi)]
        groups.append(SensoryGroup(
            name=sp["name"], xs=np.asarray(xs)[None, :],
            sigma_hat=np.array([sp["sigma_hat"]]),
            sigma_min=np.array([sp["sigma_min"]]),
            omega=sp.get("omega", 1.0),
            copies_of=sp.get("copies_of"),
        ))
    return groups


# ---------------------------------------------------------------------------
# 3.1 QUIET
# ---------------------------------------------------------------------------

def run_quiet(life=73000.0):
    n = n_grid(life)
    schedules = [look_indices(s, n) for s in SCHEDULES]
    specs = _silent_specs(10)
    samples = ({},)  # unused for all-zero groups
    results = {}
    window_ok = {}
    for regime, noisy in (("clean", False), ("noisy", True)):
        by_mode = {m: {s: {} for s in SEEDS} for m in MODES}
        for s in SEEDS:
            pm = Pacemaker(noisy=noisy, seed=s)
            for si, sched in enumerate(SCHEDULES):
                idx = schedules[si]
                groups = _groups_for_schedule(specs, idx, samples, si)
                for mode in MODES:
                    out = _run(pm, groups, idx, n, mode, store_trajectory=True)
                    by_mode[mode][s][sched] = _rate(out["phi"], idx)
                    bucket = (idx * H_W // 100.0).astype(np.int64)
                    adv_all = np.bincount(bucket, weights=out["m_traj"])
                    n_full = int((idx[-1] * H_W) // 100.0)
                    adv = adv_all[:n_full]
                    key = f"{mode}|{regime}|{sched}|{s}"
                    window_ok[key] = bool(adv.size == 0 or np.all(adv > 0))
        for mode in MODES:
            results.setdefault(mode, {})[regime] = _rate_table(
                by_mode[mode], regime, mode,
                TOL_CLEAN if regime == "clean" else TOL_NOISY)
    return {"life": life, "tables": results, "window_ok": window_ok,
            "all_windows_advance": bool(all(window_ok.values()))}


# ---------------------------------------------------------------------------
# 3.2 QUIET with sensor noise
# ---------------------------------------------------------------------------

def run_quiet_noise(life=73000.0, sigmas=(0.05, 0.2, 1.0)):
    n = n_grid(life)
    schedules = [look_indices(s, n) for s in SCHEDULES]
    out = {"life": life, "sigmas": list(sigmas), "tables": []}
    for sigma_n in sigmas:
        sig_hat = calibrate_sigma_hat(Source("silent", seed=999, sigma_n=sigma_n))
        specs = [{"name": f"noise{g}", "xs": True, "sigma_hat": sig_hat,
                  "sigma_min": SIGMA_MIN} for g in range(10)]
        for s in SEEDS:
            samples = {}
            for g in range(10):
                src = Source("silent", seed=s * 100 + g, sigma_n=sigma_n)
                xf = Receptor(src, 0.2).filtered(n)
                for si, idx in enumerate(schedules):
                    samples[(si, g)] = xf[idx]
            for regime, noisy in (("clean", False), ("noisy", True)):
                pm = Pacemaker(noisy=noisy, seed=s)
                for si, sched in enumerate(SCHEDULES):
                    idx = schedules[si]
                    for mode in MODES:
                        groups = _groups_for_schedule(specs, idx, samples, si)
                        o = _run(pm, groups, idx, n, mode)
                        out["tables"].append({
                            "sigma_n": sigma_n, "regime": regime, "mode": mode,
                            "schedule": sched, "seed": s,
                            "rate": _rate(o["phi"], idx),
                        })
    # aggregate per (sigma, regime, mode, schedule)
    agg = []
    for sigma_n in sigmas:
        for regime in ("clean", "noisy"):
            for mode in MODES:
                for sched in SCHEDULES:
                    vals = [r["rate"] for r in out["tables"]
                            if r["sigma_n"] == sigma_n and r["regime"] == regime
                            and r["mode"] == mode and r["schedule"] == sched]
                    r50 = [r["rate"] for r in out["tables"]
                           if r["sigma_n"] == sigma_n and r["regime"] == regime
                           and r["mode"] == mode and r["schedule"] == "50"]
                    errs = [abs(v - w) / w for v, w in zip(vals, r50)]
                    agg.append({
                        "sigma_n": sigma_n, "regime": regime, "mode": mode,
                        "schedule": sched,
                        "rate_mean": float(np.mean(vals)),
                        "err_mean": float(np.mean(errs)),
                        "err_max": float(np.max(errs)),
                    })
    out["aggregate"] = agg
    return out


# ---------------------------------------------------------------------------
# ACTIVE helpers
# ---------------------------------------------------------------------------

def _active_specs(sensor_noise, seed, n, schedules, steps_copies=1, tau=0.2):
    specs = []
    srcs = []
    for name, kind in (("steps", "steps"), ("bursts", "bursts"), ("ramp", "ramp")):
        src = Source(kind, seed=seed, sigma_n=sensor_noise)
        sig_hat = calibrate_sigma_hat(src, tau=tau)
        specs.append({"name": name, "xs": True, "sigma_hat": sig_hat,
                      "sigma_min": SIGMA_MIN})
        srcs.append(src)
    for c in range(2, steps_copies + 1):
        specs.append({"name": f"steps_undeclared{c}", "xs": True,
                      "sigma_hat": specs[0]["sigma_hat"], "sigma_min": SIGMA_MIN})
        srcs.append(Source("steps", seed=seed, sigma_n=sensor_noise))
    samples = {}
    for g, src in enumerate(srcs):
        xf = Receptor(src, tau).filtered(n)
        for si, idx in enumerate(schedules):
            samples[(si, g)] = xf[idx]
    return specs, samples


def run_active(life=20000.0):
    n = n_grid(life)
    schedules = [look_indices(s, n) for s in SCHEDULES]
    out = {"life": life, "tables": [], "sensory_share": [],
           "basal_only": []}
    for sensor_noise in (0.0, 0.05):
        regime = "clean" if sensor_noise == 0.0 else "noisy"
        for s in SEEDS:
            specs, samples = _active_specs(sensor_noise, s, n, schedules)
            pm = Pacemaker(noisy=(sensor_noise > 0.0), seed=s)
            for si, sched in enumerate(SCHEDULES):
                idx = schedules[si]
                for mode in MODES:
                    groups = _groups_for_schedule(specs, idx, samples, si)
                    o = _run(pm, groups, idx, n, mode)
                    out["tables"].append({
                        "regime": regime, "mode": mode, "schedule": sched,
                        "seed": s, "rate": _rate(o["phi"], idx),
                    })
            # basal-only reference at 50 Hz for the sensory share
            si = 0
            idx = schedules[si]
            for mode in MODES:
                o = _run(pm, [], idx, n, mode)
                out["basal_only"].append({
                    "regime": regime, "mode": mode, "seed": s,
                    "rate": _rate(o["phi"], idx),
                })
    agg = []
    for regime in ("clean", "noisy"):
        for mode in MODES:
            for sched in SCHEDULES:
                vals = [r["rate"] for r in out["tables"]
                        if r["regime"] == regime and r["mode"] == mode
                        and r["schedule"] == sched]
                r50 = [r["rate"] for r in out["tables"]
                       if r["regime"] == regime and r["mode"] == mode
                       and r["schedule"] == "50"]
                errs = [abs(v - w) / w for v, w in zip(vals, r50)]
                agg.append({"regime": regime, "mode": mode, "schedule": sched,
                            "rate_mean": float(np.mean(vals)),
                            "err_mean": float(np.mean(errs)),
                            "err_max": float(np.max(errs)),
                            "tolerance": TOL_CLEAN if regime == "clean" else TOL_NOISY,
                            "pass": bool(np.max(errs) <=
                                         (TOL_CLEAN if regime == "clean" else TOL_NOISY))})
    out["aggregate"] = agg
    # sensory share at 50 Hz per seed
    for regime in ("clean", "noisy"):
        for mode in MODES:
            for s in SEEDS:
                r_all = [r["rate"] for r in out["tables"]
                         if r["regime"] == regime and r["mode"] == mode
                         and r["schedule"] == "50" and r["seed"] == s][0]
                r_bas = [r["rate"] for r in out["basal_only"]
                         if r["regime"] == regime and r["mode"] == mode
                         and r["seed"] == s][0]
                out["sensory_share"].append({
                    "regime": regime, "mode": mode, "seed": s,
                    "rate_all": r_all, "rate_basal": r_bas,
                    "share_sensory": (r_all - r_bas) / r_all,
                })
    return out


# ---------------------------------------------------------------------------
# 3.4 broken controls
# ---------------------------------------------------------------------------

def run_broken(life=73000.0):
    n = n_grid(life)
    idx = look_indices("50", n)
    res = {"grow_Z": [], "per_look": []}
    for s in SEEDS:
        pm = Pacemaker(noisy=False, seed=s)
        r = {}
        for ng in (1, 10):
            specs = _silent_specs(ng)
            groups = _groups_for_schedule(specs, idx, ({},), 0)
            o = _run(pm, groups, idx, n, "grow_Z")
            r[ng] = _rate(o["phi"], idx)
        res["grow_Z"].append({"seed": s, "rate_1": r[1], "rate_10": r[10],
                              "ratio_10_over_1": r[10] / r[1]})
    for s in SEEDS:
        pm = Pacemaker(noisy=False, seed=s)
        rates = {}
        for sched in ("2", "50"):
            idx2 = look_indices(sched, n)
            specs = _silent_specs(10)
            groups = _groups_for_schedule(specs, idx2, ({},), 0)
            o = _run(pm, groups, idx2, n, "per_look")
            rates[sched] = _rate(o["phi"], idx2)
        res["per_look"].append({
            "seed": s, "rate_2": rates["2"], "rate_50": rates["50"],
            "err_2_vs_50": abs(rates["2"] - rates["50"]) / rates["50"]})
    return res


# ---------------------------------------------------------------------------
# 3.5 undeclared duplicate / odometer
# ---------------------------------------------------------------------------

def run_duplicates(life=20000.0):
    n = n_grid(life)
    schedules = [look_indices("50", n)]
    out = {"rows": []}
    for s in SEEDS:
        pm = Pacemaker(noisy=False, seed=s)
        for copies in (1, 2, 4):
            specs, samples = _active_specs(0.0, s, n, schedules,
                                           steps_copies=copies)
            si = 0
            idx = schedules[si]
            for mode in MODES:
                groups = _groups_for_schedule(specs, idx, samples, si)
                o = _run(pm, groups, idx, n, mode)
                rate = _rate(o["phi"], idx)
                ob = _run(pm, [], idx, n, mode)
                rate_b = _rate(ob["phi"], idx)
                out["rows"].append({
                    "seed": s, "copies": copies, "mode": mode,
                    "rate": rate, "rate_basal": rate_b,
                    "share_sensory": (rate - rate_b) / rate,
                })
    # ratios vs the no-copy life, per seed and mode
    ratios = []
    for s in SEEDS:
        for mode in MODES:
            base = [r for r in out["rows"] if r["seed"] == s and r["mode"] == mode
                    and r["copies"] == 1][0]
            for copies in (2, 4):
                r = [x for x in out["rows"] if x["seed"] == s and x["mode"] == mode
                     and x["copies"] == copies][0]
                ratios.append({
                    "seed": s, "mode": mode, "copies": copies,
                    "rate_ratio": r["rate"] / base["rate"],
                    "share_change": r["share_sensory"] - base["share_sensory"],
                })
    out["ratios"] = ratios
    return out


# ---------------------------------------------------------------------------
# 3.6 traces
# ---------------------------------------------------------------------------

def _trace_life(events, life, schedule, seed=0):
    n = n_grid(life)
    idx = look_indices(schedule, n)
    pm = Pacemaker(noisy=False, seed=seed)
    src = Source("ramp", seed=seed)
    x = Receptor(src, 0.2).sample(idx, n)
    sig = calibrate_sigma_hat(src)
    g = SensoryGroup("ramp", x[None, :], np.array([sig]), np.array([SIGMA_MIN]))
    u = UtilitySource(events).grid(n)
    cp, cm = UtilitySource.cumulative_integrals(u)
    ip, im = UtilitySource.interval_integrals(cp, cm, idx)
    dth = Pacemaker.dtheta(pm.phases(n), idx)
    dp = Pacemaker.p_raw_increments(dth)
    o = simulate(pm, [g], idx, n, mode="chord", utility_ip=ip,
                 utility_im=im, dp_raw=dp, record_anchors=True)
    anchors = o["anchors"]
    x_anchor = x[anchors["look_index"]][:, None]
    u_anchor = u[anchors["grid_index"]]
    tr = run_traces(anchors, x_anchor, u_anchor)
    return {"n": n, "idx": idx, "anchors": anchors, "u": u, "traces": tr,
            "phi": o["phi"]}


def _anchor_at_or_after(times, t):
    pos = np.searchsorted(times, t, side="left")
    if pos >= len(times):
        return len(times) - 1
    return int(pos)


def run_trace_units(seed=0):
    out = {"cases": []}
    for D in (10.0, 200.0):
        life = D + 50.0
        res = _trace_life([(0.0, D, 0.1)], life, "2", seed)
        k = _anchor_at_or_after(res["anchors"]["time"], D)
        lam_pos = LAMBDAS.index(500.0)
        out["cases"].append({
            "D": D,
            "T_U_500": float(res["traces"]["T_U_history"][k, lam_pos]),
            "T_U_broken_500": float(res["traces"]["T_U_broken_history"][k, lam_pos]),
            "T_x_500": float(res["traces"]["T_x_history"][k, lam_pos, 0]),
            "anchor_time": float(res["anchors"]["time"][k]),
        })
    a, b = out["cases"]
    out["ratio_extensive"] = b["T_U_500"] / a["T_U_500"]
    out["ratio_broken"] = b["T_U_broken_500"] / a["T_U_broken_500"]
    out["ratio_Tx"] = b["T_x_500"] / a["T_x_500"] if a["T_x_500"] != 0 else None
    return out


def _shock_case(quiet_len, schedule, seed=0, S=5.0, dur=0.1, after=50.0,
                t0=None):
    # Default placement: 0.03 s after the end of the quiet stretch so that the
    # shock starts strictly between 2 Hz looks.  run_trace_shock overrides this
    # with a placement that contains one 50 Hz anchor look (see there).
    if t0 is None:
        t0 = quiet_len + 0.03
    t1 = t0 + dur
    life = t1 + after
    res = _trace_life([(t0, t1, -S)], life, schedule, seed)
    times = res["anchors"]["time"]
    k_post = _anchor_at_or_after(times, t1 + 1.0)
    # anchors whose coalesced buffer can contain part of the shock
    cand = np.where((times > t0) & (times < t1 + 0.5))[0]
    if len(cand):
        eps_hist = res["traces"]["eps_history"]
        k_took = int(cand[np.argmax(np.max(eps_hist[cand], axis=1))])
    else:
        k_took = _anchor_at_or_after(times, t0 + 0.01)
    return {
        "quiet_len": quiet_len, "schedule": schedule,
        "anchor_time_post": float(times[k_post]),
        "T_U": [float(v) for v in res["traces"]["T_U_history"][k_post]],
        "T_U_broken": [float(v) for v in res["traces"]["T_U_broken_history"][k_post]],
        "eps": [float(v) for v in res["traces"]["eps_history"][k_post]],
        "anchor_time_took": float(times[k_took]),
        "max_eps_shock_anchor": float(np.max(res["traces"]["eps_history"][k_took])),
        "eps_shock_anchor": [float(v) for v in res["traces"]["eps_history"][k_took]],
        "anchors_in_shock": int(np.sum((times >= t0) & (times < t1))),
    }


def run_trace_shock():
    out = {"by_quiet": {}, "ratios": [], "post_shock_eps": [],
           "shock_start": {}}
    for quiet_len in (1000.0, 10000.0):
        # Place the shock so that one 50 Hz anchor look sits inside it, and the
        # whole 0.1 s shock fits between two 2 Hz looks.  This makes the
        # broken variant's sampling dependence visible, as the design expects;
        # the extensive trace is unaffected by the placement.
        probe = _trace_life([], quiet_len + 1.0, "50", 0)
        cand = [float(a) for a in probe["anchors"]["time"]
                if quiet_len + 0.05 < a < quiet_len + 0.45]
        t0 = (cand[0] - 0.05) if cand else (quiet_len + 0.03)
        out["shock_start"][str(int(quiet_len))] = t0
        rows = []
        for sched in SCHEDULES:
            rows.append(_shock_case(quiet_len, sched, t0=t0))
        out["by_quiet"][str(int(quiet_len))] = rows
        for r in rows:
            out["post_shock_eps"].append({
                "quiet_len": quiet_len, "schedule": r["schedule"],
                "max_eps_shock_anchor": r["max_eps_shock_anchor"],
                "eps_shock_anchor": r["eps_shock_anchor"],
            })
        ref = [r for r in rows if r["schedule"] == "50"][0]
        for r in rows:
            out.setdefault("rate_error_vs_50", []).append({
                "quiet_len": quiet_len, "schedule": r["schedule"],
                "err_T_U": [abs(a - b) / abs(b) if b != 0 else None
                            for a, b in zip(r["T_U"], ref["T_U"])],
                "err_T_U_broken": [abs(a - b) / abs(b) if b != 0 else None
                                   for a, b in zip(r["T_U_broken"], ref["T_U_broken"])],
            })
    # quiet-length ratios at the corresponding schedule
    r1 = {r["schedule"]: r for r in out["by_quiet"]["1000"]}
    r10 = {r["schedule"]: r for r in out["by_quiet"]["10000"]}
    for sched in SCHEDULES:
        a, b = r1[sched], r10[sched]
        out["ratios"].append({
            "schedule": sched,
            "ratio_T_U": [b["T_U"][i] / a["T_U"][i] if a["T_U"][i] != 0 else None
                          for i in range(len(LAMBDAS))],
            "ratio_T_U_broken": [b["T_U_broken"][i] / a["T_U_broken"][i]
                                 if a["T_U_broken"][i] != 0 else None
                                 for i in range(len(LAMBDAS))],
        })
    return out


# ---------------------------------------------------------------------------
# tau_phi sweep (an earlier revision section 2)
# ---------------------------------------------------------------------------

TAU_PHI_VALUES = (0.2, 0.5, 1.0, 2.0, 5.0)


def _tau_tag(tau):
    return str(tau).replace(".", "p")


def _rows_from_existing_02():
    """tau_phi = 0.2 rows are the already-finished runs of sections 3.2/3.3."""
    rows = []
    pa = _load_part("active")
    if pa is None:
        raise RuntimeError("part_active.json is required for the tau=0.2 rows")
    for r in pa["tables"]:
        rows.append({"tau": 0.2, "case": "active", "regime": r["regime"],
                     "mode": r["mode"], "schedule": r["schedule"],
                     "seed": r["seed"], "rate": r["rate"]})
    pq = _load_part("quiet_noise")
    if pq is None:
        raise RuntimeError("part_quiet_noise.json is required for tau=0.2")
    for r in pq["tables"]:
        if abs(r["sigma_n"] - 0.2) < 1e-12:
            rows.append({"tau": 0.2, "case": "quiet_noise",
                         "regime": r["regime"], "mode": r["mode"],
                         "schedule": r["schedule"], "seed": r["seed"],
                         "rate": r["rate"]})
    return rows


def _tau_rows(tau, active_life=20000.0, quiet_life=73000.0):
    """One tau_phi: ACTIVE and QUIET+OU(sigma_n=0.2)."""
    rows = []
    # ---- ACTIVE
    n = n_grid(active_life)
    schedules = [look_indices(s, n) for s in SCHEDULES]
    for regime, sensor_noise in (("clean", 0.0), ("noisy", 0.05)):
        for s in SEEDS:
            specs, samples = _active_specs(sensor_noise, s, n, schedules,
                                           tau=tau)
            pm = Pacemaker(noisy=(sensor_noise > 0.0), seed=s)
            for si, sched in enumerate(SCHEDULES):
                idx = schedules[si]
                for mode in MODES:
                    groups = _groups_for_schedule(specs, idx, samples, si)
                    o = _run(pm, groups, idx, n, mode)
                    rows.append({"tau": tau, "case": "active", "regime": regime,
                                 "mode": mode, "schedule": sched, "seed": s,
                                 "rate": _rate(o["phi"], idx)})
    # ---- QUIET with sensor noise, sigma_n = 0.2
    sigma_n = 0.2
    n2 = n_grid(quiet_life)
    sch2 = [look_indices(s, n2) for s in SCHEDULES]
    sig_hat = calibrate_sigma_hat(Source("silent", seed=999, sigma_n=sigma_n),
                                  tau=tau)
    specs = [{"name": f"noise{g}", "xs": True, "sigma_hat": sig_hat,
              "sigma_min": SIGMA_MIN} for g in range(10)]
    for s in SEEDS:
        samples = {}
        for g in range(10):
            src = Source("silent", seed=s * 100 + g, sigma_n=sigma_n)
            xf = Receptor(src, tau).filtered(n2)
            for si, idx in enumerate(sch2):
                samples[(si, g)] = xf[idx]
        for regime, noisy in (("clean", False), ("noisy", True)):
            pm = Pacemaker(noisy=noisy, seed=s)
            for si, sched in enumerate(SCHEDULES):
                idx = sch2[si]
                for mode in MODES:
                    groups = _groups_for_schedule(specs, idx, samples, si)
                    o = _run(pm, groups, idx, n2, mode)
                    rows.append({"tau": tau, "case": "quiet_noise",
                                 "regime": regime, "mode": mode,
                                 "schedule": sched, "seed": s,
                                 "rate": _rate(o["phi"], idx)})
    return rows


def _aggregate_tau(rows):
    agg = []
    key_seen = set()
    for r in rows:
        key_seen.add((r["tau"], r["case"], r["regime"], r["mode"]))
    for tau in sorted({k[0] for k in key_seen}):
        for case in ("active", "quiet_noise"):
            for regime in ("clean", "noisy"):
                for mode in MODES:
                    if (tau, case, regime, mode) not in key_seen:
                        continue
                    base = {(r["seed"]): r["rate"] for r in rows
                            if r["tau"] == tau and r["case"] == case
                            and r["regime"] == regime and r["mode"] == mode
                            and r["schedule"] == "50"}
                    for sched in SCHEDULES:
                        sel = [r for r in rows
                               if r["tau"] == tau and r["case"] == case
                               and r["regime"] == regime and r["mode"] == mode
                               and r["schedule"] == sched]
                        vals = [r["rate"] for r in sel]
                        errs = [abs(r["rate"] - base[r["seed"]]) / base[r["seed"]]
                                for r in sel]
                        tol = TOL_CLEAN if regime == "clean" else TOL_NOISY
                        agg.append({
                            "tau": tau, "case": case, "regime": regime,
                            "mode": mode, "schedule": sched,
                            "rate_mean": float(np.mean(vals)),
                            "err_mean": float(np.mean(errs)),
                            "err_max": float(np.max(errs)),
                            "tolerance": tol,
                            "pass": bool(np.max(errs) <= tol),
                        })
    return agg


def run_tau_sweep(taus=TAU_PHI_VALUES):
    rows = []
    for tau in taus:
        tag = _tau_tag(tau)
        if tau == 0.2:
            got = _rows_from_existing_02()
        else:
            got = _load_part(f"tau_{tag}")
            if got is None:
                print(f"  tau_phi={tau}: running...", file=sys.stderr)
                t0 = time.time()
                got = _tau_rows(tau)
                _save_part(f"tau_{tag}", got)
                print(f"  tau_phi={tau} done in {time.time() - t0:.0f}s",
                      file=sys.stderr)
            else:
                print(f"  tau_phi={tau}: reusing part", file=sys.stderr)
        rows.extend(got)
    agg = _aggregate_tau(rows)
    rate50 = []
    for r in agg:
        if r["schedule"] != "50":
            continue
        rate50.append({"tau": r["tau"], "case": r["case"],
                       "regime": r["regime"], "mode": r["mode"],
                       "rate_50hz": r["rate_mean"]})
    return {"taus": list(taus), "rows": rows, "aggregate": agg,
            "rate_50hz": rate50}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _part_path(name):
    return os.path.join(OUT, f"part_{name}.json")


def _save_part(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(_part_path(name), "w") as f:
        json.dump(obj, f)


def _load_part(name):
    p = _part_path(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _section(name, fn):
    got = _load_part(name)
    if got is not None:
        print(f"  (reusing part {name})", file=sys.stderr)
        return got
    t0 = time.time()
    got = fn()
    _save_part(name, got)
    print(f"  {name} done in {time.time() - t0:.0f}s", file=sys.stderr)
    return got


def main():
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)

    def _runtime():
        print("measuring largest case (QUIET 73000 s, 50 Hz)...", file=sys.stderr)
        n = n_grid(73000.0)
        idx = look_indices("50", n)
        pm = Pacemaker(noisy=True, seed=0)
        specs = _silent_specs(10)
        groups = _groups_for_schedule(specs, idx, ({},), 0)
        t0 = time.time()
        _run(pm, groups, idx, n, "chord")
        t_chord = time.time() - t0
        t0 = time.time()
        _run(pm, groups, idx, n, "arc")
        t_arc = time.time() - t0
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        out = {"largest_case":
               "QUIET 73000 s, 50 Hz, 10 silent groups, noisy pacemaker",
               "n_grid": int(n), "chord_seconds": t_chord,
               "arc_seconds": t_arc, "peak_memory_mb": peak_mb,
               "load_average_at_run": os.getloadavg()}
        print(f"  largest case chord {t_chord:.1f}s arc {t_arc:.1f}s "
              f"peak {peak_mb:.0f} MB", file=sys.stderr)
        return out

    runtime = _section("runtime", _runtime)
    quiet = _section("quiet", lambda: run_quiet())
    quiet_noise = _section("quiet_noise", lambda: run_quiet_noise())
    active = _section("active", lambda: run_active())
    broken = _section("broken", lambda: run_broken())
    duplicates = _section("duplicates", lambda: run_duplicates())
    trace_units = _section("traces_units", lambda: run_trace_units())
    trace_shock = _section("traces_shock", lambda: run_trace_shock())
    tau_sweep = _section("tau_sweep", lambda: run_tau_sweep())

    results = {
        "meta": {
            "design": "section 3 (20 Nov 2026)",
            "h_w": H_W,
            "seeds": list(SEEDS),
            "schedules": list(SCHEDULES),
            "modes": list(MODES),
            "constants": {
                "delta": DELTA, "Z_p": Z_P, "omega_p": OMEGA_P,
                "Z_ref": Z_REF, "omega_g": 1.0, "sigma_hat_p": SIGMA_HAT_P,
                "sigma_min": SIGMA_MIN,
            },
            "tolerances": {"clean": TOL_CLEAN, "noisy": TOL_NOISY},
            "total_seconds": time.time() - t_start,
        },
        "runtime": runtime,
        "quiet": quiet,
        "quiet_noise": quiet_noise,
        "active": active,
        "broken": broken,
        "duplicates": duplicates,
        "traces_units": trace_units,
        "traces_shock": trace_shock,
        "tau_sweep": tau_sweep,
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        write_tables(f, results)
    print(f"done in {results['meta']['total_seconds']:.0f}s", file=sys.stderr)


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def write_tables(f, r):
    f.write("# fixture readout tables\n\n")
    f.write("All rates are Phi per world second over the covered life. "
            "err = |rate(schedule) - rate(50 Hz)| / rate(50 Hz).\n\n")
    f.write("## Runtime and memory (3.7)\n\n")
    rt = r["runtime"]
    f.write(f"Largest case: {rt['largest_case']}; n_grid = {rt['n_grid']}; "
            f"CHORD {rt['chord_seconds']:.2f} s, ARC {rt['arc_seconds']:.2f} s; "
            f"peak RSS {rt['peak_memory_mb']:.0f} MB.\n\n")

    f.write("## QUIET, 73000 s, 10 silent groups (3.1)\n\n")
    f.write("| mode | regime | schedule | rate (mean) | err mean | err max | PASS |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for mode in MODES:
        for regime in ("clean", "noisy"):
            for row in r["quiet"]["tables"][mode][regime]:
                f.write(f"| {mode} | {regime} | {row['schedule']} | "
                        f"{_fmt(row['rate_mean'],6)} | {_fmt(row['err_mean'])} | "
                        f"{_fmt(row['err_max'])} | {'PASS' if row['pass'] else 'FAIL'} |\n")
    f.write(f"\nPositive Phi in every 100 s window at every schedule: "
            f"{r['quiet']['all_windows_advance']}.\n\n")

    f.write("## QUIET with sensor noise (3.2)\n\n")
    f.write("| sigma_n | regime | mode | schedule | rate (mean) | err mean | err max |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for row in r["quiet_noise"]["aggregate"]:
        f.write(f"| {row['sigma_n']} | {row['regime']} | {row['mode']} | "
                f"{row['schedule']} | {_fmt(row['rate_mean'],5)} | "
                f"{_fmt(row['err_mean'])} | {_fmt(row['err_max'])} |\n")

    f.write("\n## ACTIVE, 20000 s (3.3)\n\n")
    f.write("| regime | mode | schedule | rate (mean) | err mean | err max | PASS |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for row in r["active"]["aggregate"]:
        f.write(f"| {row['regime']} | {row['mode']} | {row['schedule']} | "
                f"{_fmt(row['rate_mean'],5)} | {_fmt(row['err_mean'])} | "
                f"{_fmt(row['err_max'])} | {'PASS' if row['pass'] else 'FAIL'} |\n")
    f.write("\nSensory share of Phi at 50 Hz:\n\n")
    f.write("| regime | mode | seed | rate all | rate basal | share sensory |\n")
    f.write("|---|---|---|---|---|---|\n")
    for row in r["active"]["sensory_share"]:
        f.write(f"| {row['regime']} | {row['mode']} | {row['seed']} | "
                f"{_fmt(row['rate_all'],5)} | {_fmt(row['rate_basal'],5)} | "
                f"{_fmt(row['share_sensory'])} |\n")

    f.write("\n## Broken controls (3.4)\n\n")
    f.write("| seed | grow_Z rate 1 group | grow_Z rate 10 groups | ratio 10/1 |\n")
    f.write("|---|---|---|---|\n")
    for row in r["broken"]["grow_Z"]:
        f.write(f"| {row['seed']} | {_fmt(row['rate_1'],5)} | {_fmt(row['rate_10'],5)} | "
                f"{_fmt(row['ratio_10_over_1'])} |\n")
    f.write("\n| seed | per_look rate 2 Hz | rate 50 Hz | err 2 vs 50 |\n")
    f.write("|---|---|---|---|\n")
    for row in r["broken"]["per_look"]:
        f.write(f"| {row['seed']} | {_fmt(row['rate_2'],5)} | {_fmt(row['rate_50'],5)} | "
                f"{_fmt(row['err_2_vs_50'])} |\n")

    f.write("\n## Undeclared duplicate / odometer (3.5)\n\n")
    f.write("| seed | mode | copies | rate | rate basal | share sensory |\n")
    f.write("|---|---|---|---|---|---|\n")
    for row in r["duplicates"]["rows"]:
        f.write(f"| {row['seed']} | {row['mode']} | {row['copies']} | "
                f"{_fmt(row['rate'],5)} | {_fmt(row['rate_basal'],5)} | "
                f"{_fmt(row['share_sensory'])} |\n")
    f.write("\n| seed | mode | copies | rate ratio vs 1 | share change |\n")
    f.write("|---|---|---|---|---|\n")
    for row in r["duplicates"]["ratios"]:
        f.write(f"| {row['seed']} | {row['mode']} | {row['copies']} | "
                f"{_fmt(row['rate_ratio'])} | {_fmt(row['share_change'])} |\n")

    f.write("\n## Trace units, u=+0.1 for D then quiet (3.6.1)\n\n")
    for c in r["traces_units"]["cases"]:
        f.write(f"- D={c['D']}: T_U(500)={c['T_U_500']:.6f}, "
                f"T_U_broken(500)={c['T_U_broken_500']:.6f}, "
                f"T_x(500)={c['T_x_500']:.6f}, anchor at {c['anchor_time']:.3f} s\n")
    f.write(f"\nRatio extensive T_U(200)/T_U(10): {_fmt(r['traces_units']['ratio_extensive'],3)}; "
            f"broken: {_fmt(r['traces_units']['ratio_broken'],3)}; "
            f"T_x: {_fmt(r['traces_units']['ratio_Tx'],3)}.\n")

    f.write("\n## Trace shock (3.6.2)\n\n")
    for quiet in ("1000", "10000"):
        f.write(f"### {quiet} s quiet before the shock\n\n")
        f.write("| schedule | post-shock anchor (s) | T_U (5/50/500) | T_U_broken | max eps shock | anchors in shock |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in r["traces_shock"]["by_quiet"][quiet]:
            tu = "/".join(f"{v:.4f}" for v in row["T_U"])
            tb = "/".join(f"{v:.4f}" for v in row["T_U_broken"])
            f.write(f"| {row['schedule']} | {row['anchor_time_post']:.3f} | {tu} | "
                    f"{tb} | {row['max_eps_shock_anchor']:.4f} | "
                    f"{row['anchors_in_shock']} |\n")
        f.write("\n")
    f.write("Quiet-length ratio T_U(10000 s)/T_U(1000 s):\n\n")
    f.write("| schedule | ratio T_U (5/50/500) | ratio broken |\n|---|---|---|\n")
    for row in r["traces_shock"]["ratios"]:
        a = "/".join(_fmt(v,4) for v in row["ratio_T_U"])
        b = "/".join(_fmt(v,4) for v in row["ratio_T_U_broken"])
        f.write(f"| {row['schedule']} | {a} | {b} |\n")
    f.write("\nRate error of T_U vs 50 Hz:\n\n")
    f.write("| quiet | schedule | err T_U (5/50/500) | err broken |\n")
    f.write("|---|---|---|---|\n")
    for row in r["traces_shock"]["rate_error_vs_50"]:
        a = "/".join(_fmt(v,4) for v in row["err_T_U"])
        b = "/".join(_fmt(v,4) for v in row["err_T_U_broken"])
        f.write(f"| {int(row['quiet_len'])} | {row['schedule']} | {a} | {b} |\n")

    if "tau_sweep" in r:
        f.write("\n## tau_phi sweep\n\n")
        f.write("The receptor copy feeding experience time has time constant "
                "tau_phi; sigma_hat is calibrated on that same copy. "
                "Tolerances: 5 % clean, 10 % noisy.\n\n")
        f.write("| tau_phi | case | regime | mode | schedule | rate (mean) | "
                "err mean | err max | PASS |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for row in r["tau_sweep"]["aggregate"]:
            f.write(f"| {row['tau']} | {row['case']} | {row['regime']} | "
                    f"{row['mode']} | {row['schedule']} | "
                    f"{_fmt(row['rate_mean'],5)} | {_fmt(row['err_mean'])} | "
                    f"{_fmt(row['err_max'])} | "
                    f"{'PASS' if row['pass'] else 'FAIL'} |\n")
        f.write("\n50 Hz rate (Phi per world second) by tau_phi:\n\n")
        f.write("| tau_phi | case | regime | mode | rate at 50 Hz |\n")
        f.write("|---|---|---|---|---|\n")
        for row in r["tau_sweep"]["rate_50hz"]:
            f.write(f"| {row['tau']} | {row['case']} | {row['regime']} | "
                    f"{row['mode']} | {_fmt(row['rate_50hz'],5)} |\n")


if __name__ == "__main__":
    main()
