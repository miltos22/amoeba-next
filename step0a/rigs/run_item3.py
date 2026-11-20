"""Step 0A item 3 pre-check: far-value readout (revision 6).

Outputs out6/results.json and out6/tables.md.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from step0a.far_value import (
    Stream, leverage, ubar_dict, v_b1, v_b1_fast, v_unp,
)
from step0a.pacemaker import PERIODS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT6 = os.path.join(ROOT, "runs", "out6")
LIFE = 20000.0
DT = 0.01
L = 50.0
CYC = sum(1.0 / p for p in PERIODS)
T975_19 = 2.093024054408263
SEEDS = list(range(20))
LAMBDAS = None


def _part(name):
    return os.path.join(OUT6, f"part_{name}.json")


def _save(name, obj):
    os.makedirs(OUT6, exist_ok=True)
    with open(_part(name), "w") as f:
        json.dump(obj, f)


def _load(name):
    p = _part(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# H5 world and record stream
# ---------------------------------------------------------------------------

def _stretches(rng, variant, life):
    out = []
    t = 0.0
    while t < life - 1e-12:
        dur = float(rng.uniform(40.0, 80.0))
        dur = min(dur, life - t)
        if variant in ("V1", "V2"):
            typ = "busy" if rng.random() < 0.5 else "quiet"
        else:
            u = float(rng.random())
            typ = "busy" if u < 0.45 else ("quiet" if u < 0.90 else "harm")
        out.append((t, dur, typ))
        t += dur
    return out


def build_h5(seed, variant, S=0.0, life=LIFE):
    vid = {"V1": 0, "V2": 1, "V3": 2}[variant]
    rng = np.random.default_rng(seed * 10007 + vid)
    stretches = _stretches(rng, variant, life)
    rngm = np.random.default_rng(seed * 10007 + 777 + vid)
    meals = []
    tm = 0.0
    while True:
        tm += float(rngm.exponential(300.0))
        if tm >= life:
            break
        meals.append(tm)
    n = int(round(life / DT)) + 1
    tg = np.arange(n, dtype=np.float64) * DT
    decline = np.empty(n, dtype=np.float64)
    decline[:] = 0.001
    for t0, dur, typ in stretches:
        i0 = int(round(t0 / DT))
        i1 = min(n, int(round((t0 + dur) / DT)))
        if i1 > i0:
            decline[i0:i1] = (0.0015 if (variant == "V2" and typ == "busy")
                              else 0.001)
    meal_idx = sorted({int(round(tm / DT)) for tm in meals
                       if 0 <= int(round(tm / DT)) < n})
    starts = sorted({int(round(t0 / DT)) for t0, _, _ in stretches})
    events = sorted(set([0] + starts + meal_idx + [n]))
    E = np.empty(n, dtype=np.float64)
    E[0] = 0.8
    meal_set = set(meal_idx)
    for a, b in zip(events[:-1], events[1:]):
        if a >= n:
            break
        dr = decline[min(a, n - 1)]
        base = E[a]
        E[a:b] = np.maximum(0.0, base - dr * DT * np.arange(b - a))
        if b >= n:
            break
        E_prev = E[b - 1]
        if b in meal_set:
            E_prev = min(1.0, E_prev + 0.3)
        E[b] = E_prev
    u = -(np.maximum(0.0, 1.0 - E)) ** 2
    if variant == "V3" and S > 0:
        for t0, dur, typ in stretches:
            if typ != "harm":
                continue
            i0 = int(np.ceil((t0 + 30.0) / DT))
            i1 = int(np.ceil((t0 + 31.0) / DT))
            i0 = max(0, i0)
            i1 = min(n, i1)
            if i1 > i0:
                u[i0:i1] += -S
    # record closing times
    ts = []
    phi = 0.0
    for t0, dur, typ in stretches:
        t1 = t0 + dur
        rho = 20.0 if typ == "busy" else 5.0
        k0 = int(np.floor(phi)) + 1
        k1 = int(np.floor(phi + rho * dur + 1e-12))
        if k1 >= k0:
            kk = np.arange(k0, k1 + 1, dtype=np.int64)
            tt = t0 + (kk - phi) / rho
            tt = tt[(tt <= t1 + 1e-9) & (tt <= life + 1e-9)]
            if len(tt):
                ts.append(tt)
        phi += rho * dur
    ts = np.concatenate(ts) if ts else np.empty(0)
    M = len(ts)
    t_end = ts
    t_start = np.concatenate(([0.0], ts[:-1]))
    cum = np.concatenate(([0.0], np.cumsum(u) * DT))
    cump = np.concatenate(([0.0], np.cumsum(np.maximum(u, 0.0)) * DT))
    cumm = np.concatenate(([0.0], np.cumsum(np.maximum(-u, 0.0)) * DT))
    I_up = np.interp(t_end, tg, cump[:-1]) - np.interp(t_start, tg, cump[:-1])
    I_um = np.interp(t_end, tg, cumm[:-1]) - np.interp(t_start, tg, cumm[:-1])
    U = I_up - I_um
    E_start = np.interp(t_start, tg, E)
    dP = CYC * (t_end - t_start)
    stream = Stream(U, dP, E_start, t_start, t_end)
    onsets = {"busy": [], "quiet": [], "harm": []}
    for t0, dur, typ in stretches:
        p = int(np.searchsorted(t_end, t0, side="left"))
        if p < M:
            onsets[typ].append(p)
    for k in onsets:
        onsets[k] = np.array(sorted(set(onsets[k])), dtype=np.int64)
    return stream, onsets, np.asarray(meals), stretches


# ---------------------------------------------------------------------------
# H5 readout helpers
# ---------------------------------------------------------------------------

def _pairs(positions):
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0:
        return (positions, np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))
    anchors = np.repeat(positions, 14)
    ks = np.tile(np.arange(1, 15, dtype=np.int64), len(positions))
    weights = np.ones(len(anchors))
    return anchors, ks, weights


def _group_value(stream, positions, readout, mode, ub, kband=None, n0=1.0):
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0:
        return 0.0
    if kband is None:
        anchors, ks, w = _pairs(positions)
        if readout == "unp":
            total, _ = v_unp(stream, anchors, ks, w, n0, ub, mode)
            return total
        return v_b1_fast(stream, anchors, ks, w, n0, ub, mode)
    anchors = positions
    ks = np.full(len(positions), kband, dtype=np.int64)
    w = np.ones(len(positions))
    if readout == "unp":
        _, per = v_unp(stream, anchors, ks, w, n0, ub, mode)
        return per.get(kband, 0.0)
    return v_b1(stream, anchors, ks, w, n0, ub, mode, kband=kband)


def _contrast(stream, pos1, pos2, readout, mode, ub, kband=None, n0=1.0):
    return (_group_value(stream, pos1, readout, mode, ub, kband, n0)
            - _group_value(stream, pos2, readout, mode, ub, kband, n0))


def _ub_all(stream, positions, mode):
    if mode == "none" or len(positions) == 0:
        return {}
    anchors, ks, _ = _pairs(positions)
    return ubar_dict(stream, anchors, ks, mode)


def h5_life(seed, variant, S):
    stream, onsets, meals, stretches = build_h5(seed, variant, S)
    groups = {"busy": onsets["busy"], "quiet": onsets["quiet"],
              "harm": onsets["harm"]}
    if variant == "V3":
        g1, g2 = groups["harm"], groups["quiet"]
    else:
        g1, g2 = groups["busy"], groups["quiet"]
    all_pos = np.unique(np.concatenate([groups["busy"], groups["quiet"],
                                        groups["harm"]]))
    ub = {"phys": _ub_all(stream, all_pos, "phys"),
          "phi": _ub_all(stream, all_pos, "phi"),
          "none": {}}
    per_life = {}
    for base in ("phys", "phi", "none"):
        mode = base
        for readout in ("b1", "unp"):
            for k in range(1, 15):
                per_life[(base, readout, k)] = _contrast(
                    stream, g1, g2, readout, mode, ub[base], kband=k)
            per_life[(base, readout, 0)] = _contrast(
                stream, g1, g2, readout, mode, ub[base], kband=None)
    # relabel null, 200 relabels, total and every band
    rng = np.random.default_rng(seed * 31 + 7)
    union = np.concatenate([g1, g2])
    n1 = len(g1)
    null = {(b, r, k): []
            for b in ("phys", "phi", "none")
            for r in ("b1", "unp")
            for k in list(range(1, 15)) + [0]}
    if len(union) >= 2 and n1 > 0:
        for _ in range(200):
            perm = rng.permutation(union)
            a1, a2 = perm[:n1], perm[n1:]
            for base in ("phys", "phi", "none"):
                mode = base
                for readout in ("b1", "unp"):
                    null[(base, readout, 0)].append(_contrast(
                        stream, a1, a2, readout, mode, ub[base]))
                    for k in range(1, 15):
                        null[(base, readout, k)].append(_contrast(
                            stream, a1, a2, readout, mode, ub[base], kband=k))
    null_sd = {}
    for key, vals in null.items():
        null_sd[key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return per_life, null_sd, stream, groups


def run_h5():
    variants = [("V1", 0.0), ("V2", 0.0),
                ("V3", 0.25), ("V3", 1.0), ("V3", 4.0)]
    rows = []
    v1_frac = []
    for variant, S in variants:
        tag = f"h5_{variant}_{str(S).replace('.', 'p')}"
        got = _load(tag)
        if got is None:
            print(f"  H5 {variant} S={S}...", file=sys.stderr)
            t0 = time.time()
            per = {}
            nulls = {}
            for seed in SEEDS:
                pl, ns, _st, _gr = h5_life(seed, variant, S)
                for key, val in pl.items():
                    per.setdefault(key, []).append(float(val))
                nulls[seed] = ns
            label = f"V3_S{S}" if variant == "V3" else variant
            new_rows = []
            for base in ("phys", "phi", "none"):
                for readout in ("b1", "unp"):
                    for k in list(range(1, 15)) + [0]:
                        vals = np.array(per[(base, readout, k)])
                        sd = float(np.std(vals, ddof=1))
                        new_rows.append({
                            "variant": label, "baseline": base,
                            "readout": readout, "band": k,
                            "mean": float(np.mean(vals)), "sd": sd,
                            "ci_lo": float(np.mean(vals) - T975_19 * sd / np.sqrt(len(vals))),
                            "ci_hi": float(np.mean(vals) + T975_19 * sd / np.sqrt(len(vals))),
                            "null_sd": float(np.mean([nulls[s][(base, readout, k)]
                                                      for s in SEEDS
                                                      if nulls[s][(base, readout, k)] is not None]))
                                if all(nulls[s][(base, readout, k)] is not None for s in SEEDS)
                                else None,
                        })
            got = {"variant": label, "S": S, "rows": new_rows,
                   "seconds": time.time() - t0}
            _save(tag, got)
            print(f"  {tag}: {got['seconds']:.0f}s", file=sys.stderr)
        rows.extend(got["rows"])
    # V1 fraction of S
    for base in ("phys", "phi", "none"):
        for readout in ("b1", "unp"):
            v1 = [r for r in rows if r["variant"] == "V1"
                  and r["baseline"] == base and r["readout"] == readout
                  and r["band"] == 0]
            if v1:
                m = v1[0]["mean"]
                for S in (0.25, 1.0, 4.0):
                    v1_frac.append({"baseline": base, "readout": readout,
                                    "S": S, "contrast": m,
                                    "fraction_of_S": m / S})
    return {"rows": rows, "v1_fraction": v1_frac}


def rows_slice(rows, label):
    return [r for r in rows if r["variant"] == label]


# ---------------------------------------------------------------------------
# H2
# ---------------------------------------------------------------------------

def run_h2():
    got = _load("h2")
    if got is not None:
        return got
    print("  H2...", file=sys.stderr)
    stream, onsets, meals, stretches = build_h5(100, "V1", 0.0)
    mid = 0.5 * (stream.t_start + stream.t_end)
    p = int(np.argmin(np.abs(mid - LIFE / 2.0)))
    cells = []
    cases = []
    for n in (1, 2, 10, 100):
        ks = np.array([5 + (i % 10) for i in range(n)])
        js = np.array([p - (2 ** k - (i // 10)) for i, k in enumerate(ks)])
        for n0 in (0.2, 1.0, 5.0):
            cases.append({"case": f"A_n{n}_n0{n0}", "anchors": js,
                          "ks": ks, "weights": np.ones(n), "n0": n0})
    ks = np.arange(5, 15)
    js = np.array([p - (2 ** k - (i // 10)) for i, k in enumerate(ks)])
    cases.append({"case": "C", "anchors": js, "ks": ks,
                  "weights": np.full(10, 0.1), "n0": 0.2})
    ksD = np.full(10, 14, dtype=np.int64)
    jsD = np.array([p - 3 - 2 ** 14 - 100 * i for i in range(10)])
    cases.append({"case": "D", "anchors": jsD, "ks": ksD,
                  "weights": np.ones(10), "n0": 1.0})
    for c in cases:
        rec = {"case": c["case"], "target": int(p),
               "anchors": [int(x) for x in c["anchors"]],
               "ks": [int(x) for x in c["ks"]],
               "n0": float(c["n0"])}
        for readout in ("unp", "b1"):
            lev = leverage(stream, p, c["anchors"], c["ks"], c["weights"],
                           c["n0"], "phys", readout, broken=False)
            rec[readout] = lev
            if readout == "b1":
                assert lev["L_direct"] <= 1.0 + 1e-9, lev
        cells.append(rec)
    out = {"target": int(p), "M": int(stream.M), "cells": cells,
           "all_b1_direct_le_1": True}
    _save("h2", out)
    return out


# ---------------------------------------------------------------------------
# H9 (a)
# ---------------------------------------------------------------------------

def run_h9a():
    got = _load("h9a")
    if got is not None:
        return got
    out = {"models": {}}
    for model in ("A", "B"):
        naive = []
        matched = []
        for rep in range(200):
            rng = np.random.default_rng(rep * 1009 + (0 if model == "A" else 1))
            n = 2000
            ing = rng.random(n) < 0.5
            x = np.where(ing, rng.beta(4, 2, n), rng.beta(2, 4, n))
            p = 0.2 + 0.5 * x + (0.15 * ing if model == "B" else 0.0)
            turn = rng.random(n) < p
            if np.any(ing) and np.any(~ing):
                naive.append(float(turn[ing].mean() - turn[~ing].mean()))
            bins = np.minimum((x * 10).astype(int), 9)
            num = 0.0
            den = 0.0
            for b in range(10):
                m = bins == b
                mi = m & ing
                mn = m & ~ing
                if mi.sum() == 0 or mn.sum() == 0:
                    continue
                d = float(turn[mi].mean() - turn[mn].mean())
                num += mi.sum() * d
                den += mi.sum()
            matched.append(num / den if den > 0 else np.nan)
        naive = np.array(naive)
        matched = np.array(matched)
        out["models"][model] = {
            "naive_mean": float(np.nanmean(naive)),
            "naive_2p5": float(np.nanpercentile(naive, 2.5)),
            "naive_97p5": float(np.nanpercentile(naive, 97.5)),
            "matched_mean": float(np.nanmean(matched)),
            "matched_2p5": float(np.nanpercentile(matched, 2.5)),
            "matched_97p5": float(np.nanpercentile(matched, 97.5)),
        }
    _save("h9a", out)
    return out


# ---------------------------------------------------------------------------
# H9 (b)
# ---------------------------------------------------------------------------

def lag_auditor(meals, costs, L=50.0):
    """meals/costs: list of (unit index, cue id). Fraction with lag > 2L."""
    by_cue = {}
    for idx, cue in meals:
        by_cue.setdefault(cue, []).append(idx)
    for cue in by_cue:
        by_cue[cue].sort()
    hit = 0
    for idx, cue in costs:
        arr = by_cue.get(cue, [])
        j = int(np.searchsorted(arr, idx, side="right")) - 1
        if j >= 0 and (idx - arr[j]) > 2 * L:
            hit += 1
    return hit / len(costs) if costs else 0.0


def _schedule_p(kind, n_costs=100):
    meals = []
    costs = []
    for i in range(n_costs):
        m = i * 150
        meals.append((m, 0))
        costs.append((m + 125, 0))
    for x in range(0, n_costs * 150, 25):
        meals.append((x + 12, 1))
    return meals, costs


def _schedule_f(kind, n_costs=100):
    meals = []
    costs = []
    for m in range(0, (n_costs + 4) * 50, 50):
        meals.append((m, 0))
    for i in range(n_costs):
        costs.append((i * 50 + 125, 0))
    for x in range(0, n_costs * 50, 25):
        meals.append((x + 12, 1))
    return meals, costs


def _schedule_mixed(n_p, n_f):
    meals, costs = [], []
    # P-like region
    for i in range(n_p):
        m = i * 150
        meals.append((m, 0))
        costs.append((m + 125, 0))
    for x in range(0, n_p * 150, 25):
        meals.append((x + 12, 1))
    base = n_p * 150 + 500
    for m in range(base, base + (n_f + 4) * 50, 50):
        meals.append((m, 0))
    for i in range(n_f):
        costs.append((base + i * 50 + 125, 0))
    for x in range(base, base + n_f * 50, 25):
        meals.append((x + 12, 1))
    return meals, costs


def run_h9b():
    got = _load("h9b")
    if got is not None:
        return got
    blocks = {}
    for name, (meals, costs) in (
            ("P", _schedule_p("P")),
            ("F", _schedule_f("F")),
            ("M96", _schedule_mixed(96, 4)),
            ("M94", _schedule_mixed(94, 6))):
        blocks[name] = {"fraction": lag_auditor(meals, costs, 50.0),
                        "n_costs": len(costs)}
    out = {"blocks": blocks,
           "threshold_095_pass_M96": bool(blocks["M96"]["fraction"] >= 0.95),
           "threshold_095_fail_M94": bool(blocks["M94"]["fraction"] < 0.95)}
    _save("h9b", out)
    return out


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

def run_cost():
    got = _load("cost")
    if got is not None:
        return got
    t0 = time.perf_counter()
    stream, onsets, meals, stretches = build_h5(0, "V1", 0.0)
    t_build = time.perf_counter() - t0
    all_pos = np.unique(np.concatenate([onsets["busy"], onsets["quiet"]]))
    t0 = time.perf_counter()
    ub = _ub_all(stream, all_pos, "phys")
    t_base = time.perf_counter() - t0
    t0 = time.perf_counter()
    for readout in ("b1", "unp"):
        for k in range(1, 15):
            _contrast(stream, onsets["busy"], onsets["quiet"], readout,
                      "phys", ub, kband=k)
        _contrast(stream, onsets["busy"], onsets["quiet"], readout, "phys",
                  ub, kband=None)
    t_read = time.perf_counter() - t0
    # per-candidate readout, 300 anchors
    rng = np.random.default_rng(0)
    pos = rng.choice(np.arange(0, stream.M - 2 ** 14), size=300, replace=False)
    ks = rng.integers(1, 15, size=300)
    w = np.ones(300)
    ubc = ubar_dict(stream, pos, ks, "phys")
    t0 = time.perf_counter()
    v_b1_fast(stream, pos, ks, w, 1.0, ubc, "phys")
    t_cand_b1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    v_unp(stream, pos, ks, w, 1.0, ubc, "phys")
    t_cand_unp = time.perf_counter() - t0
    out = {"stream_build_s": t_build, "baseline_s": t_base,
           "readouts_s": t_read,
           "life_total_s": t_build + t_base + t_read,
           "candidate_300_b1_s": t_cand_b1,
           "candidate_300_unp_s": t_cand_unp,
           "M": int(stream.M)}
    _save("cost", out)
    return out


# ---------------------------------------------------------------------------
# tables / main
# ---------------------------------------------------------------------------

def write_tables(path, res):
    with open(path, "w") as f:
        f.write("# revision 6 tables (far-value readout)\n\n")

        f.write("## Item 2: H2 leverage\n\n")
        f.write("| case | target | n0 | L_direct unp | L_total unp | "
                "L_direct B1 | L_total B1 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for c in res["item2"]["cells"]:
            f.write(f"| {c['case']} | {c['target']} | {c['n0']} | "
                    f"{_fmt(c['unp']['L_direct'],6)} | "
                    f"{_fmt(c['unp']['L_total'],6)} | "
                    f"{_fmt(c['b1']['L_direct'],6)} | "
                    f"{_fmt(c['b1']['L_total'],6)} |\n")

        f.write("\n## Item 3: H5 contrasts\n\n")
        f.write("| variant | baseline | readout | band | mean | sd | "
                "95% CI low | 95% CI high | relabel SD |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in res["item3"]["rows"]:
            f.write(f"| {r['variant']} | {r['baseline']} | {r['readout']} | "
                    f"{r['band']} | {_fmt(r['mean'],5)} | {_fmt(r['sd'],5)} | "
                    f"{_fmt(r['ci_lo'],5)} | {_fmt(r['ci_hi'],5)} | "
                    f"{_fmt(r['null_sd'],5)} |\n")
        f.write("\nV1 contrast as a fraction of S:\n\n")
        f.write("| baseline | readout | S | contrast | fraction |\n")
        f.write("|---|---|---|---|---|\n")
        for r in res["item3"]["v1_fraction"]:
            f.write(f"| {r['baseline']} | {r['readout']} | {r['S']} | "
                    f"{_fmt(r['contrast'],5)} | {_fmt(r['fraction_of_S'],5)} |\n")

        f.write("\n## Item 4: H9(a)\n\n")
        f.write("| model | estimator | mean | 2.5% | 97.5% |\n")
        f.write("|---|---|---|---|---|\n")
        for model in ("A", "B"):
            d = res["item4"]["models"][model]
            f.write(f"| {model} | naive | {_fmt(d['naive_mean'],5)} | "
                    f"{_fmt(d['naive_2p5'],5)} | {_fmt(d['naive_97p5'],5)} |\n")
            f.write(f"| {model} | matched | {_fmt(d['matched_mean'],5)} | "
                    f"{_fmt(d['matched_2p5'],5)} | {_fmt(d['matched_97p5'],5)} |\n")

        f.write("\n## Item 5: H9(b) lag auditor\n\n")
        f.write("| schedule | costs | fraction |\n|---|---|---|\n")
        for name, b in res["item5"]["blocks"].items():
            f.write(f"| {name} | {b['n_costs']} | {_fmt(b['fraction'],4)} |\n")
        f.write(f"\n0.95 threshold: M96 "
                f"{'PASS' if res['item5']['threshold_095_pass_M96'] else 'FAIL'}, "
                f"M94 {'FAIL' if res['item5']['threshold_095_fail_M94'] else 'PASS'}.\n")

        f.write("\n## Item 6: cost\n\n")
        c = res["item6"]
        f.write(f"One H5 V1 life: stream + baselines + all readouts "
                f"{c['life_total_s']:.3f} s (build {c['stream_build_s']:.3f}, "
                f"baselines {c['baseline_s']:.3f}, readouts {c['readouts_s']:.3f}); "
                f"M = {c['M']} records. Per-candidate 300 anchors: B1 "
                f"{c['candidate_300_b1_s']:.4f} s, unpatched "
                f"{c['candidate_300_unp_s']:.4f} s.\n")


def main():
    os.makedirs(OUT6, exist_ok=True)
    item2 = run_h2()
    item3 = run_h5()
    item4 = run_h9a()
    item5 = run_h9b()
    item6 = run_cost()
    res = {"meta": {"design": "revision 6 (20 Nov 2026)",
                    "dt": DT, "life": LIFE, "L": L, "CYC": CYC,
                    "seeds": SEEDS, "total_seconds": None},
           "item2": item2, "item3": item3, "item4": item4,
           "item5": item5, "item6": item6}
    with open(os.path.join(OUT6, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_tables(os.path.join(OUT6, "tables.md"), res)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
