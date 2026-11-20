"""Follow-up 9: open bands, per-channel far value, yoked control, encounter spacing."""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

from step0a.far_value import Stream
from step0a.rigs.run_item3b import (
    KMAX, DT, CYC, T975_19, build_world, sigma_k, ub_all, contrast,
    relabel_null, stats,
)
from step0a.rigs.run_item3c import R_table, Y_table, scale_v
from step0a.world_signals import _ou_on_grid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT9 = os.path.join(ROOT, "runs", "out9")
LIFE = 20000.0
SEEDS = list(range(20))
TAUS = (0.25, 1.0, 4.0)


def _part(name):
    return os.path.join(OUT9, f"part_{name}.json")


def _save(name, obj):
    os.makedirs(OUT9, exist_ok=True)
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


def open_shrunk_v(R, idx, n0=1.0):
    per = {}
    total = 0.0
    for k in sorted(R):
        r = R[k][idx]
        m = ~np.isnan(r)
        n = int(m.sum())
        v = float(np.sum(r[m])) / (n0 + n) if n else 0.0
        per[k] = v
        total += v
    return total, per


# ---------------------------------------------------------------------------
# item 1: open bands on the density worlds
# ---------------------------------------------------------------------------

def _physical_total(st, ons, kmax=KMAX):
    mdP = st.median_dP
    M = st.M
    tot = 0.0
    for k in range(1, kmax + 1):
        lo, hi = 2 ** (k - 1) * mdP, 2 ** k * mdP
        Ys = []
        for j in ons.tolist():
            base = st.prefDP[j + 1]
            if base + hi > st.prefDP[M]:
                continue
            a0 = int(np.searchsorted(st.prefDP, base + lo, side="right"))
            a1 = int(np.searchsorted(st.prefDP, base + hi, side="right"))
            if a1 > a0:
                Ys.append(float(st.prefU[a1] - st.prefU[a0]))
        N = len(Ys)
        tot += float(np.sum(Ys)) / (1.0 + N) if N else 0.0
    return tot


def run_item1():
    got = _load("i1")
    if got is not None:
        return got
    print("  item1 open bands...", file=sys.stderr)
    variants = [("rho5", 5.0, False), ("rho20", 20.0, False),
                ("rho80", 80.0, False), ("cluster", 5.0, True)]
    rows = []
    for tag, rho_h, cluster in variants:
        sh, un, ph = [], [], []
        kmaxs = []
        for seed in SEEDS:
            w = build_world(seed, "density", S=1.0, D=30.0, rho_h=rho_h,
                            cluster=cluster, clean=True)
            st = w["stream"]
            ons = w["onsets"]["harm"]
            M = st.M
            K = int(np.floor(np.log2(M))) + 1
            kmaxs.append(K)
            R = Y_table(st, ons, kmax=K)
            idx = np.arange(len(ons))
            v1, _ = open_shrunk_v(R, idx, 1.0)
            v0, _ = open_shrunk_v(R, idx, 1e-9)
            sh.append(v1)
            un.append(v0)
            ph.append(_physical_total(st, ons))
        rows.append({"variant": tag, "k_open_mean": float(np.mean(kmaxs)),
                     "V_shrunk": float(np.mean(sh)),
                     "V_unshrunk": float(np.mean(un)),
                     "V_physical": float(np.mean(ph)),
                     "k_open_values": kmaxs})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# two-channel world
# ---------------------------------------------------------------------------

def build_two(seed, S, D, pulse_on="harm", life=LIFE):
    rng = np.random.default_rng(seed * 10007 + 2)
    stretches = []
    t = 0.0
    while t < life - 1e-12:
        dur = min(float(rng.uniform(40.0, 80.0)), life - t)
        u = float(rng.random())
        typ = "busy" if u < 0.45 else ("quiet" if u < 0.90 else "harm")
        stretches.append((t, dur, typ))
        t += dur
    rngm = np.random.default_rng(seed * 10007 + 777 + 2)
    meals = []
    tm = 0.0
    while True:
        tm += float(rngm.exponential(300.0))
        if tm >= life:
            break
        meals.append(tm)
    n = int(round(life / DT)) + 1
    tg = np.arange(n, dtype=np.float64) * DT
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
        base = E[a]
        E[a:b] = np.maximum(0.0, base - 0.001 * DT * np.arange(b - a))
        if b >= n:
            break
        E_prev = E[b - 1]
        if b in meal_set:
            E_prev = min(1.0, E_prev + 0.3)
        E[b] = E_prev
    uE = -(np.maximum(0.0, 1.0 - E)) ** 2
    m = _ou_on_grid(n, 0.01, 5.0, seed * 10007 + 555, DT)
    for t0, dur, typ in stretches:
        if typ != pulse_on:
            continue
        i0 = max(0, int(np.ceil((t0 + D) / DT)))
        i1 = min(n, int(np.ceil((t0 + D + 1.0) / DT)))
        if i1 > i0:
            m[i0:i1] += S
    uM = -np.maximum(m, 0.0)
    segs = []
    for t0, dur, typ in stretches:
        t1 = t0 + dur
        rho = 20.0 if typ == "busy" else 5.0
        segs.append((t0, t1, rho))
    ts = []
    phi = 0.0
    for a, b, rho in segs:
        k0 = int(np.floor(phi)) + 1
        k1 = int(np.floor(phi + rho * (b - a) + 1e-12))
        if k1 >= k0:
            kk = np.arange(k0, k1 + 1, dtype=np.int64)
            tt = a + (kk - phi) / rho
            tt = tt[(tt <= b + 1e-9) & (tt <= life + 1e-9)]
            if len(tt):
                ts.append(tt)
        phi += rho * (b - a)
    ts = np.concatenate(ts) if ts else np.empty(0)
    t_end = ts
    t_start = np.concatenate(([0.0], ts[:-1]))
    def integ(u):
        cp = np.concatenate(([0.0], np.cumsum(np.maximum(u, 0.0)) * DT))
        cm = np.concatenate(([0.0], np.cumsum(np.maximum(-u, 0.0)) * DT))
        ip = np.interp(t_end, tg, cp[:-1]) - np.interp(t_start, tg, cp[:-1])
        im = np.interp(t_end, tg, cm[:-1]) - np.interp(t_start, tg, cm[:-1])
        return ip - im
    UE = integ(uE)
    UM = integ(uM)
    E_start = np.interp(t_start, tg, E)
    dP = CYC * (t_end - t_start)
    onsets = {"busy": [], "quiet": [], "harm": []}
    for t0, dur, typ in stretches:
        p = int(np.searchsorted(t_end, t0, side="left"))
        if p < len(ts):
            onsets[typ].append(p)
    for k in onsets:
        onsets[k] = np.array(sorted(set(onsets[k])), dtype=np.int64)
    return {"UE": Stream(UE, dP, E_start, t_start, t_end),
            "UM": Stream(UM, dP, E_start, t_start, t_end),
            "UC": Stream(UE + UM, dP, E_start, t_start, t_end),
            "onsets": onsets, "stretches": stretches, "E": E}


def scale_multi(Rs, sigs, idx, tau, n0=1.0):
    per = {}
    total = 0.0
    for k in range(1, KMAX + 1):
        vk = 0.0
        for R, sig in zip(Rs, sigs):
            r = R[k][idx]
            m = ~np.isnan(r)
            n = int(m.sum())
            if n:
                vk += float(np.sum(r[m])) / (n0 * sig[k] ** 2 / tau ** 2 + n)
        per[k] = vk
        total += vk
    return total, per


def relabel_multi(Rs, sigs, i1, i2, tau, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    union = np.concatenate([i1, i2])
    n1 = len(i1)
    per = {k: [] for k in list(range(1, KMAX + 1)) + [0]}
    for _ in range(reps):
        perm = rng.permutation(union)
        ta, pa = scale_multi(Rs, sigs, perm[:n1], tau)
        tb, pb = scale_multi(Rs, sigs, perm[n1:], tau)
        per[0].append(ta - tb)
        for k in range(1, KMAX + 1):
            per[k].append(pa[k] - pb[k])
    return {k: float(np.std(per[k], ddof=1)) for k in per}


def matured_pulses(stretches, D, life):
    return sum(1 for t0, dur, typ in stretches
               if typ == "harm" and t0 + D + 1.0 <= life)


# ---------------------------------------------------------------------------
# item 2
# ---------------------------------------------------------------------------

def run_item2():
    got = _load("i2")
    if got is not None:
        return got
    print("  item2 per-channel...", file=sys.stderr)
    variants = [(D, S) for D in (30, 120, 480, 1920) for S in (1.0, 4.0)]
    rows = []
    zrows = []
    for D, S in variants:
        tag = f"D{D}_S{S}"
        gotv = _load(tag)
        if gotv is None:
            per = {}
            zs = {}
            lives_used = LIFE
            for seed in SEEDS:
                w = build_two(seed, S, D, "harm", LIFE)
                if D == 1920 and matured_pulses(w["stretches"], D, LIFE) < 10:
                    w = build_two(seed, S, D, "harm", 30000.0)
                    lives_used = 30000.0
                on = w["onsets"]
                g1, g2 = on["harm"], on["quiet"]
                master = np.unique(np.concatenate([g1, g2]))
                i1 = np.searchsorted(master, g1)
                i2 = np.searchsorted(master, g2)
                streams = {"E": w["UE"], "M": w["UM"], "C": w["UC"]}
                sigs = {c: sigma_k(st) for c, st in streams.items()}
                R = {c: R_table(st, master, "phys") for c, st in streams.items()}
                for tau in TAUS:
                    t2, p2 = scale_multi([R["E"], R["M"]],
                                         [sigs["E"], sigs["M"]], i1, tau)
                    t2b, p2b = scale_multi([R["E"], R["M"]],
                                           [sigs["E"], sigs["M"]], i2, tau)
                    t1c, p1c = scale_v(R["C"], i1, sigs["C"], tau)
                    t1d, p1d = scale_v(R["C"], i2, sigs["C"], tau)
                    for k in range(1, KMAX + 1):
                        per.setdefault(("2ch", tau, k), []).append(p2[k] - p2b[k])
                        per.setdefault(("1ch", tau, k), []).append(p1c[k] - p1d[k])
                    per.setdefault(("2ch", tau, 0), []).append(t2 - t2b)
                    per.setdefault(("1ch", tau, 0), []).append(t1c - t1d)
                    # channel-only totals for context
                    te, pe = scale_v(R["E"], i1, sigs["E"], tau)
                    te2, pe2 = scale_v(R["E"], i2, sigs["E"], tau)
                    tm, pm = scale_v(R["M"], i1, sigs["M"], tau)
                    tm2, pm2 = scale_v(R["M"], i2, sigs["M"], tau)
                    per.setdefault(("E", tau, 0), []).append(te - te2)
                    per.setdefault(("M", tau, 0), []).append(tm - tm2)
                    for name, Rs, sg in (("2ch", [R["E"], R["M"]],
                                          [sigs["E"], sigs["M"]]),
                                         ("1ch", [R["C"]], [sigs["C"]])):
                        nsd = relabel_multi(Rs, sg, i1, i2, tau,
                                            seed=seed * 41 + int(tau * 10) +
                                            (0 if name == "2ch" else 1))
                        for k in list(range(1, KMAX + 1)) + [0]:
                            per.setdefault((name, tau, "n", k), []).append(nsd[k])
                        if nsd[0] and nsd[0] > 0:
                            zs.setdefault((name, tau), []).append(
                                per[(name, tau, 0)][-1] / nsd[0])
                        else:
                            zs.setdefault((name, tau), []).append(np.nan)
            new_rows = []
            new_z = []
            for name in ("2ch", "1ch"):
                for tau in TAUS:
                    for k in list(range(1, KMAX + 1)) + [0]:
                        s = stats(per[(name, tau, k)])
                        new_rows.append({"variant": tag, "D": D, "S": S,
                                         "readout": name, "tau": tau,
                                         "band": k, **s,
                                         "null_sd": float(np.nanmean(
                                             per[(name, tau, "n", k)]))})
                    zz = np.asarray(zs[(name, tau)], dtype=float)
                    new_z.append({"variant": tag, "D": D, "S": S,
                                  "readout": name, "tau": tau,
                                  "mean_z": float(np.nanmean(zz)),
                                  "frac_z_lt_-2.33": float(np.nanmean(zz < -2.33)),
                                  "frac_z_gt_2.33": float(np.nanmean(zz > 2.33))})
            gotv = {"variant": tag, "D": D, "S": S, "life": lives_used,
                    "rows": new_rows, "z": new_z}
            _save(tag, gotv)
        rows.extend(gotv["rows"])
        zrows.extend(gotv["z"])
    return {"rows": rows, "z": zrows}


# ---------------------------------------------------------------------------
# item 3: yoked control
# ---------------------------------------------------------------------------

def run_item3():
    got = _load("i3")
    if got is not None:
        return got
    print("  item3 yoked...", file=sys.stderr)
    D, S = 480.0, 4.0
    zs = {}
    for seed in SEEDS:
        w = build_two(seed, S, D, "quiet", LIFE)
        on = w["onsets"]
        g1, g2 = on["harm"], on["quiet"]
        master = np.unique(np.concatenate([g1, g2]))
        i1 = np.searchsorted(master, g1)
        i2 = np.searchsorted(master, g2)
        streams = {"E": w["UE"], "M": w["UM"], "C": w["UC"]}
        sigs = {c: sigma_k(st) for c, st in streams.items()}
        R = {c: R_table(st, master, "phys") for c, st in streams.items()}
        for tau in TAUS:
            for name, Rs, sg in (("2ch", [R["E"], R["M"]],
                                  [sigs["E"], sigs["M"]]),
                                 ("1ch", [R["C"]], [sigs["C"]]),
                                 ("E", [R["E"]], [sigs["E"]]),
                                 ("M", [R["M"]], [sigs["M"]])):
                if name == "2ch":
                    ta, _ = scale_multi(Rs, sg, i1, tau)
                    tb, _ = scale_multi(Rs, sg, i2, tau)
                else:
                    ta, _ = scale_v(Rs[0], i1, sg[0], tau)
                    tb, _ = scale_v(Rs[0], i2, sg[0], tau)
                nsd = relabel_multi(Rs, sg, i1, i2, tau,
                                    seed=seed * 43 + int(tau * 10) +
                                    {"2ch": 0, "1ch": 1, "E": 2, "M": 3}[name])
                zs.setdefault((name, tau), []).append(
                    (ta - tb) / nsd[0] if nsd[0] else np.nan)
    rows = []
    for (name, tau), vals in zs.items():
        v = np.asarray(vals, dtype=float)
        rows.append({"readout": name, "tau": tau,
                     "mean_z": float(np.nanmean(v)),
                     "frac_lt": float(np.nanmean(v < -2.33)),
                     "frac_gt": float(np.nanmean(v > 2.33))})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# item 4: spacing
# ---------------------------------------------------------------------------

def _band10_z(seed, life, S=4.0, D=120.0):
    w = build_world(seed, "V3", S=S, D=D, life=life)
    st = w["stream"]
    g1, g2 = w["onsets"]["harm"], w["onsets"]["quiet"]
    allp = np.unique(np.concatenate([g1, g2]))
    ub = ub_all(st, allp, "phys")
    ones = np.ones(KMAX + 1)
    c = contrast(st, g1, g2, ones, "phys", ub, "b1", kband=10)
    ns = relabel_null(st, g1, g2, ones, "phys", ub, kband=10,
                      seed=seed * 47)
    return c / ns if ns else np.nan


def run_item4():
    got = _load("i4")
    if got is not None:
        return got
    print("  item4 spacing...", file=sys.stderr)
    z1 = [_band10_z(s, 20000.0) for s in SEEDS]
    z4 = [_band10_z(s, 80000.0) for s in SEEDS]
    m1 = float(np.nanmean(z1))
    m4 = float(np.nanmean(z4))
    out = {"z_band10_20k": m1, "z_band10_80k": m4,
           "ratio_4x_over_1x": m4 / m1 if m1 != 0 else None,
           "z_20k_lives": [float(x) for x in z1],
           "z_80k_lives": [float(x) for x in z4]}
    _save("i4", out)
    return out


# ---------------------------------------------------------------------------
# tables / main
# ---------------------------------------------------------------------------

def write_tables(path, res):
    with open(path, "w") as f:
        f.write("# revision 9 tables\n\n")
        f.write("## Item 1: open Phi bands\n\n")
        f.write("| variant | k_open mean | shrunk | unshrunk | physical |\n")
        f.write("|---|---|---|---|---|\n")
        for r in res["item1"]["rows"]:
            f.write(f"| {r['variant']} | {_fmt(r['k_open_mean'],2)} | "
                    f"{_fmt(r['V_shrunk'],5)} | {_fmt(r['V_unshrunk'],5)} | "
                    f"{_fmt(r['V_physical'],5)} |\n")
        f.write("\n## Item 2: per-channel far value\n\n")
        f.write("| variant | readout | tau | band | mean | SD | 95% CI | relabel SD |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in res["item2"]["rows"]:
            f.write(f"| {r['variant']} | {r['readout']} | {r['tau']} | "
                    f"{r['band']} | {_fmt(r['mean'],4)} | {_fmt(r['sd'],4)} | "
                    f"[{_fmt(r['ci_lo'],4)}, {_fmt(r['ci_hi'],4)}] | "
                    f"{_fmt(r['null_sd'],4)} |\n")
        f.write("\nTotal z:\n\n| variant | readout | tau | mean z | frac < -2.33 | frac > +2.33 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for z in res["item2"]["z"]:
            f.write(f"| {z['variant']} | {z['readout']} | {z['tau']} | "
                    f"{_fmt(z['mean_z'],4)} | {_fmt(z['frac_z_lt_-2.33'],4)} | "
                    f"{_fmt(z['frac_z_gt_2.33'],4)} |\n")
        f.write("\n## Item 3: yoked control\n\n")
        f.write("| readout | tau | mean z | frac < -2.33 | frac > +2.33 |\n")
        f.write("|---|---|---|---|---|\n")
        for r in res["item3"]["rows"]:
            f.write(f"| {r['readout']} | {r['tau']} | {_fmt(r['mean_z'],4)} | "
                    f"{_fmt(r['frac_lt'],4)} | {_fmt(r['frac_gt'],4)} |\n")
        f.write("\n## Item 4: encounter spacing\n\n")
        f.write(f"Band-10 per-life z: 20k {res['item4']['z_band10_20k']:.4f}, "
                f"80k {res['item4']['z_band10_80k']:.4f}, ratio "
                f"{_fmt(res['item4']['ratio_4x_over_1x'],4)}.\n")


def main():
    os.makedirs(OUT9, exist_ok=True)
    item1 = run_item1()
    item2 = run_item2()
    item3 = run_item3()
    item4 = run_item4()
    res = {"meta": {"design": "revision 9 (20 Nov 2026)",
                    "dt": DT, "life": LIFE, "seeds": SEEDS},
           "item1": item1, "item2": item2, "item3": item3, "item4": item4}
    with open(os.path.join(OUT9, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_tables(os.path.join(OUT9, "tables.md"), res)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
