"""Follow-up 8: absolute-scale precision, noise-vs-width, density, V1b recheck."""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

from step0a.far_value import compute_ubar
from step0a.pacemaker import PERIODS
from step0a.rigs.run_item3b import (
    KMAX, DT, CYC, T975_19, build_world, sigma_k, precision, ub_all,
    contrast, relabel_null, group_value, stats, _pairs,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT8 = os.path.join(ROOT, "runs", "out8")
OUT7 = os.path.join(ROOT, "runs", "out7")
LIFE = 20000.0
SEEDS = list(range(20))
SEEDS60 = list(range(20, 80))


def _part(name):
    return os.path.join(OUT8, f"part_{name}.json")


def _save(name, obj):
    os.makedirs(OUT8, exist_ok=True)
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
# R tables and scale readout
# ---------------------------------------------------------------------------

def R_table(stream, positions, mode="phys", kmax=KMAX):
    positions = np.asarray(positions, dtype=np.int64)
    out = {}
    for k in range(1, kmax + 1):
        r = np.full(len(positions), np.nan)
        m = positions <= stream.M - 2 ** k - 1
        if np.any(m):
            ub, Y, dP = compute_ubar(stream, k, positions[m], mode)
            r[m] = Y - ub * dP
        out[k] = r
    return out


def Y_table(stream, positions, kmax=KMAX):
    positions = np.asarray(positions, dtype=np.int64)
    out = {}
    for k in range(1, kmax + 1):
        r = np.full(len(positions), np.nan)
        m = positions <= stream.M - 2 ** k - 1
        q = 2 ** k
        half = 2 ** (k - 1)
        p = positions[m]
        if len(p):
            r[m] = stream.prefU[p + q + 1] - stream.prefU[p + half + 1]
        out[k] = r
    return out


def scale_v(R, idx, sigma, tau, n0=1.0):
    """V_k = sum_j R_jk / (n0 sigma_k^2 / tau^2 + sum_j e_jk); total = sum_k."""
    if len(idx) == 0:
        return 0.0, {k: 0.0 for k in range(1, KMAX + 1)}
    per = {}
    total = 0.0
    for k in range(1, KMAX + 1):
        r = R[k][idx]
        m = ~np.isnan(r)
        n = int(m.sum())
        if n == 0:
            vk = 0.0
        else:
            den = n0 * sigma[k] ** 2 / (tau ** 2) + n
            vk = float(np.sum(r[m])) / den
        per[k] = vk
        total += vk
    return total, per


def shrunk_v(R, idx, n0=1.0):
    """V_k = sum_j R_jk / (n0 + N_k) (B_none form)."""
    per = {}
    total = 0.0
    for k in range(1, KMAX + 1):
        r = R[k][idx]
        m = ~np.isnan(r)
        n = int(m.sum())
        vk = float(np.sum(r[m])) / (n0 + n) if n else 0.0
        per[k] = vk
        total += vk
    return total, per


def relabel_scale(R, g1_idx, g2_idx, sigma, tau, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    union = np.concatenate([g1_idx, g2_idx])
    n1 = len(g1_idx)
    keys = list(range(1, KMAX + 1)) + [0]
    vals = {k: [] for k in keys}
    for _ in range(reps):
        perm = rng.permutation(union)
        a, b = perm[:n1], perm[n1:]
        ta, pa = scale_v(R, a, sigma, tau)
        tb, pb = scale_v(R, b, sigma, tau)
        for k in range(1, KMAX + 1):
            vals[k].append(pa[k] - pb[k])
        vals[0].append(ta - tb)
    return {k: float(np.std(vals[k], ddof=1)) if len(vals[k]) > 1 else None
            for k in keys}


# ---------------------------------------------------------------------------
# item 1
# ---------------------------------------------------------------------------

def run_item1():
    variants = [("S1_D30", 1.0, 30.0), ("S4_D30", 4.0, 30.0),
                ("S1_D120", 1.0, 120.0), ("S4_D120", 4.0, 120.0),
                ("S1_D480", 1.0, 480.0), ("S4_D480", 4.0, 480.0)]
    taus = (0.25, 1.0, 4.0)
    rows = []
    zrows = []
    for tag, S, D in variants:
        got = _load(f"i1_{tag}")
        if got is None:
            print(f"  item1 {tag}...", file=sys.stderr)
            t0 = time.time()
            per = {}
            new_rows = []
            new_z = []
            z = {(tau): [] for tau in taus}
            for seed in SEEDS:
                w = build_world(seed, "V3", S=S, D=D)
                st = w["stream"]
                g1 = w["onsets"]["harm"]
                g2 = w["onsets"]["quiet"]
                master = np.unique(np.concatenate([g1, g2]))
                i1 = np.searchsorted(master, g1)
                i2 = np.searchsorted(master, g2)
                sig = sigma_k(st)
                R = R_table(st, master, "phys")
                for tau in taus:
                    t1, p1 = scale_v(R, i1, sig, tau)
                    t2, p2 = scale_v(R, i2, sig, tau)
                    for k in range(1, KMAX + 1):
                        per.setdefault((tau, k), []).append(p1[k] - p2[k])
                    per.setdefault((tau, 0), []).append(t1 - t2)
                    nsd = relabel_scale(R, i1, i2, sig, tau,
                                        seed=seed * 23 + int(tau * 10))
                    for k in range(1, KMAX + 1):
                        per.setdefault((tau, "n", k), []).append(nsd[k])
                    per.setdefault((tau, "n", 0), []).append(nsd[0])
                    if nsd[0] and nsd[0] > 0:
                        z[tau].append((t1 - t2) / nsd[0])
                    else:
                        z[tau].append(np.nan)
            for tau in taus:
                for k in range(1, KMAX + 1):
                    s = stats(per[(tau, k)])
                    new_rows.append({"variant": tag, "S": S, "D": D, "tau": tau,
                                 "band": k, **s,
                                 "null_sd": float(np.nanmean(per[(tau, "n", k)]))})
                s = stats(per[(tau, 0)])
                new_rows.append({"variant": tag, "S": S, "D": D, "tau": tau,
                             "band": 0, **s,
                             "null_sd": float(np.nanmean(per[(tau, "n", 0)]))})
                zz = np.asarray(z[tau], dtype=float)
                new_z.append({"variant": tag, "S": S, "D": D, "tau": tau,
                              "mean_z": float(np.nanmean(zz)),
                              "frac_z_lt_-2.33": float(np.nanmean(zz < -2.33))})
            got = {"variant": tag, "S": S, "D": D,
                   "rows": new_rows,
                   "z": new_z,
                   "seconds": time.time() - t0}
            _save(f"i1_{tag}", got)
            print(f"  {tag}: {got['seconds']:.0f}s", file=sys.stderr)
        rows.extend(got["rows"])
        zrows.extend(got["z"])
    return {"rows": rows, "z": zrows}


# ---------------------------------------------------------------------------
# item 2
# ---------------------------------------------------------------------------

def _fit_b(points):
    x = np.array([math.log(w) for w, sd in points], dtype=float)
    y = np.array([math.log(sd) for w, sd in points], dtype=float)
    n = len(x)
    xm = x.mean()
    ym = y.mean()
    b = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    a = float(ym - b * xm)
    resid = y - (a + b * x)
    se = float(np.sqrt(np.sum(resid ** 2) / (n - 2) / np.sum((x - xm) ** 2)))
    return {"a": a, "b": b, "se": se, "ci_lo": b - 1.96 * se,
            "ci_hi": b + 1.96 * se, "n": n}


def build_scaled(seed, S, D, scale, life=LIFE):
    """V3 world with stretch durations divided by scale (4x encounters)."""
    rng = np.random.default_rng(seed * 10007 + 2)
    stretches = []
    t = 0.0
    while t < life - 1e-12:
        dur = min(float(rng.uniform(40.0, 80.0)) / scale, life - t)
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
    u = -(np.maximum(0.0, 1.0 - E)) ** 2
    for t0, dur, typ in stretches:
        if typ != "harm":
            continue
        i0 = max(0, int(np.ceil((t0 + D) / DT)))
        i1 = min(n, int(np.ceil((t0 + D + 1.0) / DT)))
        if i1 > i0:
            u[i0:i1] += -S
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
    cump = np.concatenate(([0.0], np.cumsum(np.maximum(u, 0.0)) * DT))
    cumm = np.concatenate(([0.0], np.cumsum(np.maximum(-u, 0.0)) * DT))
    I_up = np.interp(t_end, tg, cump[:-1]) - np.interp(t_start, tg, cump[:-1])
    I_um = np.interp(t_end, tg, cumm[:-1]) - np.interp(t_start, tg, cumm[:-1])
    E_start = np.interp(t_start, tg, E)
    dP = CYC * (t_end - t_start)
    from step0a.far_value import Stream
    stream = Stream(I_up - I_um, dP, E_start, t_start, t_end)
    onsets = {"busy": [], "quiet": [], "harm": []}
    for t0, dur, typ in stretches:
        p = int(np.searchsorted(t_end, t0, side="left"))
        if p < len(ts):
            onsets[typ].append(p)
    for k in onsets:
        onsets[k] = np.array(sorted(set(onsets[k])), dtype=np.int64)
    return {"stream": stream, "onsets": onsets, "E": E}


def run_item2():
    got = _load("i2")
    if got is not None:
        return got
    print("  item2 fit + 4x...", file=sys.stderr)
    # fit from revision 7 gamma-0 relabel SDs
    r7 = json.load(open(os.path.join(OUT7, "results.json")))
    points = []
    for r in r7["item2"]["rows"]:
        if r["gamma"] != 0 or not (4 <= r["band"] <= 13):
            continue
        if r["null_sd"] and r["null_sd"] > 0:
            points.append((2 ** (r["band"] - 1), r["null_sd"]))
    fit7 = _fit_b(points)
    # 4x encounters, S=4 D=120, band-10 per-life z at 1x and 4x
    ones = np.ones(KMAX + 1)
    zs = {}
    for scale in (1, 4):
        zz = []
        for seed in SEEDS:
            w = (build_world(seed, "V3", S=4.0, D=120.0) if scale == 1
                 else build_scaled(seed, 4.0, 120.0, 4))
            st = w["stream"]
            g1 = w["onsets"]["harm"]
            g2 = w["onsets"]["quiet"]
            allp = np.unique(np.concatenate([g1, g2]))
            ub = ub_all(st, allp, "phys")
            c = contrast(st, g1, g2, ones, "phys", ub, "b1", kband=10)
            ns = relabel_null(st, g1, g2, ones, "phys", ub, kband=10,
                              seed=seed * 31 + scale)
            zz.append(c / ns if ns else np.nan)
        zs[scale] = zz
    m1 = float(np.nanmean(zs[1]))
    m4 = float(np.nanmean(zs[4]))
    out = {"fit_gamma0_out7": fit7,
           "z_band10_1x": m1, "z_band10_4x": m4,
           "ratio_4x_over_1x": m4 / m1 if m1 != 0 else None,
           "z_band10_1x_lives": [float(x) for x in zs[1]],
           "z_band10_4x_lives": [float(x) for x in zs[4]]}
    _save("i2", out)
    return out


# ---------------------------------------------------------------------------
# item 3: density decomposition
# ---------------------------------------------------------------------------

def _pulse_distance(st, stretches, D):
    """Phi distance from each harm cue to its pulse start."""
    out = []
    for t0, dur, typ in stretches:
        if typ != "harm":
            continue
        tp = t0 + D
        if tp >= st.t_end[-1]:
            continue
        p = int(np.searchsorted(st.t_end, tp, side="left"))
        j = int(np.searchsorted(st.t_end, t0, side="left"))
        if p < st.M and p > j:
            out.append((j, p, p - j))
    return out


def run_item3():
    got = _load("i3")
    if got is not None:
        return got
    print("  item3 density...", file=sys.stderr)
    variants = [("rho5", 5.0, False), ("rho20", 20.0, False),
                ("rho80", 80.0, False), ("cluster", 5.0, False)]
    rows = []
    totals = []
    for tag, rho_h, cluster in variants:
        acc = {}
        tot = {"shrunk": [], "unshrunk": [], "physical": []}
        for seed in SEEDS:
            w = build_world(seed, "density", S=1.0, D=30.0, rho_h=rho_h,
                            cluster=(tag == "cluster"), clean=True)
            st = w["stream"]
            ons = w["onsets"]["harm"]
            R = Y_table(st, ons)
            v1, p1 = shrunk_v(R, np.arange(len(ons)), 1.0)
            v0, p0 = shrunk_v(R, np.arange(len(ons)), 1e-9)
            tot["shrunk"].append(v1)
            tot["unshrunk"].append(v0)
            # physical-time bands
            mdP = st.median_dP
            M = st.M
            for k in range(1, KMAX + 1):
                q = 2 ** k
                half = 2 ** (k - 1)
                lo, hi = half * mdP, q * mdP
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
                acc.setdefault(("phys", k), []).append(
                    float(np.sum(Ys)) / (1.0 + N) if N else 0.0)
            tot["physical"].append(sum(
                acc[("phys", k)][-1] for k in range(1, KMAX + 1)))
            # A_k
            pulses = _pulse_distance(st, w["stretches"], 30.0)
            for k in range(1, KMAX + 1):
                A = [a for _j, _p, a in pulses if 2 ** (k - 1) < a <= 2 ** k]
                acc.setdefault(("A", k), []).append(
                    float(np.mean(A)) if A else None)
                acc.setdefault(("W", k), []).append(
                    int(np.sum(ons <= M - 2 ** k - 1)))
                r = R[k]
                m = ~np.isnan(r)
                acc.setdefault(("Vsh", k), []).append(
                    float(np.sum(r[m])) / (1.0 + int(m.sum())) if m.any() else 0.0)
                acc.setdefault(("Vun", k), []).append(
                    float(np.mean(r[m])) if m.any() else 0.0)
            for k in range(1, KMAX + 1):
                acc.setdefault(("Vp", k), []).append(acc[("phys", k)][-1])
        def avg(key):
            v = [x for x in acc.get(key, []) if x is not None]
            return float(np.mean(v)) if v else None
        vs = sum(avg(("Vsh", k)) for k in range(1, KMAX + 1))
        vu = sum(avg(("Vun", k)) for k in range(1, KMAX + 1))
        vp = float(np.mean(tot["physical"]))
        for k in range(1, KMAX + 1):
            rows.append({"variant": tag, "band": k,
                         "A": avg(("A", k)), "W": avg(("W", k)),
                         "V_shrunk": avg(("Vsh", k)),
                         "V_unshrunk": avg(("Vun", k)),
                         "V_physical": avg(("Vp", k))})
        totals.append({"variant": tag,
                       "V_shrunk_total": vs, "V_unshrunk_total": vu,
                       "V_physical_total": vp})
    out = {"rows": rows, "totals": totals}
    _save("i3", out)
    return out


# ---------------------------------------------------------------------------
# item 4: V1b 60 lives
# ---------------------------------------------------------------------------

def run_item4():
    got = _load("i4")
    if got is not None:
        return got
    print("  item4 V1b 60 lives...", file=sys.stderr)
    t0 = time.time()
    vals = {("g0", k): [] for k in list(range(1, KMAX + 1)) + [0]}
    vals.update({("tau1", k): [] for k in list(range(1, KMAX + 1)) + [0]})
    for seed in SEEDS60:
        w = build_world(seed, "V1b")
        st = w["stream"]
        g1 = w["onsets"]["busy"]
        g2 = w["onsets"]["quiet"]
        allp = np.unique(np.concatenate([g1, g2]))
        sig = sigma_k(st)
        R = R_table(st, allp, "phys")
        i1 = np.searchsorted(allp, g1)
        i2 = np.searchsorted(allp, g2)
        ones = np.ones(KMAX + 1)
        ub = ub_all(st, allp, "phys")
        t1, p1 = scale_v(R, i1, sig, 1.0)
        t2, p2 = scale_v(R, i2, sig, 1.0)
        for k in range(1, KMAX + 1):
            vals[("g0", k)].append(contrast(st, g1, g2, ones, "phys", ub, "b1", kband=k))
            vals[("tau1", k)].append(p1[k] - p2[k])
        vals[("g0", 0)].append(contrast(st, g1, g2, ones, "phys", ub, "b1"))
        vals[("tau1", 0)].append(t1 - t2)
    rows = []
    for name in ("g0", "tau1"):
        for k in list(range(1, KMAX + 1)) + [0]:
            rows.append({"readout": name, "band": k,
                         **stats(vals[(name, k)])})
    out = {"rows": rows, "seconds": time.time() - t0}
    _save("i4", out)
    return out


# ---------------------------------------------------------------------------
# tables / main
# ---------------------------------------------------------------------------

def write_tables(path, res):
    with open(path, "w") as f:
        f.write("# revision 8 tables\n\n")
        f.write("## Item 1: absolute-scale precision\n\n")
        f.write("| variant | tau | band | mean | SD | 95% CI | relabel SD |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in res["item1"]["rows"]:
            f.write(f"| {r['variant']} | {r['tau']} | {r['band']} | "
                    f"{_fmt(r['mean'],4)} | {_fmt(r['sd'],4)} | "
                    f"[{_fmt(r['ci_lo'],4)}, {_fmt(r['ci_hi'],4)}] | "
                    f"{_fmt(r['null_sd'],4)} |\n")
        f.write("\nTotal z:\n\n| variant | tau | mean z | frac z < -2.33 |\n")
        f.write("|---|---|---|---|\n")
        for z in res["item1"]["z"]:
            f.write(f"| {z['variant']} | {z['tau']} | {_fmt(z['mean_z'],4)} | "
                    f"{_fmt(z['frac_z_lt_-2.33'],4)} |\n")
        f.write("\n## Item 2: noise vs width\n\n")
        fit = res["item2"]["fit_gamma0_out7"]
        f.write(f"Fit log SD_k = a + b log(width_k) over bands 4-13: "
                f"b = {fit['b']:.4f}, 95% CI [{fit['ci_lo']:.4f}, "
                f"{fit['ci_hi']:.4f}], n = {fit['n']}.\n\n")
        f.write(f"Band-10 per-life z, S=4 D=120: 1x {res['item2']['z_band10_1x']:.4f}, "
                f"4x {res['item2']['z_band10_4x']:.4f}, ratio "
                f"{_fmt(res['item2']['ratio_4x_over_1x'],4)}.\n")
        f.write("\n## Item 3: density decomposition\n\n")
        f.write("| variant | band | A_k | W_k | V shrunk | V unshrunk | V physical |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in res["item3"]["rows"]:
            f.write(f"| {r['variant']} | {r['band']} | {_fmt(r['A'],3)} | "
                    f"{_fmt(r['W'],2)} | {_fmt(r['V_shrunk'],5)} | "
                    f"{_fmt(r['V_unshrunk'],5)} | {_fmt(r['V_physical'],5)} |\n")
        f.write("\nTotals:\n\n| variant | shrunk | unshrunk | physical |\n")
        f.write("|---|---|---|---|\n")
        for t in res["item3"]["totals"]:
            f.write(f"| {t['variant']} | {_fmt(t['V_shrunk_total'],5)} | "
                    f"{_fmt(t['V_unshrunk_total'],5)} | "
                    f"{_fmt(t['V_physical_total'],5)} |\n")
        f.write("\n## Item 4: V1b 60 lives\n\n")
        f.write("| readout | band | mean | SD | 95% CI |\n")
        f.write("|---|---|---|---|---|\n")
        for r in res["item4"]["rows"]:
            f.write(f"| {r['readout']} | {r['band']} | {_fmt(r['mean'],5)} | "
                    f"{_fmt(r['sd'],5)} | [{_fmt(r['ci_lo'],5)}, "
                    f"{_fmt(r['ci_hi'],5)}] |\n")


def main():
    os.makedirs(OUT8, exist_ok=True)
    item1 = run_item1()
    item2 = run_item2()
    item3 = run_item3()
    item4 = run_item4()
    res = {"meta": {"design": "revision 8 (20 Nov 2026)",
                    "dt": DT, "life": LIFE, "seeds": SEEDS,
                    "seeds60": SEEDS60},
           "item1": item1, "item2": item2, "item3": item3, "item4": item4}
    with open(os.path.join(OUT8, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_tables(os.path.join(OUT8, "tables.md"), res)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
