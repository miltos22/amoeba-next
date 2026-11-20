"""Follow-up 7: oracle, band precision, all-band H2, density, V1b, cost."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from step0a.far_value import (
    Stream, leverage, ubar_dict, v_b1, v_b1_fast, v_unp, v_b1_windows,
    v_maxov_windows,
)
from step0a.pacemaker import PERIODS
from step0a.rigs.run_item3 import build_h5 as build_v1_world

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT7 = os.path.join(ROOT, "runs", "out7")
LIFE = 20000.0
DT = 0.01
CYC = sum(1.0 / p for p in PERIODS)
T975_19 = 2.093024054408263
SEEDS = list(range(20))
KMAX = 14


def _part(name):
    return os.path.join(OUT7, f"part_{name}.json")


def _save(name, obj):
    os.makedirs(OUT7, exist_ok=True)
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
# world builder
# ---------------------------------------------------------------------------

def _stretches(rng, kind, life):
    out = []
    t = 0.0
    while t < life - 1e-12:
        dur = min(float(rng.uniform(40.0, 80.0)), life - t)
        if kind == "V1b":
            typ = "busy" if rng.random() < 0.5 else "quiet"
        else:
            u = float(rng.random())
            typ = "busy" if u < 0.45 else ("quiet" if u < 0.90 else "harm")
        out.append((t, dur, typ))
        t += dur
    return out


def build_world(seed, kind="V3", S=0.0, D=30.0, rho_h=5.0, cluster=False,
                meals_mode="cap", E0=0.8, clean=False, life=LIFE):
    vid = {"V3": 2, "V1b": 3, "density": 2}[kind]
    rng = np.random.default_rng(seed * 10007 + vid)
    stretches = _stretches(rng, kind, life)
    meals = []
    if not clean:
        rngm = np.random.default_rng(seed * 10007 + 777 + vid)
        tm = 0.0
        while True:
            tm += float(rngm.exponential(300.0))
            if tm >= life:
                break
            meals.append(tm)
    n = int(round(life / DT)) + 1
    tg = np.arange(n, dtype=np.float64) * DT
    u = np.zeros(n, dtype=np.float64)
    E = None
    if clean:
        pass
    else:
        decline = np.full(n, 0.001)
        for t0, dur, typ in stretches:
            i0 = int(round(t0 / DT))
            i1 = min(n, int(round((t0 + dur) / DT)))
            if i1 > i0:
                decline[i0:i1] = 0.001
        meal_idx = sorted({int(round(tm / DT)) for tm in meals
                           if 0 <= int(round(tm / DT)) < n})
        starts = sorted({int(round(t0 / DT)) for t0, _, _ in stretches})
        events = sorted(set([0] + starts + meal_idx + [n]))
        E = np.empty(n, dtype=np.float64)
        E[0] = E0
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
                if meals_mode == "linear":
                    E_prev = E_prev + 0.5 * (1.0 - E_prev)
                else:
                    E_prev = min(1.0, E_prev + 0.3)
            E[b] = E_prev
        u = -(np.maximum(0.0, 1.0 - E)) ** 2
    if S > 0:
        for t0, dur, typ in stretches:
            if typ != "harm":
                continue
            i0 = int(np.ceil((t0 + D) / DT))
            i1 = int(np.ceil((t0 + D + 1.0) / DT))
            i0 = max(0, i0)
            i1 = min(n, i1)
            if i1 > i0:
                u[i0:i1] += -S
    # rate segments
    segs = []
    for t0, dur, typ in stretches:
        t1 = t0 + dur
        if typ == "busy":
            segs.append((t0, t1, 20.0))
        elif typ == "quiet":
            segs.append((t0, t1, 5.0))
        else:
            if cluster:
                a = min(t1, t0 + 29.0)
                b = min(t1, t0 + 32.0)
                if a > t0:
                    segs.append((t0, a, 5.0))
                if b > a:
                    segs.append((a, b, 200.0))
                if t1 > b:
                    segs.append((b, t1, 5.0))
            else:
                segs.append((t0, t1, rho_h))
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
    E_start = E0 * np.ones(len(ts)) if E is None else np.interp(t_start, tg, E)
    dP = CYC * (t_end - t_start)
    stream = Stream(I_up - I_um, dP, E_start, t_start, t_end)
    onsets = {"busy": [], "quiet": [], "harm": []}
    for t0, dur, typ in stretches:
        p = int(np.searchsorted(t_end, t0, side="left"))
        if p < len(ts):
            onsets[typ].append(p)
    for k in onsets:
        onsets[k] = np.array(sorted(set(onsets[k])), dtype=np.int64)
    return {"stream": stream, "onsets": onsets, "meals": meals,
            "stretches": stretches, "u": u, "tg": tg, "E": E}


# ---------------------------------------------------------------------------
# sigma_k (all records as anchors, B_phys)
# ---------------------------------------------------------------------------

def _ubar_all_phys(stream, k):
    q = 2 ** k
    half = 2 ** (k - 1)
    Mv = stream.M - q
    i = np.arange(Mv, dtype=np.int64)
    Y = stream.prefU[i + q + 1] - stream.prefU[i + half + 1]
    dP = stream.prefDP[i + q + 1] - stream.prefDP[i + half + 1]
    b = stream.bin[:Mv]
    PB = np.zeros((4, Mv + 1))
    DB = np.zeros((4, Mv + 1))
    for bb in range(4):
        m = (b == bb)
        PB[bb, 1:] = np.cumsum(np.where(m, Y, 0.0))
        DB[bb, 1:] = np.cumsum(np.where(m, dP, 0.0))
    PtY = np.concatenate(([0.0], np.cumsum(Y)))
    PtD = np.concatenate(([0.0], np.cumsum(dP)))
    left = np.minimum(i - half, Mv - 1)
    left_ok = left >= 0
    li = np.clip(left + 1, 0, Mv)
    right_start = i + half
    right_ok = right_start <= Mv - 1
    ri = np.clip(right_start, 0, Mv)
    sy = (np.where(left_ok, PB[b, li], 0.0)
          + np.where(right_ok, PB[b, Mv] - PB[b, ri], 0.0))
    sd = (np.where(left_ok, DB[b, li], 0.0)
          + np.where(right_ok, DB[b, Mv] - DB[b, ri], 0.0))
    sya = (np.where(left_ok, PtY[li], 0.0)
           + np.where(right_ok, PtY[Mv] - PtY[ri], 0.0))
    sda = (np.where(left_ok, PtD[li], 0.0)
           + np.where(right_ok, PtD[Mv] - PtD[ri], 0.0))
    ub_all = np.where(sda > 0, sya / np.maximum(sda, 1e-300), 0.0)
    D0 = half * stream.median_dP * 10.0
    den = sd + D0
    ub = np.where(den > 0, (sy + D0 * ub_all) / np.maximum(den, 1e-300), 0.0)
    R = Y - ub * dP
    return R


def sigma_k(stream):
    sig = np.zeros(KMAX + 1)
    for k in range(1, KMAX + 1):
        if stream.M - 2 ** k <= 1:
            sig[k] = 0.0
            continue
        R = _ubar_all_phys(stream, k)
        if len(R) == 0:
            sig[k] = 0.0
            continue
        med = float(np.median(R))
        sig[k] = 1.4826 * float(np.median(np.abs(R - med)))
    return sig


def precision(sig, gamma):
    p = np.ones(KMAX + 1)
    if gamma == 0:
        return p
    s1 = sig[1] if sig[1] > 0 else 1e-12
    for k in range(1, KMAX + 1):
        p[k] = (s1 / max(sig[k], 1e-12)) ** gamma
    return p


# ---------------------------------------------------------------------------
# readout helpers with per-band prec
# ---------------------------------------------------------------------------

def _pairs(positions, prec):
    pos = np.asarray(positions, dtype=np.int64)
    if len(pos) == 0:
        return pos, np.empty(0, dtype=np.int64), np.empty(0)
    anchors = np.repeat(pos, KMAX)
    ks = np.tile(np.arange(1, KMAX + 1, dtype=np.int64), len(pos))
    w = prec[ks]
    return anchors, ks, w


def group_value(stream, positions, prec, mode, ub, readout, kband=None,
                n0=1.0):
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) == 0:
        return 0.0
    if kband is None:
        anchors, ks, w = _pairs(positions, prec)
        if readout == "unp":
            return v_unp(stream, anchors, ks, w, n0, ub, mode)[0]
        return v_b1_fast(stream, anchors, ks, w, n0, ub, mode)
    anchors = positions
    ks = np.full(len(positions), kband, dtype=np.int64)
    w = np.full(len(positions), prec[kband])
    if readout == "unp":
        return v_unp(stream, anchors, ks, w, n0, ub, mode)[1].get(kband, 0.0)
    return v_b1(stream, anchors, ks, w, n0, ub, mode, kband=kband)


def contrast(stream, g1, g2, prec, base, ub, readout, kband=None, n0=1.0):
    return (group_value(stream, g1, prec, base, ub, readout, kband, n0)
            - group_value(stream, g2, prec, base, ub, readout, kband, n0))


def ub_all(stream, positions, mode):
    if mode == "none" or len(positions) == 0:
        return {}
    anchors, ks, _ = _pairs(positions, np.ones(KMAX + 1))
    return ubar_dict(stream, anchors, ks, mode)


def relabel_null(stream, g1, g2, prec, base, ub, kband=None, reps=200,
                 seed=0):
    rng = np.random.default_rng(seed)
    union = np.concatenate([g1, g2])
    n1 = len(g1)
    if len(union) < 2 or n1 == 0:
        return None
    vals = []
    for _ in range(reps):
        perm = rng.permutation(union)
        vals.append(contrast(stream, perm[:n1], perm[n1:], prec, base, ub,
                             "b1", kband))
    return float(np.std(vals, ddof=1))


def stats(vals):
    v = np.asarray(vals, dtype=float)
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    return {"mean": m, "sd": sd,
            "ci_lo": m - T975_19 * sd / np.sqrt(len(v)),
            "ci_hi": m + T975_19 * sd / np.sqrt(len(v))}


# ---------------------------------------------------------------------------
# item 2
# ---------------------------------------------------------------------------

def run_item2():
    variants = [("S1_D30", 1.0, 30.0), ("S4_D30", 4.0, 30.0),
                ("S1_D120", 1.0, 120.0), ("S4_D120", 4.0, 120.0),
                ("S1_D480", 1.0, 480.0), ("S4_D480", 4.0, 480.0)]
    all_rows = []
    zrows = []
    for tag, S, D in variants:
        got = _load(f"i2_{tag}")
        if got is None:
            print(f"  item2 {tag}...", file=sys.stderr)
            t0 = time.time()
            rows = []
            sig_life = []
            z = {g: [] for g in (0, 1, 2)}
            per_life = {}
            for seed in SEEDS:
                w = build_world(seed, "V3", S=S, D=D)
                st = w["stream"]
                ons = w["onsets"]
                g1, g2 = ons["harm"], ons["quiet"]
                allp = np.unique(np.concatenate([g1, g2]))
                sig = sigma_k(st)
                sig_life.append(sig)
                ub = ub_all(st, allp, "phys")
                for gamma in (0, 1, 2):
                    prec = precision(sig, gamma)
                    for k in list(range(1, 15)) + [0]:
                        per_life.setdefault((gamma, "v", k), []).append(
                            contrast(st, g1, g2, prec, "phys", ub, "b1",
                                     None if k == 0 else k))
                    for k in list(range(1, 15)) + [0]:
                        nsd = relabel_null(st, g1, g2, prec, "phys", ub,
                                           kband=(None if k == 0 else k),
                                           seed=seed * 17 + gamma * 101 + k)
                        per_life.setdefault((gamma, "n", k), []).append(nsd)
                    nsd0 = per_life[(gamma, "n", 0)][-1]
                    if nsd0 and nsd0 > 0:
                        z[gamma].append(per_life[(gamma, "v", 0)][-1] / nsd0)
                    else:
                        z[gamma].append(np.nan)
            sig_mean = np.mean(np.array(sig_life), axis=0)
            for gamma in (0, 1, 2):
                for k in list(range(1, 15)) + [0]:
                    vals = per_life[(gamma, "v", k)]
                    s = stats(vals)
                    nsd = per_life[(gamma, "n", k)]
                    rows.append({"variant": tag, "S": S, "D": D,
                                 "gamma": gamma, "band": k,
                                 **s,
                                 "null_sd": float(np.nanmean(nsd))})
                zz = np.asarray(z[gamma], dtype=float)
                zrows.append({"variant": tag, "S": S, "D": D, "gamma": gamma,
                              "mean_z": float(np.nanmean(zz)),
                              "frac_z_lt_-2.33": float(np.nanmean(zz < -2.33))})
            got = {"variant": tag, "S": S, "D": D,
                   "sigma_mean": [float(x) for x in sig_mean],
                   "rows": rows, "z": zrows[-1] if False else
                   [r for r in zrows if r["variant"] == tag],
                   "seconds": time.time() - t0}
            _save(f"i2_{tag}", got)
            print(f"  {tag}: {got['seconds']:.0f}s", file=sys.stderr)
        all_rows.extend(got["rows"])
        zrows.extend(got["z"])
    return {"rows": all_rows, "z": zrows}


# ---------------------------------------------------------------------------
# item 3
# ---------------------------------------------------------------------------

def run_item3():
    got = _load("i3")
    if got is not None:
        return got
    w = build_v1_world(100, "V1", 0.0)
    st = w[0]
    mid = 0.5 * (st.t_start + st.t_end)
    p = int(np.argmin(np.abs(mid - LIFE / 2.0)))
    rows = []
    for n in (1, 10, 100):
        ks_design = np.array([5 + (i % 10) for i in range(n)])
        js = np.array([p - (2 ** k - (i // 10))
                       for i, k in enumerate(ks_design)])
        anchors = np.repeat(js, KMAX)
        ks = np.tile(np.arange(1, KMAX + 1), n)
        weights = np.ones(len(anchors))
        for readout in ("unp", "b1"):
            lev = leverage(st, p, anchors, ks, weights, 1.0, "phys",
                           readout, broken=False)
            rows.append({"n": n, "target": p, "readout": readout, **lev})
            if readout == "b1":
                assert lev["L_direct"] <= 1.0 + 1e-9, lev
    out = {"rows": rows, "target": p, "M": int(st.M)}
    _save("i3", out)
    return out


# ---------------------------------------------------------------------------
# item 4
# ---------------------------------------------------------------------------

def run_item4():
    variants = [("rho5", 5.0, False), ("rho20", 20.0, False),
                ("rho80", 80.0, False), ("cluster", 5.0, True)]
    rows = []
    for tag, rho_h, cluster in variants:
        got = _load(f"i4_{tag}")
        if got is None:
            print(f"  item4 {tag}...", file=sys.stderr)
            t0 = time.time()
            vb, vu, ap = [], [], []
            checks_a, checks_b = [], []
            for seed in SEEDS:
                w = build_world(seed, "density", S=1.0, D=30.0,
                                rho_h=rho_h, cluster=cluster, clean=True)
                st = w["stream"]
                ons = w["onsets"]
                g = ons["harm"]
                # (a) physical integral over the record-covered interval
                cum = np.concatenate(([0.0], np.cumsum(w["u"]) * DT))
                phys = float(np.interp(st.t_end[-1], w["tg"], cum[:-1]))
                checks_a.append(abs(float(st.U.sum()) - phys)
                                / max(abs(phys), 1.0))
                # direct a_r and V_B1 (B_none)
                anchors, ks, ww = _pairs(g, np.ones(KMAX + 1))
                M = st.M
                W = {}
                for e, k, pp in zip(ww, ks, anchors):
                    if pp <= M - 2 ** k - 1:
                        W[k] = W.get(k, 0.0) + e
                dl = np.zeros(M + 1)
                for pp, k, e in zip(anchors.tolist(), ks.tolist(), ww.tolist()):
                    if pp > M - 2 ** k - 1:
                        continue
                    q = 2 ** k
                    half = 2 ** (k - 1)
                    wgt = e / (1.0 + W[k])
                    dl[pp + 1 + half] += wgt
                    dl[pp + q + 1] -= wgt
                a = np.cumsum(dl[:M])
                Vd = float(np.sum(np.minimum(1.0, a) * st.U))
                Vf = v_b1_fast(st, anchors, ks, ww, 1.0, {}, "none")
                checks_b.append(abs(Vf - Vd) / max(abs(Vd), 1.0))
                vb.append(Vf)
                vu.append(v_unp(st, anchors, ks, ww, 1.0, {}, "none")[0])
                # pulse-weighted a_p
                vals = []
                for t0, dur, typ in w["stretches"]:
                    if typ != "harm":
                        continue
                    a0 = t0 + 30.0
                    a1 = a0 + 1.0
                    m = (st.t_end > a0) & (st.t_start < a1)
                    if not np.any(m):
                        continue
                    uu = -st.U[m]
                    if uu.sum() <= 0:
                        continue
                    vals.append(float(np.sum(uu * a[m]) / np.sum(uu)))
                if vals:
                    ap.append(float(np.mean(vals)))
            got = {"variant": tag, "rho_h": rho_h, "cluster": cluster,
                   "V_B1_mean": float(np.mean(vb)),
                   "V_unp_mean": float(np.mean(vu)),
                   "a_p_mean": float(np.mean(ap)) if ap else None,
                   "max_rel_err_a": float(np.max(checks_a)),
                   "max_rel_err_b": float(np.max(checks_b)),
                   "seconds": time.time() - t0}
            _save(f"i4_{tag}", got)
            print(f"  {tag}: {got['seconds']:.0f}s", file=sys.stderr)
        rows.append(got)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# item 5
# ---------------------------------------------------------------------------

def run_item5():
    got = _load("i5")
    if got is not None:
        return got
    print("  item5 V1b...", file=sys.stderr)
    t0 = time.time()
    rows = []
    deficits = []
    gammas = (0, 2)
    bases = ("phys", "phi", "none")
    for seed in SEEDS:
        w = build_world(seed, "V1b")
        st = w["stream"]
        ons = w["onsets"]
        g1, g2 = ons["busy"], ons["quiet"]
        allp = np.unique(np.concatenate([g1, g2]))
        sig = sigma_k(st)
        deficits.append(float(np.mean(1.0 - w["E"])))
        ub = {b: ub_all(st, allp, b) for b in bases}
        for gamma in gammas:
            prec = precision(sig, gamma)
            for base in bases:
                for k in list(range(1, 15)) + [0]:
                    val = contrast(st, g1, g2, prec, base, ub[base], "b1",
                                   None if k == 0 else k)
                    nsd = relabel_null(st, g1, g2, prec, base, ub[base],
                                       kband=(None if k == 0 else k),
                                       seed=seed * 19 + gamma * 103 + k)
                    rows.append({"seed": seed, "gamma": gamma,
                                 "baseline": base, "band": k,
                                 "value": val, "null_sd": nsd})
    # aggregate
    agg = []
    for gamma in gammas:
        for base in bases:
            for k in list(range(1, 15)) + [0]:
                sel = [r for r in rows if r["gamma"] == gamma
                       and r["baseline"] == base and r["band"] == k]
                vals = [r["value"] for r in sel]
                nsd = [r["null_sd"] for r in sel if r.get("null_sd") is not None]
                s = stats(vals)
                agg.append({"gamma": gamma, "baseline": base, "band": k,
                            **s, "null_sd": float(np.mean(nsd)) if nsd else None})
    out = {"rows": agg, "deficit_mean": float(np.mean(deficits)),
           "seconds": time.time() - t0}
    _save("i5", out)
    return out


# ---------------------------------------------------------------------------
# item 6
# ---------------------------------------------------------------------------

def run_item6():
    got = _load("i6")
    if got is not None:
        return got
    w = build_world(0, "V1b")
    st = w["stream"]
    sig = sigma_k(st)
    prec = precision(sig, 2)
    rng = np.random.default_rng(0)
    pos = rng.choice(np.arange(0, st.M - 2 ** 14), size=300, replace=False)
    anchors = np.repeat(pos, KMAX)
    ks = np.tile(np.arange(1, KMAX + 1), 300)
    weights = prec[ks]
    ub = ubar_dict(st, anchors, ks, "phys")
    t0 = time.perf_counter()
    v_b1_fast(st, anchors, ks, weights, 1.0, ub, "phys")
    t1 = time.perf_counter()
    v_unp(st, anchors, ks, weights, 1.0, ub, "phys")
    t2 = time.perf_counter()
    out = {"b1_s": t1 - t0, "unp_s": t2 - t1}
    _save("i6", out)
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def write_tables(path, res):
    with open(path, "w") as f:
        f.write("# revision 7 tables\n\n")
        f.write("## Item 1: oracle\n\n")
        f.write("Expected O1-O7 values and anchor-level equality are asserted "
                "in `test_far_value_oracle.py` (`2 passed`); O2 "
                "v_maxov = 4 vs v_b1 = 6.\n")
        f.write("\n## Item 2: band precision\n\n")
        f.write("| variant | gamma | band | mean | SD | 95% CI | relabel SD |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in res["item2"]["rows"]:
            f.write(f"| {r['variant']} | {r['gamma']} | {r['band']} | "
                    f"{_fmt(r['mean'],4)} | {_fmt(r['sd'],4)} | "
                    f"[{_fmt(r['ci_lo'],4)}, {_fmt(r['ci_hi'],4)}] | "
                    f"{_fmt(r['null_sd'],4)} |\n")
        f.write("\nTotal z (contrast / relabel SD):\n\n")
        f.write("| variant | gamma | mean z | frac z < -2.33 |\n")
        f.write("|---|---|---|---|\n")
        for z in res["item2"]["z"]:
            f.write(f"| {z['variant']} | {z['gamma']} | {_fmt(z['mean_z'],4)} | "
                    f"{_fmt(z['frac_z_lt_-2.33'],4)} |\n")
        f.write("\nsigma_k means by variant:\n\n")
        f.write("| variant | k=1 | k=7 | k=14 |\n|---|---|---|---|\n")
        for r in res["item2"]["rows"]:
            pass
        # sigma means are stored per variant in the first row group; read from res
        for v, sig in res["item2"]["sigma"].items():
            f.write(f"| {v} | {_fmt(sig[1],5)} | {_fmt(sig[7],5)} | "
                    f"{_fmt(sig[14],5)} |\n")
        f.write("\n## Item 3: all-band H2\n\n")
        f.write("| n | readout | L_direct | L_total |\n|---|---|---|---|\n")
        for r in res["item3"]["rows"]:
            f.write(f"| {r['n']} | {r['readout']} | {_fmt(r['L_direct'],8)} | "
                    f"{_fmt(r['L_total'],8)} |\n")
        f.write("\n## Item 4: record density\n\n")
        f.write("| variant | V_B1 | V_unp | mean a_p | max rel err (a) | "
                "max rel err (b) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in res["item4"]["rows"]:
            f.write(f"| {r['variant']} | {_fmt(r['V_B1_mean'],6)} | "
                    f"{_fmt(r['V_unp_mean'],6)} | {_fmt(r['a_p_mean'],5)} | "
                    f"{_fmt(r['max_rel_err_a'],2)} | {_fmt(r['max_rel_err_b'],2)} |\n")
        f.write("\n## Item 5: V1b\n\n")
        f.write(f"Life-mean deficit 1 - E: {_fmt(res['item5']['deficit_mean'],5)}.\n\n")
        f.write("| gamma | baseline | band | mean | SD | 95% CI | relabel SD |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in res["item5"]["rows"]:
            f.write(f"| {r['gamma']} | {r['baseline']} | {r['band']} | "
                    f"{_fmt(r['mean'],4)} | {_fmt(r['sd'],4)} | "
                    f"[{_fmt(r['ci_lo'],4)}, {_fmt(r['ci_hi'],4)}] | "
                    f"{_fmt(r['null_sd'],4)} |\n")
        f.write("\n## Item 6: cost\n\n")
        c = res["item6"]
        f.write(f"300 anchors x 14 bands, gamma 2: B1 {c['b1_s']:.6f} s, "
                f"unpatched {c['unp_s']:.6f} s.\n")


def main():
    os.makedirs(OUT7, exist_ok=True)
    item2 = run_item2()
    # collect sigma means per variant from parts
    sig = {}
    for tag, S, D in (("S1_D30", 1, 30), ("S4_D30", 4, 30),
                      ("S1_D120", 1, 120), ("S4_D120", 4, 120),
                      ("S1_D480", 1, 480), ("S4_D480", 4, 480)):
        p = _load(f"i2_{tag}")
        if p:
            sig[tag] = p["sigma_mean"]
    item2["sigma"] = sig
    item3 = run_item3()
    item4 = run_item4()
    item5 = run_item5()
    item6 = run_item6()
    res = {"meta": {"design": "revision 7 (20 Nov 2026)",
                    "dt": DT, "life": LIFE, "seeds": SEEDS},
           "item2": item2, "item3": item3, "item4": item4,
           "item5": item5, "item6": item6}
    with open(os.path.join(OUT7, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_tables(os.path.join(OUT7, "tables.md"), res)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
