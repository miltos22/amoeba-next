"""Follow-up 10: P_raw open bands, random-pulse must-not-learn, density."""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

from step0a.far_value import Stream
from step0a.rigs.run_item3b import DT, CYC, T975_19, stats
from step0a.rigs.run_item3d import build_two
from step0a.world_signals import _ou_on_grid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT10 = os.path.join(ROOT, "runs", "out10")
LIFE = 20000.0
SEEDS = list(range(20))
TAUS = (0.25, 1.0, 4.0)


def _part(name):
    return os.path.join(OUT10, f"part_{name}.json")


def _save(name, obj):
    os.makedirs(OUT10, exist_ok=True)
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
# P_raw open-band readout
# ---------------------------------------------------------------------------

def praw_K(st):
    c = st.median_dP
    span = float(st.prefDP[st.M])
    k = 1
    K = 0
    while 2 ** (k - 1) * c <= span and k < 80:
        K = k
        k += 1
    return K


def praw_R(st, K):
    """R_ik for every record as anchor, P_raw bands, B_phys."""
    M = st.M
    c = st.median_dP
    bases = st.prefDP[1:M + 1]
    bins = st.bin
    out = {}
    sig = np.zeros(K + 1)
    for k in range(1, K + 1):
        lo = 2 ** (k - 1) * c
        hi = 2 ** k * c
        a0 = np.searchsorted(st.prefDP, bases + lo, side="right")
        a1 = np.searchsorted(st.prefDP, bases + hi, side="right")
        mature = ((bases + hi <= st.prefDP[M]) & (a1 > a0)
                  & (a0 < M) & (a1 <= M))
        Y = np.full(M, np.nan)
        dP = np.full(M, np.nan)
        Y[mature] = st.prefU[a1[mature]] - st.prefU[a0[mature]]
        dP[mature] = st.prefDP[a1[mature]] - st.prefDP[a0[mature]]
        Y0 = np.where(mature, Y, 0.0)
        dP0 = np.where(mature, dP, 0.0)
        PB = np.zeros((4, M + 1))
        DB = np.zeros((4, M + 1))
        for bb in range(4):
            m = (bins == bb)
            PB[bb, 1:] = np.cumsum(np.where(m, Y0, 0.0))
            DB[bb, 1:] = np.cumsum(np.where(m, dP0, 0.0))
        PtY = np.concatenate(([0.0], np.cumsum(Y0)))
        PtD = np.concatenate(([0.0], np.cumsum(dP0)))
        half = 2 ** (k - 1)
        i = np.arange(M, dtype=np.int64)
        left = np.minimum(i - half, M - 1)
        left_ok = left >= 0
        li = np.clip(left + 1, 0, M)
        rs = i + half
        right_ok = rs <= M - 1
        ri = np.clip(rs, 0, M)
        sy = (np.where(left_ok, PB[bins, li], 0.0)
              + np.where(right_ok, PB[bins, M] - PB[bins, ri], 0.0))
        sd = (np.where(left_ok, DB[bins, li], 0.0)
              + np.where(right_ok, DB[bins, M] - DB[bins, ri], 0.0))
        sya = (np.where(left_ok, PtY[li], 0.0)
               + np.where(right_ok, PtY[M] - PtY[ri], 0.0))
        sda = (np.where(left_ok, PtD[li], 0.0)
               + np.where(right_ok, PtD[M] - PtD[ri], 0.0))
        ub_all = np.where(sda > 0, sya / np.maximum(sda, 1e-300), 0.0)
        D0 = half * c * 10.0
        den = sd + D0
        ub = np.where(den > 0, (sy + D0 * ub_all) / np.maximum(den, 1e-300), 0.0)
        R = np.where(mature, Y - ub * dP, np.nan)
        out[k] = R
        if np.any(mature):
            med = float(np.median(R[mature]))
            sig[k] = 1.4826 * float(np.median(np.abs(R[mature] - med)))
    return out, sig


def scale_open(Rs, sigs, idx, tau, K, n0=1.0):
    per = {}
    total = 0.0
    for k in range(1, K + 1):
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


def praw_life(sts, K):
    R = {}
    S = {}
    for name, st in (("E", sts[0]), ("M", sts[1]), ("C", sts[2])):
        R[name], S[name] = praw_R(st, K)
    return R, S


def life_z(sts, onsets, tau, K, reps=200, seed=0, RS=None):
    """Per-life z cells for E/M bands and E/M/2ch/1ch totals (vectorised splits)."""
    if RS is None:
        R, S = praw_life(sts, K)
    else:
        R, S = RS
    g1, g2 = onsets["harm"], onsets["quiet"]
    master = np.unique(np.concatenate([g1, g2]))
    i1 = np.searchsorted(master, g1)
    i2 = np.searchsorted(master, g2)
    U = len(master)
    n1 = len(i1)

    # observed contrasts
    o = {}
    for name in ("E", "M", "C"):
        Rs, sg = [R[name]], [S[name]]
        ta, pa = scale_open(Rs, sg, i1, tau, K)
        tb, pb = scale_open(Rs, sg, i2, tau, K)
        o[name] = (ta - tb, {k: pa[k] - pb[k] for k in range(1, K + 1)})
    o["2ch"] = (o["E"][0] + o["M"][0],
                {k: o["E"][1][k] + o["M"][1][k] for k in range(1, K + 1)})
    o["1ch"] = o["C"]

    # random group-1 masks, same for all bands/channels/taus
    rng = np.random.default_rng(seed)
    masks = np.zeros((reps, U))
    for r in range(reps):
        perm = rng.permutation(U)
        masks[r, perm[:n1]] = 1.0

    names = ("E", "M", "C", "2ch", "1ch")
    rep = {name: {k: np.zeros(reps) for k in list(range(1, K + 1)) + [0]}
           for name in names}
    valid_total = {}
    for name in ("E", "M", "C"):
        for k in range(1, K + 1):
            rk = R[name][k][master]
            if name == "E":
                valid_total[k] = (~np.isnan(rk)).astype(np.float64)
            rk0 = np.nan_to_num(rk, nan=0.0)
            tot = float(np.sum(rk0))
            s1 = masks @ rk0
            c1 = masks @ valid_total[k]
            c2 = float(valid_total[k].sum()) - c1
            d = S[name][k] ** 2 / (tau ** 2)
            v1 = np.where(c1 > 0, s1 / np.maximum(d + c1, 1e-300), 0.0)
            v2 = np.where(c2 > 0, (tot - s1) / np.maximum(d + c2, 1e-300), 0.0)
            con = v1 - v2
            rep[name][k] = con
            if name != "C":
                rep["2ch"][k] = rep["2ch"][k] + con
        rep[name][0] = np.sum([rep[name][k] for k in range(1, K + 1)], axis=0)
        if name != "C":
            rep["2ch"][0] = rep["2ch"][0] + rep[name][0]
    rep["1ch"] = rep["C"]

    z = {name: {} for name in names}
    for name in names:
        for k in rep[name]:
            sd = float(np.std(rep[name][k], ddof=1))
            z[name][k] = (o[name][1].get(k, o[name][0]) / sd if sd else np.nan)
    zE = {k: z["E"][k] for k in range(1, K + 1)}
    zM = {k: z["M"][k] for k in range(1, K + 1)}
    totals = {name: z[name][0] for name in names}
    return zE, zM, totals, R


# ---------------------------------------------------------------------------
# two-channel P_raw life
# ---------------------------------------------------------------------------

def run_item1():
    got = _load("i1")
    if got is not None:
        return got
    print("  item1 P_raw open bands...", file=sys.stderr)
    variants = [(D, S) for D in (30, 120, 480, 1920) for S in (1.0, 4.0)]
    rows = []
    for D, S in variants:
        per = {}
        Ks = []
        for seed in SEEDS:
            w = build_two(seed, S, D, "harm", LIFE)
            sts = (w["UE"], w["UM"], w["UC"])
            K = praw_K(w["UC"])
            Ks.append(K)
            RS = praw_life(sts, K)
            for tau in TAUS:
                zE, zM, totals, _ = life_z(sts, w["onsets"], tau, K,
                                           seed=seed * 53 + int(tau * 10),
                                           RS=RS)
                for k, v in zE.items():
                    per.setdefault(("E", tau, k), []).append(v)
                for k, v in zM.items():
                    per.setdefault(("M", tau, k), []).append(v)
                for name, v in totals.items():
                    per.setdefault((name, tau, 0), []).append(v)
        new_rows = []
        Kmax = max(Ks)
        for tau in TAUS:
            for k in range(1, Kmax + 1):
                for ch in ("E", "M"):
                    vals = np.asarray(per.get((ch, tau, k), []), dtype=float)
                    vals = vals[~np.isnan(vals)]
                    if len(vals):
                        new_rows.append({"D": D, "S": S, "tau": tau,
                                         "readout": ch, "band": k,
                                         "mean_z": float(np.mean(vals)),
                                         "frac_lt": float(np.mean(vals < -2.33)),
                                         "frac_gt": float(np.mean(vals > 2.33))})
            for name in ("E", "M", "2ch", "1ch"):
                vals = np.asarray(per.get((name, tau, 0), []), dtype=float)
                vals = vals[~np.isnan(vals)]
                new_rows.append({"D": D, "S": S, "tau": tau,
                                 "readout": name, "band": 0,
                                 "mean_z": float(np.mean(vals)),
                                 "frac_lt": float(np.mean(vals < -2.33)),
                                 "frac_gt": float(np.mean(vals > 2.33))})
        rows.extend(new_rows)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# item 2: random pulses
# ---------------------------------------------------------------------------

def build_random(seed, S, life=LIFE):
    rng = np.random.default_rng(seed * 10007 + 2)
    stretches = []
    t = 0.0
    while t < life - 1e-12:
        dur = min(float(rng.uniform(40.0, 80.0)), life - t)
        u = float(rng.random())
        typ = "busy" if u < 0.45 else ("quiet" if u < 0.90 else "harm")
        stretches.append((t, dur, typ))
        t += dur
    n_harm = sum(1 for _, _, typ in stretches if typ == "harm")
    rngp = np.random.default_rng(seed * 10007 + 321)
    pulses = np.sort(rngp.uniform(0.0, life - 1.0, n_harm))
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
    meal_idx = sorted({int(round(tm / DT)) for tm in meals})
    starts = sorted({int(round(t0 / DT)) for t0, _, _ in stretches})
    events = sorted(set([0] + starts + meal_idx + [n]))
    E = np.empty(n)
    E[0] = 0.8
    meal_set = set(meal_idx)
    for a, b in zip(events[:-1], events[1:]):
        if a >= n:
            break
        base = E[a]
        E[a:b] = np.maximum(0.0, base - 0.001 * DT * np.arange(b - a))
        if b >= n:
            break
        e = E[b - 1]
        if b in meal_set:
            e = min(1.0, e + 0.3)
        E[b] = e
    uE = -(np.maximum(0.0, 1.0 - E)) ** 2
    m = _ou_on_grid(n, 0.01, 5.0, seed * 10007 + 555, DT)
    for tp in pulses:
        i0 = max(0, int(np.ceil(tp / DT)))
        i1 = min(n, int(np.ceil((tp + 1.0) / DT)))
        if i1 > i0:
            m[i0:i1] += S
    uM = -np.maximum(m, 0.0)
    segs = [(t0, t0 + dur, 20.0 if typ == "busy" else 5.0)
            for t0, dur, typ in stretches]
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
    UE, UM = integ(uE), integ(uM)
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
            "onsets": onsets, "pulses": pulses}


def run_item2():
    got = _load("i2")
    if got is not None:
        return got
    print("  item2 random pulses...", file=sys.stderr)
    per = {}
    sample = None
    Ks = []
    for S in (1.0, 4.0):
        for seed in SEEDS:
            w = build_random(seed, S)
            if seed == 0 and S == 4.0:
                sample = {"pulses": [float(x) for x in w["pulses"][:10]],
                          "harm": [float(w["UC"].t_end[p])
                                   for p in w["onsets"]["harm"][:10]],
                          "quiet": [float(w["UC"].t_end[p])
                                    for p in w["onsets"]["quiet"][:10]],
                          "n_pulses": int(len(w["pulses"]))}
            K = praw_K(w["UC"])
            Ks.append(K)
            RS = praw_life((w["UE"], w["UM"], w["UC"]), K)
            for tau in TAUS:
                zE, zM, totals, _ = life_z(
                    (w["UE"], w["UM"], w["UC"]), w["onsets"], tau, K,
                    seed=seed * 61 + int(tau * 10) + (0 if S == 1 else 1),
                    RS=RS)
                for k, v in zE.items():
                    per.setdefault(("E", S, tau, k), []).append(v)
                for k, v in zM.items():
                    per.setdefault(("M", S, tau, k), []).append(v)
                for name, v in totals.items():
                    per.setdefault((name, S, tau, 0), []).append(v)
    rows = []
    Kmax = max(Ks)
    for D in (30, 120, 480, 1920):
        for S in (1.0, 4.0):
            for tau in TAUS:
                for k in range(1, Kmax + 1):
                    for ch in ("E", "M"):
                        vals = np.asarray(per.get((ch, S, tau, k), []), dtype=float)
                        vals = vals[~np.isnan(vals)]
                        if len(vals):
                            rows.append({"D": D, "S": S, "tau": tau,
                                         "readout": ch, "band": k,
                                         "mean_z": float(np.mean(vals)),
                                         "frac_lt": float(np.mean(vals < -2.33)),
                                         "frac_gt": float(np.mean(vals > 2.33))})
                for name in ("E", "M", "2ch", "1ch"):
                    vals = np.asarray(per.get((name, S, tau, 0), []), dtype=float)
                    vals = vals[~np.isnan(vals)]
                    rows.append({"D": D, "S": S, "tau": tau,
                                 "readout": name, "band": 0,
                                 "mean_z": float(np.mean(vals)),
                                 "frac_lt": float(np.mean(vals < -2.33)),
                                 "frac_gt": float(np.mean(vals > 2.33))})
    return {"rows": rows, "sample_first_life": sample}


# ---------------------------------------------------------------------------
# item 3: density in the two-channel world
# ---------------------------------------------------------------------------

def build_two_density(seed, S, D, rho_h, cluster, life=LIFE):
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
    tg = np.arange(n) * DT
    meal_idx = sorted({int(round(tm / DT)) for tm in meals})
    starts = sorted({int(round(t0 / DT)) for t0, _, _ in stretches})
    events = sorted(set([0] + starts + meal_idx + [n]))
    E = np.empty(n)
    E[0] = 0.8
    meal_set = set(meal_idx)
    for a, b in zip(events[:-1], events[1:]):
        if a >= n:
            break
        base = E[a]
        E[a:b] = np.maximum(0.0, base - 0.001 * DT * np.arange(b - a))
        if b >= n:
            break
        e = E[b - 1]
        if b in meal_set:
            e = min(1.0, e + 0.3)
        E[b] = e
    uE = -(np.maximum(0.0, 1.0 - E)) ** 2
    m = _ou_on_grid(n, 0.01, 5.0, seed * 10007 + 555, DT)
    for t0, dur, typ in stretches:
        if typ != "harm":
            continue
        i0 = max(0, int(np.ceil((t0 + D) / DT)))
        i1 = min(n, int(np.ceil((t0 + D + 1.0) / DT)))
        if i1 > i0:
            m[i0:i1] += S
    uM = -np.maximum(m, 0.0)
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
    def integ(u):
        cp = np.concatenate(([0.0], np.cumsum(np.maximum(u, 0.0)) * DT))
        cm = np.concatenate(([0.0], np.cumsum(np.maximum(-u, 0.0)) * DT))
        ip = np.interp(t_end, tg, cp[:-1]) - np.interp(t_start, tg, cp[:-1])
        im = np.interp(t_end, tg, cm[:-1]) - np.interp(t_start, tg, cm[:-1])
        return ip - im
    UE, UM = integ(uE), integ(uM)
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
            "onsets": onsets}


def run_item3():
    got = _load("i3")
    if got is not None:
        return got
    print("  item3 density two-channel...", file=sys.stderr)
    variants = [("rho5", 5.0, False), ("rho20", 20.0, False),
                ("rho80", 80.0, False), ("cluster", 5.0, True)]
    rows = []
    for tag, rho_h, cluster in variants:
        vals = {}
        for seed in SEEDS:
            w = build_two_density(seed, 4.0, 480.0, rho_h, cluster)
            K = praw_K(w["UC"])
            R = {}
            sig = {}
            for name, st in (("E", w["UE"]), ("M", w["UM"]), ("C", w["UC"])):
                R[name], sig[name] = praw_R(st, K)
            g1 = w["onsets"]["harm"]
            idx = np.arange(len(g1))
            for n0, lab in ((1.0, "shrunk"), (1e-9, "unshrunk")):
                tE = scale_open([R["E"]], [sig["E"]], idx, 1.0, K, n0)[0]
                tM = scale_open([R["M"]], [sig["M"]], idx, 1.0, K, n0)[0]
                vals.setdefault((lab, "M"), []).append(tM)
                vals.setdefault((lab, "E"), []).append(tE)
                vals.setdefault((lab, "2ch"), []).append(tM + tE)
        row = {"variant": tag}
        for lab in ("shrunk", "unshrunk"):
            for name in ("M", "E", "2ch"):
                row[f"{name}_{lab}"] = float(np.mean(vals[(lab, name)]))
        rows.append(row)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# tables / main
# ---------------------------------------------------------------------------

def write_tables(path, res):
    with open(path, "w") as f:
        f.write("# revision 10 tables\n\n")
        f.write("## Item 1: P_raw open bands, per-channel\n\n")
        f.write("| D | S | tau | readout | band | mean z | frac < -2.33 | frac > +2.33 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in res["item1"]["rows"]:
            f.write(f"| {r['D']} | {r['S']} | {r['tau']} | {r['readout']} | "
                    f"{r['band']} | {_fmt(r['mean_z'],4)} | "
                    f"{_fmt(r['frac_lt'],4)} | {_fmt(r['frac_gt'],4)} |\n")

        f.write("\n## Item 2: random-pulse must-not-learn\n\n")
        sm = res["item2"]["sample_first_life"]
        f.write(f"First life (S=4): {sm['n_pulses']} pulses. First 10 pulse times: "
                + ", ".join(f"{x:.3f}" for x in sm["pulses"]) + ".\n\n")
        f.write("First 10 harm onset times: "
                + ", ".join(f"{x:.3f}" for x in sm["harm"]) + ".\n\n")
        f.write("First 10 quiet onset times: "
                + ", ".join(f"{x:.3f}" for x in sm["quiet"]) + ".\n\n")
        f.write("| D | S | tau | readout | band | mean z | frac < -2.33 | frac > +2.33 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in res["item2"]["rows"]:
            f.write(f"| {r['D']} | {r['S']} | {r['tau']} | {r['readout']} | "
                    f"{r['band']} | {_fmt(r['mean_z'],4)} | "
                    f"{_fmt(r['frac_lt'],4)} | {_fmt(r['frac_gt'],4)} |\n")

        f.write("\n## Item 3: density, two-channel, P_raw open bands\n\n")
        f.write("| variant | M shrunk | E shrunk | 2ch shrunk | M unshrunk | "
                "E unshrunk | 2ch unshrunk |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in res["item3"]["rows"]:
            f.write(f"| {r['variant']} | {_fmt(r['M_shrunk'],5)} | "
                    f"{_fmt(r['E_shrunk'],5)} | {_fmt(r['2ch_shrunk'],5)} | "
                    f"{_fmt(r['M_unshrunk'],5)} | {_fmt(r['E_unshrunk'],5)} | "
                    f"{_fmt(r['2ch_unshrunk'],5)} |\n")


def main():
    os.makedirs(OUT10, exist_ok=True)
    item1 = run_item1()
    item2 = run_item2()
    item3 = run_item3()
    res = {"meta": {"design": "revision 10 (20 Nov 2026)",
                    "dt": DT, "life": LIFE, "seeds": SEEDS},
           "item1": item1, "item2": item2, "item3": item3}
    with open(os.path.join(OUT10, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_tables(os.path.join(OUT10, "tables.md"), res)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
