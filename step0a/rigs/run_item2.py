"""Step 0A item 2: interval-censoring fixtures (H6) + the revision 2 gap.

Outputs to out3/ (results.json, tables.md, part_*.json, resumable).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from step0a.censoring import (
    TAU_CENS, atoms_from_readout, atom_impulse_flags, band_bounds,
    batch_band_bounds, mark_bearing, read_atoms,
)
from step0a.experience_time import (
    DELTA, SIGMA_HAT_P, SIGMA_MIN, Z_P, Z_REF, OMEGA_P, GridGroup,
    organ_stream, read_organ,
)
from step0a.pacemaker import Pacemaker
from step0a.receptors import Receptor, calibrate_sigma_hat, exact_filter
from step0a.traces import run_traces
from step0a.world_signals import (
    H_W, Source, UtilitySource, grid_times, look_indices, n_grid,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT3 = os.path.join(ROOT, "runs", "out3")
SEEDS = (0, 1, 2)
SCHEDULES = ("2", "5", "15", "50", "irregular")
LAMBDAS = (5.0, 50.0, 500.0, 5000.0)
S = 1.0
TAU = 0.5 * S
LIFE_SYN = 20000.0
BLOCK = 1000.0


def _part(name):
    return os.path.join(OUT3, f"part_{name}.json")


def _save(name, obj):
    os.makedirs(OUT3, exist_ok=True)
    with open(_part(name), "w") as f:
        json.dump(obj, f)


def _load(name):
    p = _part(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _regime(noisy):
    return "noisy" if noisy else "clean"


# ---------------------------------------------------------------------------
# item 1: RECEPTOR, ACTIVE with sensor noise (gap from revision 2)
# ---------------------------------------------------------------------------

def _active_noise_groups(seed, h, life, sigma_n=0.05):
    n = n_grid(life, h)
    groups = []
    for kind in ("steps", "bursts", "ramp"):
        src = Source(kind, seed=seed, sigma_n=sigma_n)
        x = Receptor(src, 0.2).filtered(n, h)
        sig = calibrate_sigma_hat(src, tau=0.2, h=h)
        groups.append(GridGroup(kind, x[None, :], np.array([sig]),
                                np.array([SIGMA_MIN])))
    return groups, n


def run_item1():
    life = 20000.0
    rows = []
    for seed in SEEDS:
        hs = [H_W, H_W / 2.0] if seed == 0 else [H_W]
        for h in hs:
            groups, n = _active_noise_groups(seed, h, life)
            for noisy in (False, True):
                pm = Pacemaker(noisy=noisy, seed=seed)
                organ = organ_stream(pm, groups, n, h)
                base = None
                rr = []
                for sched in SCHEDULES:
                    idx = look_indices(sched, n, seed=0, h=h)
                    ro = read_organ(organ, idx)
                    rate = float(ro["phi"]) / (float(idx[-1]) * h)
                    rr.append((sched, rate))
                    if sched == "50":
                        base = rate
                for sched, rate in rr:
                    rows.append({
                        "case": "active_sensor_noise", "seed": seed,
                        "h_tag": f"hdiv{int(round(H_W / h))}",
                        "regime": _regime(noisy), "schedule": sched,
                        "rate": rate,
                        "err": abs(rate - base) / base,
                    })
    agg = []
    for h_tag in ("hdiv1", "hdiv2"):
        for reg in ("clean", "noisy"):
            for sched in SCHEDULES:
                sel = [r for r in rows if r["h_tag"] == h_tag
                       and r["regime"] == reg and r["schedule"] == sched]
                if not sel:
                    continue
                tol = 0.05 if reg == "clean" else 0.10
                agg.append({
                    "h_tag": h_tag, "regime": reg, "schedule": sched,
                    "seeds": [r["seed"] for r in sel],
                    "rate_mean": float(np.mean([r["rate"] for r in sel])),
                    "err_mean": float(np.mean([r["err"] for r in sel])),
                    "err_max": float(np.max([r["err"] for r in sel])),
                    "tolerance": tol,
                    "pass": bool(np.max([r["err"] for r in sel]) <= tol),
                })
    return {"rows": rows, "aggregate": agg}


# ---------------------------------------------------------------------------
# item 2: H6(a)
# ---------------------------------------------------------------------------

def run_item2():
    ends = np.array([105.0])
    ip = np.array([0.0])
    im = np.array([S])
    a, b = 102.0, 104.0
    lo, hi, _, _ = band_bounds(a, b, ends, ip, im)
    positions = np.linspace(100.01, 104.99, 101)
    inside = 0
    for p in positions:
        true = -S if (a < p <= b) else 0.0
        if lo - 1e-12 <= true <= hi + 1e-12:
            inside += 1
    broken = -S * (b - a) / (ends[0] - 100.0)
    wrong = 0
    for p in positions:
        true = -S if (a < p <= b) else 0.0
        if abs(broken - true) > 0.1 * S:
            wrong += 1
    return {"atom": [100.0, 105.0], "band": [a, b],
            "y_min": float(lo), "y_max": float(hi),
            "tau_cens": TAU, "positions": len(positions),
            "inside": int(inside), "broken_value": float(broken),
            "broken_wrong": int(wrong),
            "pass_inside": bool(inside == 101),
            "pass_broken": bool(wrong > 0)}


# ---------------------------------------------------------------------------
# item 3: H6(b)
# ---------------------------------------------------------------------------

def run_item3():
    life = 200.0
    n = n_grid(life)
    pm = Pacemaker(noisy=False, seed=0)
    organ = organ_stream(pm, [], n)
    idx = look_indices("50", n)
    u = UtilitySource([(50.0, 50.2, -5.0), (120.0, 120.5, 2.0)]).grid(n)
    cp, cm = UtilitySource.cumulative_integrals(u)
    ip, im = UtilitySource.interval_integrals(cp, cm, idx)
    normal = read_atoms(organ, idx, ip, im)
    broken = read_atoms(organ, idx, ip, im, anchor_every_look=True)
    direct_p = float(cp[idx[-1]])
    direct_m = float(cm[idx[-1]])
    coal_p = float(normal["I_plus"].sum() + normal["pending_I_plus"])
    coal_m = float(normal["I_minus"].sum() + normal["pending_I_minus"])
    rel_p = abs(coal_p - direct_p) / max(abs(direct_p), 1e-30)
    rel_m = abs(coal_m - direct_m) / max(abs(direct_m), 1e-30)
    return {"looks": int(len(idx)), "m0_looks": normal["n_m0_looks"],
            "anchors_at_m0": normal["anchors_at_m0"],
            "direct_I_plus": direct_p, "direct_I_minus": direct_m,
            "coalesced_I_plus": coal_p, "coalesced_I_minus": coal_m,
            "rel_diff_I_plus": rel_p, "rel_diff_I_minus": rel_m,
            "broken_anchors_at_m0": broken["anchors_at_m0"],
            "pass_anchors": bool(normal["anchors_at_m0"] == 0),
            "pass_conservation": bool(rel_p < 1e-9 and rel_m < 1e-9),
            "pass_broken": bool(broken["anchors_at_m0"] > 0)}


# ---------------------------------------------------------------------------
# item 4: traces
# ---------------------------------------------------------------------------

def _quiet_organ(life):
    n = n_grid(life)
    pm = Pacemaker(noisy=False, seed=0)
    return organ_stream(pm, [], n), n


def _trace_read(organ, n, schedule, u):
    idx = look_indices(schedule, n)
    cp, cm = UtilitySource.cumulative_integrals(u)
    ip, im = UtilitySource.interval_integrals(cp, cm, idx)
    ro = read_organ(organ, idx, utility_ip=ip, utility_im=im,
                    record_anchors=True)
    an = ro["anchors"]
    u_anchor = u[an["grid_index"]]
    x0 = np.zeros((len(an["m"]), 0))
    tr = run_traces(an, x0, u_anchor, lambdas=LAMBDAS)
    return an, tr, u


def run_trace_cases():
    rows = []
    # Case 1: constant u = -0.01 for D, then quiet
    for D in (100.0, 2000.0):
        life = D + 50.0
        organ, n = _quiet_organ(life)
        u = UtilitySource([(0.0, D, -0.01)]).grid(n)
        out = {}
        for sched in ("2", "50"):
            an, tr, _ = _trace_read(organ, n, sched, u)
            k = int(np.searchsorted(an["time"], D, side="left"))
            out[sched] = (an, tr, k)
        for a, lam in enumerate(LAMBDAS):
            v2 = float(out["2"][1]["T_U_history"][out["2"][2], a])
            v50 = float(out["50"][1]["T_U_history"][out["50"][2], a])
            rows.append({"case": f"case1_D{int(D)}", "lambda": lam,
                         "T_U_2": v2, "T_U_50": v50,
                         "rel_diff": abs(v2 - v50) / abs(v50) if v50 != 0 else None,
                         "pass": bool(v50 != 0 and abs(v2 - v50) / abs(v50) <= 0.05)})
    # Case 2: 600 s quiet, one shock, 10 s quiet
    shock0, shock1, after = 600.03, 600.23, 10.0
    life = shock1 + after
    organ, n = _quiet_organ(life)
    u = UtilitySource([(shock0, shock1, -5.0)]).grid(n)
    out = {}
    for sched in ("2", "50"):
        an, tr, _ = _trace_read(organ, n, sched, u)
        out[sched] = (an, tr)
    for a, lam in enumerate(LAMBDAS):
        rec = {}
        for sched in ("2", "50"):
            an, tr = out[sched]
            v = float(tr["T_U_history"][-1, a])
            cand = np.where((an["time"] > shock0) & (an["time"] < shock1 + 0.5))[0]
            eps = float(np.max(tr["eps_history"][cand, a])) if len(cand) else 0.0
            rec[sched] = (v, eps)
        v2, e2 = rec["2"]
        v50, e50 = rec["50"]
        rows.append({"case": "case2_shock", "lambda": lam,
                     "T_U_2": v2, "T_U_50": v50,
                     "eps_2": e2, "eps_50": e50,
                     "rel_diff": abs(v2 - v50) / abs(v50) if v50 != 0 else None,
                     "pass": bool(v50 != 0 and abs(v2 - v50) / abs(v50) <= 0.05)})
    return {"rows": rows,
            "shock_start": shock0, "shock_end": shock1,
            "pass_all": bool(all(r["pass"] for r in rows))}


# ---------------------------------------------------------------------------
# item 5 + item 4 m distribution: synthetic life
# ---------------------------------------------------------------------------

def _synthetic_events(seed, n, h):
    rng = np.random.default_rng(seed * 7919 + 17)
    u = np.full(n, -0.01, dtype=np.float64)
    events = []
    t = 0.0
    while True:
        t += float(rng.exponential(120.0))
        if t >= LIFE_SYN:
            break
        i0 = int(round(t / h))
        i1 = int(round((t + 0.2) / h))
        u[i0:min(i1, n)] += -S / 0.2
        events.append((t, t + 0.2, -S / 0.2))
    t = 0.0
    while True:
        t += float(rng.exponential(120.0))
        if t >= LIFE_SYN:
            break
        i0 = int(round(t / h))
        i1 = int(round((t + 0.5) / h))
        u[i0:min(i1, n)] += S / 0.5
        events.append((t, t + 0.5, S / 0.5))
    return u, events


def _synthetic_groups(seed, n, h):
    t = grid_times(n, h)
    block = (t // BLOCK).astype(np.int64)
    quiet_mask = (block % 2 == 0)
    active_mask = ~quiet_mask
    a = np.exp(-h / 0.2)
    b = 1.0 - a
    groups = []
    for g in range(10):
        src = Source("silent", seed=seed * 100 + g, sigma_n=0.2)
        raw = src.to_grid(n)
        x = exact_filter(raw * quiet_mask, a, b)
        sig = calibrate_sigma_hat(src, tau=0.2, h=h)
        groups.append(GridGroup(f"quiet_noise{g}", x[None, :],
                                np.array([sig]), np.array([SIGMA_MIN])))
    for kind in ("steps", "bursts", "ramp"):
        src = Source(kind, seed=seed)
        raw = src.to_grid(n)
        x = exact_filter(raw * active_mask, a, b)
        sig = calibrate_sigma_hat(src, tau=0.2, h=h)
        groups.append(GridGroup(kind, x[None, :], np.array([sig]),
                                np.array([SIGMA_MIN])))
    return groups


def run_synthetic_seed(seed):
    os.makedirs(OUT3, exist_ok=True)
    n = n_grid(LIFE_SYN)
    groups = _synthetic_groups(seed, n, H_W)
    pm = Pacemaker(noisy=False, seed=seed)
    organ = organ_stream(pm, groups, n, H_W)
    u, events = _synthetic_events(seed, n, H_W)
    cp, cm = UtilitySource.cumulative_integrals(u)

    part = {"seed": seed, "n": int(n), "phi_total": int(organ["phi"]),
            "kmax": {}, "m_stats": {}, "bands": {}, "n_anchors": {}}

    for sched in ("2", "50"):
        idx = look_indices(sched, n, seed=0, h=H_W)
        ip, im = UtilitySource.interval_integrals(cp, cm, idx)
        ro = read_organ(organ, idx, utility_ip=ip, utility_im=im,
                        record_anchors=True)
        an = ro["anchors"]
        atoms = atoms_from_readout(ro, H_W)
        atom_imp = atom_impulse_flags(atoms, events, H_W)
        # m distribution by block
        m = ro["m_look"]
        blk = ((idx * H_W) // BLOCK).astype(np.int64) % 2
        ms = {}
        for name, bval in (("quiet", 0), ("active", 1)):
            mm = m[blk == bval]
            ms[name] = {"n": int(len(mm)),
                        "median": float(np.median(mm)) if len(mm) else None,
                        "p90": float(np.percentile(mm, 90)) if len(mm) else None,
                        "max": int(mm.max()) if len(mm) else None}
        part["m_stats"][sched] = ms

        # bands for anchors in the first half
        t_anchor = atoms["t_end"]
        phi = atoms["phi_end"]
        first_half = np.where(t_anchor <= LIFE_SYN / 2.0)[0]
        phi_end_life = float(phi[-1])
        rem = phi_end_life - phi[first_half]
        kmax = int(np.floor(np.log2(float(rem.min())))) if len(rem) else 0
        part["kmax"][sched] = kmax
        part["n_anchors"][sched] = int(len(phi))
        ends = phi
        net_ip = atoms["I_plus"]
        net_im = atoms["I_minus"]
        blk_anchor = (t_anchor[first_half] // BLOCK).astype(np.int64) % 2
        bands = {}
        for k in range(1, kmax + 1):
            aa = phi[first_half] + (2.0 ** (k - 1))
            bb = phi[first_half] + (2.0 ** k)
            lo, hi, fo, la = batch_band_bounds(aa, bb, ends, net_ip, net_im)
            cen = (hi - lo) > TAU
            bear = mark_bearing(fo, la, atom_imp)
            rec = {}
            for gname, gmask in (("all", np.ones(len(cen), dtype=bool)),
                                 ("quiet_block", blk_anchor == 0),
                                 ("active_block", blk_anchor == 1)):
                for bname, bmask in (("imp", bear), ("neu", ~bear)):
                    mask = gmask & bmask
                    rec[f"{gname}_{bname}"] = int(np.sum(mask))
                    rec[f"{gname}_{bname}_cens"] = int(np.sum(mask & cen))
            bands[str(k)] = rec
        part["bands"][sched] = bands
        # keep the m arrays for pooled statistics (npy, not JSON)
        np.save(os.path.join(OUT3, f"m_{seed}_{sched}.npy"), m)
    return part


def run_item5():
    parts = []
    for seed in SEEDS:
        tag = f"synth_{seed}"
        got = _load(tag)
        if got is None:
            t0 = time.time()
            got = run_synthetic_seed(seed)
            _save(tag, got)
            print(f"  synthetic seed {seed} done in {time.time() - t0:.0f}s",
                  file=sys.stderr)
        else:
            print(f"  synthetic seed {seed}: reuse", file=sys.stderr)
        parts.append(got)

    kmax = min(min(p["kmax"].values()) for p in parts)
    agg = {}
    for sched in ("2", "50"):
        rows = []
        for k in range(1, kmax + 1):
            tot = {}
            for p in parts:
                rec = p["bands"][sched][str(k)]
                for key, val in rec.items():
                    tot[key] = tot.get(key, 0) + val
            def frac(num, den):
                return tot[num] / tot[den] if tot[den] else None
            f_imp = frac("all_imp_cens", "all_imp")
            f_neu = frac("all_neu_cens", "all_neu")
            f_imp_q = frac("quiet_block_imp_cens", "quiet_block_imp")
            f_neu_q = frac("quiet_block_neu_cens", "quiet_block_neu")
            f_imp_a = frac("active_block_imp_cens", "active_block_imp")
            f_neu_a = frac("active_block_neu_cens", "active_block_neu")
            diff = (f_imp - f_neu) if (f_imp is not None and f_neu is not None) else None
            rows.append({
                "rate_hz": int(sched), "k": k,
                "n_imp": tot["all_imp"], "n_neu": tot["all_neu"],
                "f_imp": f_imp, "f_neu": f_neu, "diff": diff,
                "f_imp_quiet": f_imp_q, "f_neu_quiet": f_neu_q,
                "f_imp_active": f_imp_a, "f_neu_active": f_neu_a,
                "pass": (None if diff is None else bool(diff <= 0.05)),
            })
        agg[sched] = rows

    # pooled m distributions (item 4)
    m_dist = {}
    for sched in ("2", "50"):
        n = n_grid(LIFE_SYN)
        idx = look_indices(sched, n, seed=0, h=H_W)
        blk_seed = ((idx * H_W) // BLOCK).astype(np.int64) % 2
        arrs = [np.load(os.path.join(OUT3, f"m_{s}_{sched}.npy"))
                for s in SEEDS]
        m = np.concatenate(arrs)
        blk = np.concatenate([blk_seed for _ in SEEDS])
        m_dist[sched] = {}
        for name, bval in (("quiet", 0), ("active", 1)):
            mm = m[blk == bval]
            m_dist[sched][name] = {
                "n": int(len(mm)),
                "median": float(np.median(mm)),
                "p90": float(np.percentile(mm, 90)),
                "max": int(mm.max()),
            }
    rule5 = {}
    for sched in ("2", "50"):
        rule5[sched] = {
            "fail_k": [r["k"] for r in agg[sched] if r["pass"] is False],
            "na_k": [r["k"] for r in agg[sched] if r["pass"] is None],
            "pass_k": [r["k"] for r in agg[sched] if r["pass"] is True],
        }
    return {"parts": parts, "aggregate": agg, "kmax": kmax,
            "m_dist": m_dist, "rule5": rule5,
            "pass_all": bool(all(r["pass"] is not False for rows in agg.values()
                                 for r in rows))}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def write_tables(path, res):
    with open(path, "w") as f:
        f.write("# revision 3 tables (item 2: H6 censoring)\n\n")
        f.write("Experience time = RECEPTOR mode (h_w = 1/150 s), delta 0.25.\n\n")

        f.write("## Item 1: RECEPTOR, ACTIVE with sensor noise (sigma_n=0.05)\n\n")
        f.write("| h | regime | schedule | rate (mean) | err mean | err max | PASS |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for x in res["item1"]["aggregate"]:
            f.write(f"| {x['h_tag']} | {x['regime']} | {x['schedule']} | "
                    f"{_fmt(x['rate_mean'],5)} | {_fmt(x['err_mean'],6)} | "
                    f"{_fmt(x['err_max'],6)} | "
                    f"{'PASS' if x['pass'] else 'FAIL'} |\n")

        i2 = res["item2"]
        f.write("\n## Item 2: H6(a)\n\n")
        f.write(f"atom {i2['atom']}, band {i2['band']}, Y bounds "
                f"[{i2['y_min']:.6f}, {i2['y_max']:.6f}], tau_cens {i2['tau_cens']}. "
                f"inside {i2['inside']}/{i2['positions']} "
                f"({'PASS' if i2['pass_inside'] else 'FAIL'}); broken "
                f"linear value {i2['broken_value']:.6f}, wrong {i2['broken_wrong']} "
                f"({'PASS' if i2['pass_broken'] else 'FAIL'}).\n")

        i3 = res["item3"]
        f.write("\n## Item 3: H6(b)\n\n")
        f.write(f"looks {i3['looks']}, m=0 looks {i3['m0_looks']}, "
                f"anchors at m=0 {i3['anchors_at_m0']} "
                f"({'PASS' if i3['pass_anchors'] else 'FAIL'}); "
                f"I_plus direct {i3['direct_I_plus']:.12f} vs coalesced "
                f"{i3['coalesced_I_plus']:.12f} (rel {i3['rel_diff_I_plus']:.3e}), "
                f"I_minus direct {i3['direct_I_minus']:.12f} vs coalesced "
                f"{i3['coalesced_I_minus']:.12f} (rel {i3['rel_diff_I_minus']:.3e}) "
                f"({'PASS' if i3['pass_conservation'] else 'FAIL'}); broken "
                f"anchors at m=0 {i3['broken_anchors_at_m0']} "
                f"({'PASS' if i3['pass_broken'] else 'FAIL'}).\n")

        f.write("\n## Item 4: traces (H6c)\n\n")
        f.write("| case | lambda | T_U 2 Hz | T_U 50 Hz | rel diff | PASS |\n")
        f.write("|---|---|---|---|---|---|\n")
        for x in res["item4"]["traces"]["rows"]:
            f.write(f"| {x['case']} | {x['lambda']} | {x['T_U_2']:.6f} | "
                    f"{x['T_U_50']:.6f} | {_fmt(x['rel_diff'],6)} | "
                    f"{'PASS' if x['pass'] else 'FAIL'} |\n")
        f.write("\nCase 2 eps at the shock (2 Hz / 50 Hz):\n\n")
        f.write("| lambda | eps 2 Hz | eps 50 Hz |\n|---|---|---|\n")
        for x in res["item4"]["traces"]["rows"]:
            if x["case"] == "case2_shock":
                f.write(f"| {x['lambda']} | {_fmt(x['eps_2'],6)} | "
                        f"{_fmt(x['eps_50'],6)} |\n")
        f.write("\nm distribution per look (pooled over seeds):\n\n")
        f.write("| rate | block | n | median | p90 | max |\n")
        f.write("|---|---|---|---|---|---|\n")
        for sched in ("2", "50"):
            for name in ("quiet", "active"):
                d = res["item4"]["m_dist"][sched][name]
                f.write(f"| {sched} Hz | {name} | {d['n']} | "
                        f"{_fmt(d['median'],3)} | {_fmt(d['p90'],3)} | "
                        f"{d['max']} |\n")

        f.write("\n## Item 5: H6(d) selective censoring\n\n")
        f.write(f"kmax = {res['item5']['kmax']}. Fractions are over "
                "first-half anchors, pooled over seeds 0-2. "
                "diff = f_imp - f_neu; PASS if diff <= 0.05.\n\n")
        for sched in ("2", "50"):
            f.write(f"### {sched} Hz\n\n")
            f.write("| k | n imp | n neutral | f cens imp | f cens neutral | diff | "
                    "PASS | f imp quiet | f neu quiet | f imp active | f neu active |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
            for x in res["item5"]["aggregate"][sched]:
                f.write(f"| {x['k']} | {x['n_imp']} | {x['n_neu']} | "
                        f"{_fmt(x['f_imp'],4)} | {_fmt(x['f_neu'],4)} | "
                        f"{_fmt(x['diff'],4)} | "
                        f"{'N/A' if x['pass'] is None else ('PASS' if x['pass'] else 'FAIL')} | "
                        f"{_fmt(x['f_imp_quiet'],4)} | {_fmt(x['f_neu_quiet'],4)} | "
                        f"{_fmt(x['f_imp_active'],4)} | {_fmt(x['f_neu_active'],4)} |\n")
            f.write("\n")


def main():
    t0 = time.time()
    os.makedirs(OUT3, exist_ok=True)
    item1 = _load("item1")
    if item1 is None:
        print("item 1...", file=sys.stderr)
        item1 = run_item1()
        _save("item1", item1)
    else:
        print("item 1: reuse", file=sys.stderr)
    item2 = _load("item2")
    if item2 is None:
        item2 = run_item2()
        _save("item2", item2)
    item3 = _load("item3")
    if item3 is None:
        item3 = run_item3()
        _save("item3", item3)
    item4_traces = _load("item4_traces")
    if item4_traces is None:
        print("item 4 traces...", file=sys.stderr)
        item4_traces = run_trace_cases()
        _save("item4_traces", item4_traces)
    item5 = _load("item5")
    if item5 is None:
        print("item 5 synthetic...", file=sys.stderr)
        item5 = run_item5()
        _save("item5", item5)
    else:
        print("item 5: reuse", file=sys.stderr)

    item4 = {"traces": item4_traces, "m_dist": item5["m_dist"]}
    res = {"meta": {"design": "revision 3 (20 Nov 2026)",
                    "h_w": H_W, "seeds": list(SEEDS),
                    "schedules": list(SCHEDULES),
                    "lambdas": list(LAMBDAS), "S": S, "tau_cens": TAU,
                    "constants": {"delta": DELTA, "Z_p": Z_P,
                                  "omega_p": OMEGA_P, "Z_ref": Z_REF,
                                  "sigma_hat_p": SIGMA_HAT_P,
                                  "sigma_min": SIGMA_MIN},
                    "total_seconds": None},
           "item1": item1, "item2": item2, "item3": item3,
           "item4": item4, "item5": item5}
    res["meta"]["total_seconds"] = time.time() - t0
    with open(os.path.join(OUT3, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_tables(os.path.join(OUT3, "tables.md"), res)
    print(f"done in {res['meta']['total_seconds']:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
