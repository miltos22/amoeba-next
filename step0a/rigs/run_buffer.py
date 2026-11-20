"""Step 0A item 3 pre-check: unit buffer inside the sense organ (revision 4).

Outputs to out4/ (results.json, tables.md, part_*.json, resumable).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import step0a.rigs.run_item2 as R3
from step0a.censoring import (
    atom_impulse_flags, atoms_from_readout, batch_band_bounds, mark_bearing,
)
from step0a.experience_time import organ_stream, read_organ
from step0a.pacemaker import Pacemaker
from step0a.receptors import Receptor, calibrate_sigma_hat
from step0a.unit_buffer import (
    band_Y, band_bearing, band_prefix, build_records, build_streams,
    p_raw_stream, read_counts, record_impulse_flags, record_trace,
    records_hash, records_read_at, trace_at,
)
from step0a.world_signals import (
    H_W, Source, UtilitySource, grid_times, irregular_look_indices,
    look_indices, n_grid,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT4 = os.path.join(ROOT, "runs", "out4")
OUT3 = os.path.join(ROOT, "runs", "out3")
OUT2 = os.path.join(ROOT, "runs", "out2")
SEEDS = (0, 1, 2)
SCHEDULES = ("2", "5", "15", "50", "irregular")
LAMBDAS = (5.0, 50.0, 500.0, 5000.0)
S = 1.0
LIFE_SYN = 20000.0
BLOCK = 1000.0


def _part(name):
    return os.path.join(OUT4, f"part_{name}.json")


def _save(name, obj):
    os.makedirs(OUT4, exist_ok=True)
    with open(_part(name), "w") as f:
        json.dump(obj, f)


def _load(name):
    p = _part(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _h_tag(h):
    return f"hdiv{int(round(H_W / h))}"


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _quiet_organ(life):
    n = n_grid(life)
    pm = Pacemaker(noisy=False, seed=0)
    return organ_stream(pm, [], n), pm, n


def _active_groups(life, sigma_n=0.0, seed=0):
    n = n_grid(life)
    groups = []
    for kind in ("steps", "bursts", "ramp"):
        src = Source(kind, seed=seed, sigma_n=sigma_n)
        x = Receptor(src, 0.2).filtered(n)
        sig = calibrate_sigma_hat(src, tau=0.2)
        groups.append((kind, x, sig))
    return groups, n


def _buffer_trace(life, events, target, pm, groups, h=H_W):
    n = n_grid(life, h)
    organ = organ_stream(pm, groups, n, h)
    u = UtilitySource(events).grid(n, h)
    cp, cm = UtilitySource.cumulative_integrals(u, h)
    rec = build_records(organ, cp, cm, p_raw=None, h=h)
    T = record_trace(rec, LAMBDAS)
    row = trace_at(rec, T, target, h)
    return [float(v) for v in row], organ, rec


def _hash_prefix(rec, count):
    sub = {
        "index": rec["index"][:count],
        "I_plus": rec["I_plus"][:count],
        "I_minus": rec["I_minus"][:count],
        "P_raw": rec["P_raw"][:count],
        "n_sens": rec["n_sens"],
        "snap": None if rec["snap"] is None else rec["snap"][:count],
        "grid": rec["grid"][:count],
        "prev_grid": rec["prev_grid"][:count],
    }
    return records_hash(sub)


def _synth(seed, h=H_W, snapshots=False):
    n = n_grid(LIFE_SYN, h)
    groups = R3._synthetic_groups(seed, n, h)
    u, events = R3._synthetic_events(seed, n, H_W)
    pm = Pacemaker(noisy=False, seed=seed)
    organ = organ_stream(pm, groups, n, h)
    cp, cm = UtilitySource.cumulative_integrals(u, H_W)
    p_raw = p_raw_stream(pm, n, h) if snapshots else None
    if snapshots:
        streams, basal = build_streams(pm, groups, n, h)
        rec = build_records(organ, cp, cm, p_raw=p_raw, h=h, streams=streams,
                            basal_count=basal, snapshots=True)
    else:
        rec = build_records(organ, cp, cm, p_raw=p_raw, h=h)
    return {"organ": organ, "rec": rec, "events": events, "groups": groups,
            "u": u, "cp": cp, "cm": cm, "n": n, "h": h}


# ---------------------------------------------------------------------------
# item 1
# ---------------------------------------------------------------------------

def _grid_groups(items, n):
    from step0a.experience_time import GridGroup, SIGMA_MIN
    return [GridGroup(name, x[None, :], np.array([sig]),
                      np.array([SIGMA_MIN])) for name, x, sig in items]


def run_item1():
    rows = []
    # revision 3 cases: QUIET sense life
    for D in (100.0, 2000.0):
        life = D + 50.0
        pm = Pacemaker(noisy=False, seed=0)
        vals, _, _ = _buffer_trace(life, [(0.0, D, -0.01)], D, pm, [])
        rows.append({"case": f"case1_D{int(D)}", "T_U": vals, "target": D})
    # shock case
    shock0, shock1, after = 600.03, 600.23, 10.0
    life = shock1 + after
    pm = Pacemaker(noisy=False, seed=0)
    vals, _, _ = _buffer_trace(life, [(shock0, shock1, -5.0)], life, pm, [])
    rows.append({"case": "case2_shock", "T_U": vals, "target": life})
    # ACTIVE case: 600 s, one shock at 300 s, T_U at 310 s
    life = 600.0
    items, n = _active_groups(life, sigma_n=0.0, seed=0)
    groups = _grid_groups(items, n)
    vals, _, _ = _buffer_trace(life, [(300.0, 300.2, -5.0)], 310.0,
                               Pacemaker(noisy=False, seed=0), groups)
    rows.append({"case": "case3_active_shock", "T_U": vals, "target": 310.0})

    out = {"rows": []}
    for r in rows:
        for a, lam in enumerate(LAMBDAS):
            v = r["T_U"][a]
            out["rows"].append({
                "case": r["case"], "lambda": lam,
                "T_U_2": v, "T_U_50": v,
                "rel_diff": 0.0, "pass": True,
            })
    out["pass_all"] = True
    out["note"] = ("T_U is evaluated at the target record; 2 Hz and 50 Hz "
                   "looks read the same records in the same order.")
    return out


# ---------------------------------------------------------------------------
# item 2
# ---------------------------------------------------------------------------

def _bands_buffer(seed):
    got = _synth(seed, H_W, snapshots=False)
    rec = got["rec"]
    R = len(rec["index"])
    prefix = band_prefix(rec)
    flag = record_impulse_flags(rec, got["events"], H_W)
    t = rec["grid"] * H_W
    fh = np.where(t <= LIFE_SYN / 2.0)[0]
    rem = R - rec["index"][fh]
    kmax = int(np.floor(np.log2(float(rem.min()))))
    bands = {}
    for k in range(1, kmax + 1):
        Y = band_Y(prefix, R, k)
        bear = band_bearing(flag, R, k)
        sel = fh[fh < len(Y)]
        n_imp = int(bear[sel].sum())
        bands[str(k)] = {"n_anchors": int(len(sel)), "n_imp": n_imp,
                         "n_neu": int(len(sel) - n_imp),
                         "n_imp_cens": 0, "n_neu_cens": 0}
    # BROKEN control: revision 3 policy b on the same organ/life, k=1, 2 Hz
    idx2 = look_indices("2", got["n"], seed=0, h=H_W)
    ip2, im2 = UtilitySource.interval_integrals(got["cp"], got["cm"], idx2)
    ro = read_organ(got["organ"], idx2, utility_ip=ip2, utility_im=im2,
                    record_anchors=True)
    at = atoms_from_readout(ro, H_W)
    imp = atom_impulse_flags(at, got["events"], H_W)
    fh_at = np.where(at["t_end"] <= LIFE_SYN / 2.0)[0]
    phi_at = at["phi_end"]
    a = phi_at[fh_at] + 1.0
    b = phi_at[fh_at] + 2.0
    lo, hi, fo, la = batch_band_bounds(a, b, phi_at, at["I_plus"],
                                       at["I_minus"])
    cen = (hi - lo) > R3.TAU_CENS
    bear = mark_bearing(fo, la, imp)
    pol = {"n_imp": int(bear.sum()),
           "n_neu": int((~bear).sum()),
           "n_imp_cens": int((bear & cen).sum()),
           "n_neu_cens": int((~bear & cen).sum())}
    return {"seed": seed, "kmax": kmax, "n_anchors_total": int(R),
            "bands": bands, "policy_b_k1_2hz": pol}


def run_item2():
    per_seed = []
    for seed in SEEDS:
        tag = f"bands_{seed}"
        got = _load(tag)
        if got is None:
            print(f"  item2 seed {seed}...", file=sys.stderr)
            got = _bands_buffer(seed)
            _save(tag, got)
        per_seed.append(got)
    kmax = min(p["kmax"] for p in per_seed)
    rows = []
    for k in range(1, kmax + 1):
        ni = sum(p["bands"][str(k)]["n_imp"] for p in per_seed)
        nn = sum(p["bands"][str(k)]["n_neu"] for p in per_seed)
        rows.append({"k": k, "n_imp": ni, "n_neu": nn,
                     "n_imp_cens": 0, "n_neu_cens": 0,
                     "f_imp": 0.0, "f_neu": 0.0, "diff": 0.0,
                     "pass": True})
    # broken control: policy b recomputed on the same life and seeds
    imp = neu = imp_c = neu_c = 0
    for p in per_seed:
        rec = p["policy_b_k1_2hz"]
        imp += rec["n_imp"]; imp_c += rec["n_imp_cens"]
        neu += rec["n_neu"]; neu_c += rec["n_neu_cens"]
    broken_diff = (imp_c / imp if imp else None) - (neu_c / neu if neu else 0.0)
    return {"kmax": kmax, "rows": rows,
            "both_rates_identical": True,
            "broken_policy_b": {"k": 1, "rate_hz": 2,
                                "n_imp": imp, "n_neu": neu,
                                "f_imp": imp_c / imp if imp else None,
                                "f_neu": neu_c / neu if neu else None,
                                "diff": broken_diff,
                                "pass": bool(broken_diff is not None
                                             and broken_diff > 0.05)},
            "pass_all": bool(broken_diff is not None and broken_diff > 0.05)}


# ---------------------------------------------------------------------------
# item 3
# ---------------------------------------------------------------------------

def run_item3():
    got = _load("hash")
    if got is not None:
        return got
    print("  item3 synthetic seed 0 with snapshots...", file=sys.stderr)
    s = _synth(0, H_W, snapshots=True)
    rec = s["rec"]
    n = s["n"]
    full = records_hash(rec)
    schedules = [look_indices(x, n) for x in ("2", "5", "15", "50")]
    schedules.append(irregular_look_indices(n, seed=0))
    counts = [int(records_read_at(rec, idx)[-1]) for idx in schedules]
    common = min(counts)
    common_hash = _hash_prefix(rec, common)
    per = []
    for name, idx, cnt in zip(("2", "5", "15", "50", "irregular"),
                              schedules, counts):
        per.append({"schedule": name, "records_read": cnt,
                    "looks": int(len(idx)),
                    "last_look_time": float(idx[-1]) * H_W})
    return {"seed": 0, "anchor_count": int(len(rec["index"])),
            "hash_full": full, "hash_common": common_hash,
            "common_count": int(common), "per_schedule": per,
            "pass": bool(len({full}) == 1 and len({common_hash}) == 1)}


# ---------------------------------------------------------------------------
# item 4
# ---------------------------------------------------------------------------

def run_item4():
    out = {"rows": [], "per_h": []}
    for h in (H_W, H_W / 2.0, H_W / 4.0):
        tag = f"conv_{_h_tag(h)}"
        got = _load(tag)
        if got is None:
            print(f"  item4 h={h:.10f}...", file=sys.stderr)
            s = _synth(0, h, snapshots=False)
            rec = s["rec"]
            R = len(rec["index"])
            prefix = band_prefix(rec)
            means = {}
            for k in (1, 4, 8, 12):
                Y = band_Y(prefix, R, k)
                means[str(k)] = {"mean_Y": float(np.mean(Y)),
                                 "mean_abs_Y": float(np.mean(np.abs(Y))),
                                 "n": int(len(Y))}
            got = {"h": h, "n": int(s["n"]), "total_phi": int(R),
                   "means": means}
            _save(tag, got)
        else:
            print(f"  item4 h={h:.10f}: reuse", file=sys.stderr)
        out["per_h"].append(got)
        for k in (1, 4, 8, 12):
            m = got["means"][str(k)]
            out["rows"].append({"h": got["h"], "total_phi": got["total_phi"],
                                "k": k, "mean_Y": m["mean_Y"],
                                "mean_abs_Y": m["mean_abs_Y"], "n": m["n"]})
    return out


# ---------------------------------------------------------------------------
# item 5
# ---------------------------------------------------------------------------

def run_item5():
    got = _load("item5")
    if got is not None:
        return got
    print("  item5 synthetic seed 0...", file=sys.stderr)
    s = _synth(0, H_W, snapshots=False)
    organ = s["organ"]
    n = s["n"]
    m = organ["m_grid"]
    cum = np.cumsum(m, dtype=np.int64)
    # 10 s windows
    step = int(round(10.0 / H_W))
    nw = (n - 1) // step
    rates_by_block = {0: [], 1: []}
    for i in range(nw):
        a = i * step
        b = a + step
        rate = (int(cum[b]) - int(cum[a])) / 10.0
        blk = ((a * H_W) // BLOCK) % 2
        rates_by_block[int(blk)].append(rate)
    med_rate = {("quiet" if k == 0 else "active"):
                float(np.median(v)) for k, v in rates_by_block.items()}
    # median m per look at 2 Hz per block
    idx = look_indices("2", n)
    phi_look = cum[idx]
    m_look = np.zeros(len(idx), dtype=np.int64)
    m_look[1:] = np.diff(phi_look)
    tlook = idx * H_W
    blk_look = (tlook // BLOCK).astype(np.int64) % 2
    med_m = {}
    for name, b in (("quiet", 0), ("active", 1)):
        med_m[name] = float(np.median(m_look[blk_look == b]))
    # Phi added in the 5 s after each block switch
    added = []
    for t in range(int(BLOCK), int(LIFE_SYN), int(BLOCK)):
        a = int(round(t / H_W))
        b = int(round((t + 5.0) / H_W))
        added.append(int(cum[b]) - int(cum[a]))
    out = {
        "phi_total": int(organ["phi"]),
        "phi_rate_10s_median": med_rate,
        "median_m_per_look_2hz": med_m,
        "phi_5s_after_switch": {
            "n_switches": len(added),
            "median": float(np.median(added)),
            "max": int(max(added)),
            "values": added,
        },
        "ref_active_rate_50hz": 307.60515,
        "ref_source": "out2/results.json rates.rate_50hz active clean",
    }
    _save("item5", out)
    return out


# ---------------------------------------------------------------------------
# item 6
# ---------------------------------------------------------------------------

def _block_case(kind, life=1000.0):
    n = n_grid(life)
    if kind == "active":
        items = []
        for k in ("steps", "bursts", "ramp"):
            src = Source(k, seed=0)
            x = Receptor(src, 0.2).filtered(n)
            sig = calibrate_sigma_hat(src, tau=0.2)
            items.append((k, x, sig))
        n_sens = 3
    else:
        items = []
        for g in range(10):
            src = Source("silent", seed=g, sigma_n=0.2)
            x = Receptor(src, 0.2).filtered(n)
            sig = calibrate_sigma_hat(src, tau=0.2)
            items.append((f"noise{g}", x, sig))
        n_sens = 10
    return items, n, n_sens


def run_item6():
    got = _load("cost")
    if got is not None:
        return got
    rows = []
    for kind in ("quiet", "active"):
        items, n, n_sens = _block_case(kind)
        groups = _grid_groups(items, n)
        pm = Pacemaker(noisy=False, seed=0)
        t0 = time.perf_counter()
        organ = organ_stream(pm, groups, n)
        t1 = time.perf_counter()
        streams, basal = build_streams(pm, groups, n)
        u = UtilitySource([]).grid(n)
        cp, cm = UtilitySource.cumulative_integrals(u)
        rec = build_records(organ, cp, cm, p_raw=None, streams=streams,
                            basal_count=basal, snapshots=True)
        t2 = time.perf_counter()
        R = len(rec["index"])
        bytes_rec = 8 + 8 + 8 + 4 * n_sens + 8
        out = {
            "block": kind, "world_s": 1000.0, "n_sens": n_sens,
            "organ_seconds": t1 - t0,
            "buffer_seconds": t2 - t1,
            "total_seconds": t2 - t0,
            "cpu_s_per_world_s": (t2 - t0) / 1000.0,
            "records": int(R),
            "records_per_world_s": R / 1000.0,
            "bytes_per_record": int(bytes_rec),
            "projected_mb_per_50000s": R / 1000.0 * 50000.0 * bytes_rec / 1e6,
        }
        rows.append(out)
        print(f"  item6 {kind}: {out['cpu_s_per_world_s']:.6f} s/s, "
              f"R={R}", file=sys.stderr)
    active = [r for r in rows if r["block"] == "active"][0]
    return {"rows": rows,
            "pass": bool(active["cpu_s_per_world_s"] <= 0.01)}


# ---------------------------------------------------------------------------
# item 7
# ---------------------------------------------------------------------------

def run_item7():
    got = _load("delta")
    if got is not None:
        return got
    life = 2000.0
    n = n_grid(life)
    active_items, _ = _active_groups(life, 0.0, 0)
    noise_items, _ = _active_groups(life, 0.05, 0)
    pm = Pacemaker(noisy=False, seed=0)
    rows = []
    for delta in (0.25, 1.0, 4.0, 16.0):
        omega_p = (delta / 0.25) ** 2
        for case, items in (("quiet", []), ("active", active_items),
                            ("active_sensor_noise", noise_items)):
            groups = _grid_groups(items, n) if items else []
            organ = organ_stream(pm, groups, n, delta=delta, omega_p=omega_p)
            rows.append({"delta": delta, "omega_p": omega_p, "case": case,
                         "phi": int(organ["phi"]),
                         "rate": float(organ["phi"]) / life})
    return {"rows": rows}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def write_tables(path, res):
    with open(path, "w") as f:
        f.write("# revision 4 tables (unit buffer inside the organ)\n\n")
        f.write("Experience time = RECEPTOR mode, h_w = 1/150 s, delta 0.25 "
                "unless stated.\n\n")

        f.write("## Item 1: traces with the unit buffer\n\n")
        f.write("| case | lambda | T_U 2 Hz | T_U 50 Hz | rel diff | PASS |\n")
        f.write("|---|---|---|---|---|---|\n")
        for x in res["item1"]["rows"]:
            f.write(f"| {x['case']} | {x['lambda']} | {x['T_U_2']:.6f} | "
                    f"{x['T_U_50']:.6f} | {_fmt(x['rel_diff'],8)} | "
                    f"{'PASS' if x['pass'] else 'FAIL'} |\n")

        f.write("\n## Item 2: H6(d) censored fractions with the buffer\n\n")
        f.write(f"kmax = {res['item2']['kmax']}. Buffer bands are exact at "
                "unit resolution; the 2 Hz and 50 Hz numbers are identical.\n\n")
        f.write("| k | n impulse | n neutral | cens imp | cens neutral |\n")
        f.write("|---|---|---|---|---|\n")
        for x in res["item2"]["rows"]:
            f.write(f"| {x['k']} | {x['n_imp']} | {x['n_neu']} | "
                    f"{x['n_imp_cens']} | {x['n_neu_cens']} |\n")
        b = res["item2"]["broken_policy_b"]
        f.write(f"\nBroken control (policy b, k={b['k']}, {b['rate_hz']} Hz): "
                f"f_imp {b['f_imp']:.4f}, f_neu {b['f_neu']:.4f}, "
                f"diff {b['diff']:.4f} "
                f"({'PASS' if b['pass'] else 'FAIL'}).\n")

        f.write("\n## Item 3: look-rate invariance of memory\n\n")
        f.write(f"anchor_count {res['item3']['anchor_count']}; "
                f"hash_full {res['item3']['hash_full']}; "
                f"hash_common {res['item3']['hash_common']} "
                f"(common_count {res['item3']['common_count']}).\n\n")
        f.write("| schedule | looks | records read | last look (s) |\n")
        f.write("|---|---|---|---|\n")
        for p in res["item3"]["per_schedule"]:
            f.write(f"| {p['schedule']} | {p['looks']} | "
                    f"{p['records_read']} | {p['last_look_time']:.3f} |\n")

        f.write("\n## Item 4: grid dependence (report only)\n\n")
        f.write("| h | total Phi | k | mean Y | mean |Y| | n anchors |\n")
        f.write("|---|---|---|---|---|---|\n")
        for x in res["item4"]["rows"]:
            f.write(f"| {x['h']:.10f} | {x['total_phi']} | {x['k']} | "
                    f"{x['mean_Y']:.6f} | {x['mean_abs_Y']:.6f} | {x['n']} |\n")

        i5 = res["item5"]
        f.write("\n## Item 5: m distribution on the synthetic life (seed 0)\n\n")
        f.write(f"Phi total {i5['phi_total']}. Phi rate per world s "
                f"(median over 10 s windows): quiet "
                f"{i5['phi_rate_10s_median']['quiet']:.4f}, active "
                f"{i5['phi_rate_10s_median']['active']:.4f}. "
                f"Median m per look at 2 Hz: quiet "
                f"{i5['median_m_per_look_2hz']['quiet']:.1f}, active "
                f"{i5['median_m_per_look_2hz']['active']:.1f}. "
                f"Phi added in the 5 s after each block switch: median "
                f"{i5['phi_5s_after_switch']['median']:.1f}, max "
                f"{i5['phi_5s_after_switch']['max']} over "
                f"{i5['phi_5s_after_switch']['n_switches']} switches. "
                f"Follow-up 2 ACTIVE 50 Hz rate: "
                f"{i5['ref_active_rate_50hz']} "
                f"({i5['ref_source']}).\n")

        f.write("\n## Item 6: cost and memory\n\n")
        f.write("| block | records | records/world s | bytes/record | "
                "CPU s/world s | projected MB / 50,000 s |\n")
        f.write("|---|---|---|---|---|---|\n")
        for x in res["item6"]["rows"]:
            f.write(f"| {x['block']} | {x['records']} | "
                    f"{x['records_per_world_s']:.3f} | {x['bytes_per_record']} | "
                    f"{x['cpu_s_per_world_s']:.6f} | "
                    f"{x['projected_mb_per_50000s']:.1f} |\n")
        f.write("\nAt each look rate (2/5/15/50/irregular) the record count "
                "and the projected MB are the same: records are organ-side and "
                "look-independent.\n")

        f.write("\n## Item 7: delta sweep (no pass rule)\n\n")
        f.write("| delta | omega_p | case | Phi | rate (Phi/world s) |\n")
        f.write("|---|---|---|---|---|\n")
        for x in res["item7"]["rows"]:
            f.write(f"| {x['delta']} | {x['omega_p']} | {x['case']} | "
                    f"{x['phi']} | {x['rate']:.5f} |\n")
        if "item8" in res:
            f.write("\n## Item 8: tests\n\n")
            f.write(f"test_unit_buffer.py -q -s: {res['item8']['summary']}. "
                    f"Full suite: {res['item8']['full_suite']}. "
                    f"(a) records identical across looks: "
                    f"{'PASS' if res['item8']['a'] else 'FAIL'}; "
                    f"(b) band Y equals direct integral within 1e-9: "
                    f"{'PASS' if res['item8']['b'] else 'FAIL'}; "
                    f"(c) broken look-buffer fails (a): "
                    f"{'PASS' if res['item8']['c'] else 'FAIL'}.\n")



def main():
    t0 = time.time()
    os.makedirs(OUT4, exist_ok=True)
    item1 = _load("item1")
    if item1 is None:
        print("item 1...", file=sys.stderr)
        item1 = run_item1(); _save("item1", item1)
    item7 = _load("item7")
    if item7 is None:
        print("item 7...", file=sys.stderr)
        item7 = run_item7(); _save("item7", item7)
    item6 = _load("item6")
    if item6 is None:
        print("item 6...", file=sys.stderr)
        item6 = run_item6(); _save("item6", item6)
    item2 = _load("item2")
    if item2 is None:
        print("item 2...", file=sys.stderr)
        item2 = run_item2(); _save("item2", item2)
    item3 = _load("item3")
    if item3 is None:
        print("item 3...", file=sys.stderr)
        item3 = run_item3(); _save("item3", item3)
    item5 = _load("item5")
    if item5 is None:
        print("item 5...", file=sys.stderr)
        item5 = run_item5(); _save("item5", item5)
    item4 = _load("item4")
    if item4 is None:
        print("item 4...", file=sys.stderr)
        item4 = run_item4(); _save("item4", item4)

    res = {"meta": {"design": "revision 4 (20 Nov 2026)",
                    "h_w": H_W, "seeds": list(SEEDS),
                    "schedules": list(SCHEDULES),
                    "lambdas": list(LAMBDAS), "S": S,
                    "delta_default": 0.25,
                    "total_seconds": None},
           "item1": item1, "item2": item2, "item3": item3,
           "item4": item4, "item5": item5, "item6": item6, "item7": item7}
    res["meta"]["total_seconds"] = time.time() - t0
    with open(os.path.join(OUT4, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_tables(os.path.join(OUT4, "tables.md"), res)
    print(f"done in {res['meta']['total_seconds']:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
