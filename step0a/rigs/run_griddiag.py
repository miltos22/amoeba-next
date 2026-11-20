"""Follow-up 5: why total Phi grows with the grid in the synthetic life.

New code only.  Outputs out5/results.json and out5/tables.md.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import step0a.rigs.run_item2 as R3
from step0a.experience_time import GridGroup, SIGMA_MIN, organ_stream
from step0a.pacemaker import Pacemaker
from step0a.receptors import calibrate_sigma_hat, exact_filter
from step0a.world_signals import (
    H_W, Source, UtilitySource, _ou_on_grid, grid_times, n_grid,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT5 = os.path.join(ROOT, "runs", "out5")
SEEDS = (0, 1, 2)
LIFE = 20000.0
BLOCK = 1000.0
HS = (H_W, H_W / 2.0, H_W / 4.0)
H_FINE = H_W / 8.0


def _part(name):
    return os.path.join(OUT5, f"part_{name}.json")


def _save(name, obj):
    os.makedirs(OUT5, exist_ok=True)
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


def _h_tag(h):
    return f"hdiv{int(round(H_W / h))}"


# ---------------------------------------------------------------------------
# item 1: code facts + OU statistics
# ---------------------------------------------------------------------------

CODE_FACTS = {
    "a_ref_convergence": [
        "run_receptor.py:216 `def run_convergence(seed=0):`",
        "run_receptor.py:223 `groups, n = build_groups(case, seed, h, life)`",
        "run_receptor.py:68 `def build_groups(case, seed, h, life):`",
        "run_receptor.py:78 `src = Source(\"silent\", seed=seed * 100 + g, sigma_n=SIGMA_N)`",
        "run_receptor.py:79 `x = Receptor(src, 0.2).filtered(n, h)`",
        "So each h regenerates the source on a new grid n = n_grid(life, h); "
        "the OU noise is a new realisation, and the receptor filter uses "
        "a = exp(-h/0.2) on that grid.",
    ],
    "b_item4": [
        "run_buffer.py:301 `def run_item4():`",
        "run_buffer.py:307 `s = _synth(0, h, snapshots=False)`",
        "run_buffer.py:118 `def _synth(seed, h=H_W, snapshots=False):`",
        "run_buffer.py:120 `groups = R3._synthetic_groups(seed, n, h)`",
        "run_item2.py:290 `def _synthetic_groups(seed, n, h):`",
        "So item 4 rebuilds every source at each h from the same seeds, with "
        "the OU noise redrawn on each grid and sigma_hat recalibrated at each h.",
    ],
    "c_ou_generator": [
        "world_signals.py:71 `def _ou_on_grid(n, sigma, tau, seed, h=H_W)`",
        "world_signals.py:81 `a = math.exp(-h / tau)`",
        "world_signals.py:85 `drive = z * (sigma * math.sqrt(1.0 - a * a) / b)`",
        "world_signals.py:87 `return exact_filter(drive, a, b, x0=float(rng.standard_normal()) * sigma)`",
        "The update is x_{k+1} = a x_k + sigma sqrt(1 - a^2) z_k with a = exp(-h/tau).",
        "Stationary variance: sum over the recursion gives sigma^2 for any h "
        "(the drive variance is sigma^2 (1 - a^2) and the geometric sum is 1/(1-a^2)).",
        "Autocorrelation at lag m is a^|m| = exp(-m h/tau) = exp(-Delta t/tau), "
        "so the continuous-time correlation and the Lorentzian spectrum are h-independent; "
        "only aliasing above the Nyquist frequency differs, and the tau=0.5 s "
        "corner is 1/(2 pi tau) = 0.318 Hz, far below 10 Hz.",
        "The realisation is not the same at different h: z = rng.standard_normal(n) "
        "draws a different number of samples with the same seed.",
    ],
    "d_rendering": [
        "world_signals.py:104 `def raw_grid(self, n, h=H_W)`",
        "world_signals.py:111 `if self.kind == \"steps\":` ... event times are "
        "Exp(20) draws independent of h, then `np.searchsorted(b, t)` at "
        "world_signals.py:125 samples the same physical piecewise function on each grid.",
        "world_signals.py:128 `if self.kind == \"bursts\":` ... start times are "
        "Exp(30) draws; `i0 = round(tt/h)`, `i1 = round((tt+0.3)/h)` and "
        "`out[i0:min(i1,n)] = 3.0` at world_signals.py:141 render a 0.3 s burst "
        "with a grid-dependent number of samples and rounding at the edges.",
        "run_item2.py:290-294 builds the 0/1 masks from `t = grid_times(n, h)`, "
        "`block = (t // 1000)` and `quiet_mask = (block % 2 == 0)`; 1000 s is a "
        "multiple of every tested h, so the block boundaries are the same physical times.",
        "run_item2.py:301 and 308 apply `raw * quiet_mask` / `raw * active_mask` "
        "before `exact_filter(raw, a, b)` with `a = exp(-h/0.2)`.",
    ],
}


def _welch_power(x, fs, fmax=10.0, L=2 ** 15):
    n = len(x)
    if n < 16:
        return None, None
    L = min(L, n)
    w = np.hanning(L)
    step = max(1, L // 2)
    freqs = np.fft.rfftfreq(L, 1.0 / fs)
    psd = np.zeros_like(freqs)
    cnt = 0
    for start in range(0, n - L + 1, step):
        seg = x[start:start + L]
        seg = seg - seg.mean()
        X = np.fft.rfft(seg * w)
        psd += np.abs(X) ** 2
        cnt += 1
    if cnt == 0:
        return None, None
    psd *= 2.0 / (fs * np.sum(w ** 2) * cnt)
    sel = freqs <= fmax
    power = float(np.trapezoid(psd[sel], freqs[sel]))
    return power, {"freqs": freqs.tolist(), "psd": psd.tolist()}


def run_item1():
    rows = []
    for h in (H_W, H_W / 2.0, H_W / 4.0, H_FINE):
        n = n_grid(LIFE, h)
        vars_ = []
        powers = []
        for seed in SEEDS:
            x = _ou_on_grid(n, 0.2, 0.5, seed, h)
            vars_.append(float(np.var(x)))
            p, _ = _welch_power(x, 1.0 / h)
            powers.append(p)
        rows.append({"h": h, "h_tag": _h_tag(h), "n": int(n),
                     "var_mean": float(np.mean(vars_)),
                     "var_seeds": vars_,
                     "power_0_10hz_mean": (float(np.mean(powers))
                                           if all(p is not None for p in powers)
                                           else None),
                     "power_0_10hz_seeds": powers})
    return {"code": CODE_FACTS, "ou_rows": rows,
            "sigma": 0.2, "tau": 0.5,
            "analytic_0_10hz_power": 0.04 * (2.0 / np.pi)
            * np.arctan(2.0 * np.pi * 10.0 * 0.5)}


# ---------------------------------------------------------------------------
# split helper
# ---------------------------------------------------------------------------

def split_phi(m_grid, n, h):
    t = grid_times(n, h)
    block = ((t // BLOCK).astype(np.int64)) % 2
    switch = np.zeros(n, dtype=bool)
    for s in range(int(BLOCK), int(LIFE), int(BLOCK)):
        switch |= (t >= s) & (t < s + 5.0)
    m = m_grid.astype(np.int64)
    total = int(m.sum())
    quiet = int(m[block == 0].sum())
    active = int(m[block == 1].sum())
    sw = int(m[switch].sum())
    quiet_d = int(m[(block == 0) & ~switch].sum())
    active_d = int(m[(block == 1) & ~switch].sum())
    return {
        "total": total,
        "quiet_total": quiet,
        "active_total": active,
        "switch_total": sw,
        "quiet_disjoint": quiet_d,
        "active_disjoint": active_d,
        "switch_disjoint": sw,
        "quiet_fraction": quiet / total if total else None,
        "active_fraction": active / total if total else None,
        "switch_fraction": sw / total if total else None,
        "step": h,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# item 2: regenerated synthetic life
# ---------------------------------------------------------------------------

def run_item2():
    out = {"per_h": [], "rel_vs_hw": []}
    for h in HS:
        tag = f"item2_{_h_tag(h)}"
        got = _load(tag)
        if got is None:
            print(f"  item2 h={h:.10f}...", file=sys.stderr)
            n = n_grid(LIFE, h)
            groups = R3._synthetic_groups(0, n, h)
            organ = organ_stream(Pacemaker(noisy=False, seed=0), groups, n, h)
            got = split_phi(organ["m_grid"], n, h)
            _save(tag, got)
        out["per_h"].append(got)
    base = out["per_h"][0]
    for got in out["per_h"]:
        out["rel_vs_hw"].append({
            "h": got["step"], "h_tag": _h_tag(got["step"]),
            "rel_total": (got["total"] - base["total"]) / base["total"],
            "rel_quiet": (got["quiet_total"] - base["quiet_total"]) / base["quiet_total"],
            "rel_active": (got["active_total"] - base["active_total"]) / base["active_total"],
            "rel_switch": (got["switch_total"] - base["switch_total"]) / base["switch_total"],
        })
    return out


# ---------------------------------------------------------------------------
# one physical signal at h_w/8 (items 3 and 4)
# ---------------------------------------------------------------------------

def _coarse_average(fine, n_c, R):
    s = np.zeros(n_c, dtype=np.float64)
    m = min(n_c - 1, (len(fine) - 1) // R)
    if m > 0:
        s[:m] = fine[:m * R].reshape(m, R).mean(axis=1)
    return s


def _cal_sigma(kind, sigma_n, h, noise_on):
    if kind == "silent":
        if not noise_on:
            return 0.0
        src = Source("silent", seed=999, sigma_n=sigma_n)
    else:
        src = Source(kind, seed=999)
    n_cal = n_grid(600.0, H_FINE)
    fine = src.to_grid(n_cal, H_FINE)
    R = int(round(h / H_FINE))
    n_c = n_grid(600.0, h)
    s = _coarse_average(fine, n_c, R)
    a = np.exp(-h / 0.2)
    x = exact_filter(s, a, 1.0 - a)
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _one_signal(noise_on):
    n_fine = n_grid(LIFE, H_FINE)
    t_fine = grid_times(n_fine, H_FINE)
    block = ((t_fine // BLOCK).astype(np.int64)) % 2
    quiet_mask = (block == 0)
    active_mask = ~quiet_mask

    specs = []
    for g in range(10):
        specs.append({"name": f"quiet_noise{g}", "kind": "silent",
                      "sigma_n": 0.2, "seed": g, "mask": quiet_mask})
    for kind in ("steps", "bursts", "ramp"):
        specs.append({"name": kind, "kind": kind, "sigma_n": 0.0,
                      "seed": 0, "mask": active_mask})

    coarse = {}
    for si, sp in enumerate(specs):
        if sp["kind"] == "silent" and not noise_on:
            s = None
        else:
            src = Source(sp["kind"], seed=sp["seed"], sigma_n=sp["sigma_n"])
            raw = src.to_grid(n_fine, H_FINE)
            s = raw * sp["mask"]
        for hi, h in enumerate(HS):
            n_c = n_grid(LIFE, h)
            R = int(round(h / H_FINE))
            if s is None:
                coarse[(si, hi)] = np.zeros(n_c)
            else:
                coarse[(si, hi)] = _coarse_average(s, n_c, R)

    u_fine, _events = R3._synthetic_events(0, n_fine, H_FINE)
    utility_means = {}
    for hi, h in enumerate(HS):
        n_c = n_grid(LIFE, h)
        R = int(round(h / H_FINE))
        utility_means[_h_tag(h)] = float(_coarse_average(u_fine, n_c, R).mean())
    del u_fine

    out = {"per_h": [], "sigma_hat": {}, "utility_means": utility_means}
    for hi, h in enumerate(HS):
        print(f"  one-signal noise_on={noise_on} h={h:.10f}...", file=sys.stderr)
        n_c = n_grid(LIFE, h)
        groups = []
        for si, sp in enumerate(specs):
            if sp["kind"] == "silent" and not noise_on:
                continue
            s_c = coarse[(si, hi)]
            a = np.exp(-h / 0.2)
            x = exact_filter(s_c, a, 1.0 - a)
            sig = _cal_sigma(sp["kind"], sp["sigma_n"], h, noise_on)
            groups.append(GridGroup(sp["name"], x[None, :], np.array([sig]),
                                    np.array([SIGMA_MIN])))
            out["sigma_hat"].setdefault(_h_tag(h), {})[sp["name"]] = sig
        organ = organ_stream(Pacemaker(noisy=False, seed=0), groups, n_c, h)
        got = split_phi(organ["m_grid"], n_c, h)
        out["per_h"].append(got)
        del groups, organ
    base = out["per_h"][0]
    for got in out["per_h"]:
        out.setdefault("rel_vs_hw", []).append({
            "h": got["step"], "h_tag": _h_tag(got["step"]),
            "rel_total": (got["total"] - base["total"]) / base["total"],
            "rel_quiet": (got["quiet_total"] - base["quiet_total"]) / base["quiet_total"],
            "rel_active": (got["active_total"] - base["active_total"]) / base["active_total"],
            "rel_switch": (got["switch_total"] - base["switch_total"]) / base["switch_total"],
        })
    # step relative changes
    prev = None
    for k, got in enumerate(out["per_h"]):
        if prev is not None:
            out.setdefault("step_rel", []).append({
                "h": got["step"],
                "rel_total_step": (got["total"] - prev["total"]) / prev["total"],
                "rel_quiet_step": (got["quiet_total"] - prev["quiet_total"]) / prev["quiet_total"],
                "rel_active_step": (got["active_total"] - prev["active_total"]) / prev["active_total"],
                "rel_switch_step": (got["switch_total"] - prev["switch_total"]) / prev["switch_total"],
            })
        prev = got
    return out


def run_item3():
    got = _load("item3")
    if got is None:
        got = _one_signal(noise_on=True)
        _save("item3", got)
    return got


def run_item4():
    got = _load("item4")
    if got is None:
        got = _one_signal(noise_on=False)
        _save("item4", got)
    return got


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def write_tables(path, res):
    with open(path, "w") as f:
        f.write("# revision 5 tables (grid diagnostic)\n\n")
        f.write("Synthetic life: seed 0, 20,000 s, 1,000 s blocks "
                "(even = QUIET, odd = ACTIVE).\n\n")

        f.write("## Item 1: OU statistics across h\n\n")
        f.write("| h | n | var (mean seeds) | 0-10 Hz power (mean seeds) |\n")
        f.write("|---|---|---|---|\n")
        for x in res["item1"]["ou_rows"]:
            f.write(f"| {x['h']:.10f} | {x['n']} | {_fmt(x['var_mean'],6)} | "
                    f"{_fmt(x['power_0_10hz_mean'],6)} |\n")
        f.write(f"\nAnalytic 0-10 Hz one-sided power "
                f"{res['item1']['analytic_0_10hz_power']:.6f} for sigma=0.2, "
                f"tau=0.5 s.\n")

        def split_table(key):
            f.write("| h | total Phi | quiet | active | switch 5 s | "
                    "quiet (disjoint) | active (disjoint) | switch fraction |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for x in res[key]["per_h"]:
                f.write(f"| {x['step']:.10f} | {x['total']} | "
                        f"{x['quiet_total']} | {x['active_total']} | "
                        f"{x['switch_total']} | {x['quiet_disjoint']} | "
                        f"{x['active_disjoint']} | "
                        f"{_fmt(x['switch_fraction'],4)} |\n")
            f.write("\n")
            f.write("| h | rel total | rel quiet | rel active | rel switch |\n")
            f.write("|---|---|---|---|---|\n")
            for x in res[key]["rel_vs_hw"]:
                f.write(f"| {x['h']:.10f} | {_fmt(x['rel_total'],4)} | "
                        f"{_fmt(x['rel_quiet'],4)} | {_fmt(x['rel_active'],4)} | "
                        f"{_fmt(x['rel_switch'],4)} |\n")
            f.write("\n")

        f.write("## Item 2: regenerated at each h\n\n")
        split_table("item2")
        f.write("## Item 3: one physical signal at h_w/8, sensor noise on\n\n")
        split_table("item3")
        f.write("Step relative changes:\n\n")
        f.write("| h | rel total step | rel quiet step | rel active step | "
                "rel switch step |\n")
        f.write("|---|---|---|---|---|\n")
        for x in res["item3"]["step_rel"]:
            f.write(f"| {x['h']:.10f} | {_fmt(x['rel_total_step'],4)} | "
                    f"{_fmt(x['rel_quiet_step'],4)} | "
                    f"{_fmt(x['rel_active_step'],4)} | "
                    f"{_fmt(x['rel_switch_step'],4)} |\n")
        f.write("\n## Item 4: one physical signal at h_w/8, sensor noise off\n\n")
        split_table("item4")
        f.write("Step relative changes:\n\n")
        f.write("| h | rel total step | rel quiet step | rel active step | "
                "rel switch step |\n")
        f.write("|---|---|---|---|---|\n")
        for x in res["item4"]["step_rel"]:
            f.write(f"| {x['h']:.10f} | {_fmt(x['rel_total_step'],4)} | "
                    f"{_fmt(x['rel_quiet_step'],4)} | "
                    f"{_fmt(x['rel_active_step'],4)} | "
                    f"{_fmt(x['rel_switch_step'],4)} |\n")


def main():
    os.makedirs(OUT5, exist_ok=True)
    item1 = _load("item1")
    if item1 is None:
        print("item 1...", file=sys.stderr)
        item1 = run_item1(); _save("item1", item1)
    item2 = run_item2()
    item3 = run_item3()
    item4 = run_item4()
    res = {"meta": {"design": "revision 5 (20 Nov 2026)",
                    "h_w": H_W, "h_fine": H_FINE, "hs": list(HS),
                    "seeds": list(SEEDS), "life": LIFE,
                    "block": BLOCK, "total_seconds": None},
           "item1": item1, "item2": item2, "item3": item3, "item4": item4}
    with open(os.path.join(OUT5, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    write_tables(os.path.join(OUT5, "tables.md"), res)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
