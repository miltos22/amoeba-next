"""Grid driver (second group): 62 L2 cells, cost estimate, resumable 6-process runner."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D = os.path.join(ROOT, "runs_b")
OUT = os.path.join(HERE, "out_b")
CELLS = os.path.join(OUT, "cells.jsonl")
COSTFILE = os.path.join(D, "cost_estimate.txt")
LOGS = os.path.join(D, "logs")

VARIANTS = {
    "A": dict(features="F1", tau_r=None, cv=None, mu_c=0.5),
    "B": dict(features="F1", tau_r=100.0, cv=None, mu_c=0.5),
    "C": dict(features="F1", tau_r=100.0, cv=0.3, mu_c=0.5),
    "D": dict(features="F2", tau_r=100.0, cv=None, mu_c=0.5),
    "E": dict(features="F2", tau_r=100.0, cv=None, mu_c=0.1),
}


def cell(name, rho, T, contingent, sign=1):
    v = VARIANTS[name]
    return dict(learner="L2", kappa=1.0, features=v["features"], rho=float(rho),
                T=float(T), sign=int(sign), background=True,
                contingent=bool(contingent), dt=0.1, alpha_p=0.1, alpha_v=0.02,
                n_choices=800, tau_r=v["tau_r"], cv=v["cv"], mu_c=v["mu_c"],
                variant=name)


def build_cells():
    cells = []
    for name in ("B", "C", "D", "E"):
        for rho in (4.0, 0.25):
            for T in (3.0, 30.0, 300.0):
                for cont in (True, False):
                    cells.append(cell(name, rho, T, cont))
    for rho in (4.0, 0.25):
        for T in (3.0, 30.0, 300.0):
            cells.append(cell("A", rho, T, False))
    for name in ("C", "E"):
        for T in (3.0, 30.0):
            for cont in (True, False):
                cells.append(cell(name, 4.0, T, cont, sign=-1))
    cells.sort(key=lambda c: c["T"])  # stable: all T=3 first, then 30, then 300
    return cells


def run_kwargs(p):
    kw = {k: p[k] for k in ("learner", "features", "rho", "T", "sign", "background",
                            "contingent", "dt", "alpha_p", "alpha_v", "kappa",
                            "n_choices", "tau_r", "cv", "mu_c") if k in p}
    return kw


def read_decision():
    if not os.path.exists(COSTFILE):
        return "full"
    txt = open(COSTFILE).read()
    if "DECISION: skip_f2_r4_T300" in txt:
        return "skip_f2_r4_T300"
    return "full"


def apply_decision(cells, decision):
    if decision == "skip_f2_r4_T300":
        return [c for c in cells
                if not (c["features"] == "F2" and c["rho"] == 4.0 and c["T"] == 300.0)]
    return cells


# ---------------------------------------------------------------- cost
def cost_mode():
    os.makedirs(LOGS, exist_ok=True)
    tapes = {}
    for name, feats in (("B", "F1"), ("D", "F2")):
        p = cell(name, 4.0, 3.0, True)
        t0 = time.time()
        toy.run_cell(**run_kwargs(p), max_steps=20_000)
        wall = time.time() - t0
        tapes[feats] = wall / 20_000.0
        print("cost: %s 20000 steps wall=%.2f s -> %.6f s/step"
              % (feats, wall, tapes[feats]), flush=True)

    cells = build_cells()
    per_feat = {}
    core_by_variant = {}
    core_hours_skip = 0.0
    for c in cells:
        st = toy.Stream(T=c["T"], rho=c["rho"], sign=c["sign"],
                        background=c["background"], contingent=c["contingent"],
                        seeds=range(toy.NS), dt=c["dt"], n_choices=c["n_choices"])
        s = st.n_steps
        per_feat[c["features"]] = per_feat.get(c["features"], 0) + s
        cost = s * tapes[c["features"]]
        core_by_variant[c["variant"]] = core_by_variant.get(c["variant"], 0.0) + cost
        if not (c["features"] == "F2" and c["rho"] == 4.0 and c["T"] == 300.0):
            core_hours_skip += cost / 3600.0
    total_steps = sum(per_feat.values())
    core_hours = sum(core_by_variant.values()) / 3600.0
    skip = core_hours > 20.0
    txt = []
    txt.append("COST ESTIMATE (grid)")
    txt.append("measured: 20000 steps, L2 kappa 1.0, 16 seeds, 800 choices")
    for feats in ("F1", "F2"):
        txt.append("  %s: %.6f s/step" % (feats, tapes[feats]))
    txt.append("total cells: %d" % len(cells))
    txt.append("total steps: %s (F1 %d, F2 %d)"
               % (total_steps, per_feat.get("F1", 0), per_feat.get("F2", 0)))
    txt.append("projected core-hours (all cells): %.2f" % core_hours)
    txt.append("projected core-hours without F2, rho 4, T 300: %.2f" % core_hours_skip)
    txt.append("threshold: 20 core-hours")
    txt.append("DECISION: %s" % ("skip_f2_r4_T300" if skip else "full"))
    if skip:
        txt.append("  -> not run: the F2 cells at rho 4, T 300")
    open(COSTFILE, "w").write("\n".join(txt) + "\n")
    print("\n".join(txt), flush=True)
    return 0


# ---------------------------------------------------------------- grid
def finished_ids():
    done = set()
    if os.path.exists(CELLS):
        for line in open(CELLS):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["cell_id"])
                except Exception:
                    pass
    return done


def run_pending(cells, concurrency=6):
    os.makedirs(LOGS, exist_ok=True)
    done = finished_ids()
    todo = [c for c in cells if toy.cell_id(c) not in done]
    print("grid: %d cells, %d done, %d to run" % (len(cells), len(done), len(todo)),
          flush=True)
    runner = os.path.join(HERE, "run_cell_b.py")
    t_start = time.time()
    finished = 0
    running = []
    with open(CELLS, "a") as fo:
        while todo or running:
            while todo and len(running) < concurrency:
                p = todo.pop(0)
                cmd = ["nice", "-n", "19", "taskset", "-c", "0-7",
                       sys.executable, runner, "--params", json.dumps(p)]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
                running.append([proc, p, time.time()])
            time.sleep(0.5)
            still = []
            for item in running:
                proc, p, t0 = item
                if proc.poll() is None:
                    still.append(item)
                    continue
                out, err = proc.communicate()
                cid = toy.cell_id(p)
                if proc.returncode != 0 or not out.strip():
                    tag = hashlib.sha1(cid.encode()).hexdigest()[:10]
                    open(os.path.join(LOGS, "fail_%s.log" % tag), "w").write(
                        "params: %s\nreturncode: %s\nstdout:\n%s\nstderr:\n%s\n"
                        % (json.dumps(p), proc.returncode, out, err))
                    print("grid: FAILED %s (logs/fail_%s.log)" % (cid, tag), flush=True)
                    continue
                res = json.loads(out.strip().splitlines()[-1])
                res["cell_id"] = cid
                fo.write(json.dumps(res) + "\n")
                fo.flush()
                finished += 1
                print("grid: done %3d  %7.1fs  %s"
                      % (finished, time.time() - t0, cid), flush=True)
            running = still
    print("grid: finished %d cells in %.1f s" % (finished, time.time() - t_start),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cost", "plan", "run"])
    args = ap.parse_args()
    cells = apply_decision(build_cells(), read_decision())
    if args.mode == "cost":
        return cost_mode()
    if args.mode == "plan":
        print("cells: %d decision=%s" % (len(cells), read_decision()))
        for c in cells:
            print(toy.cell_id(c))
        return 0
    run_pending(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
