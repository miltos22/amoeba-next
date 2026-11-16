"""W23 grid driver: cell list, cost estimate, resumable 8-process runner."""

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
D = os.path.join(ROOT, "design_notes")
OUT = os.path.join(HERE, "out", "cells.jsonl")
LOGS = os.path.join(D, "logs")
COSTFILE = os.path.join(D, "cost_estimate.txt")

LEARNERS = [("L1", 1.0), ("L2", 0.043), ("L2", 1.0), ("L3", 1.0)]
MAIN_RHO = [4.0, 1.0, 0.25]
MAIN_T = [3.0, 30.0, 300.0]
SPOT_T = [3.0, 30.0, 300.0]


def base_cell(learner, kappa, features, rho, T, **over):
    p = dict(learner=learner, kappa=float(kappa), features=features, rho=float(rho),
             T=float(T), sign=1, background=True, contingent=True, dt=0.1,
             alpha_p=0.1, alpha_v=0.02, n_choices=400)
    p.update(over)
    return p


def main_cells():
    out = []
    for learner, kappa in LEARNERS:
        for feats in ("F1", "F2"):
            for rho in MAIN_RHO:
                for T in MAIN_T:
                    out.append(base_cell(learner, kappa, feats, rho, T))
    return out


def spot_cells():
    out = []
    for learner, kappa in (("L2", 1.0), ("L3", 1.0)):
        for feats in ("F1", "F2"):
            for T in SPOT_T:
                out.append(base_cell(learner, kappa, feats, 4.0, T, sign=-1))
                out.append(base_cell(learner, kappa, feats, 4.0, T, background=False))
                out.append(base_cell(learner, kappa, feats, 4.0, T, alpha_p=0.03))
                out.append(base_cell(learner, kappa, feats, 4.0, T, alpha_p=0.3))
                out.append(base_cell(learner, kappa, feats, 4.0, T, dt=0.05))
    return out


def broken_cells():
    out = []
    for learner, kappa in LEARNERS:
        for feats in ("F1", "F2"):
            for T in (3.0, 300.0):
                out.append(base_cell(learner, kappa, feats, 4.0, T, contingent=False))
    return out


def all_cells():
    return main_cells() + spot_cells() + broken_cells()


def read_cost_decision():
    if not os.path.exists(COSTFILE):
        return False
    txt = open(COSTFILE).read()
    return "DECISION: cut250" in txt


def apply_cut(cells, cut):
    if not cut:
        return cells
    for p in cells:
        if p["T"] == 300.0 and p["rho"] == 4.0:
            p["n_choices"] = 250
    return cells


# ---------------------------------------------------------------- cost
def total_grid_steps():
    tot = {"main": 0, "spot": 0, "broken": 0}
    for name, fn in (("main", main_cells), ("spot", spot_cells), ("broken", broken_cells)):
        for p in fn():
            tot[name] += toy.Stream(T=p["T"], rho=p["rho"], sign=p["sign"],
                                    background=p["background"],
                                    contingent=p["contingent"],
                                    seeds=range(toy.NS), dt=p["dt"],
                                    n_choices=p["n_choices"]).n_steps
    return tot


def cost_mode():
    os.makedirs(LOGS, exist_ok=True)
    rows = []
    for T in (3.0, 30.0):
        p = base_cell("L2", 1.0, "F2", 4.0, T)
        t0 = time.time()
        r = toy.run_cell(**p)
        wall = time.time() - t0
        rows.append((T, r["n_steps"], wall, r["n_steps"] / wall))
        print("cost: T=%g steps=%d wall=%.2fs rate=%.1f steps/s"
              % (T, r["n_steps"], wall, r["n_steps"] / wall), flush=True)
    rate = sum(r[1] for r in rows) / sum(r[2] for r in rows)
    print("cost: combined single-process rate %.1f steps/s" % rate, flush=True)

    # parallel probe: 8 copies of the T=30 cell at once, same launch form as the grid
    runner = os.path.join(HERE, "run_cell.py")
    p30 = json.dumps(base_cell("L2", 1.0, "F2", 4.0, 30.0))
    cmd = ["nice", "-n", "19", "taskset", "-c", "0-7", sys.executable, runner, "--params", p30]
    t0 = time.time()
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True) for _ in range(8)]
    outs = [pr.communicate() for pr in procs]
    wall_par = time.time() - t0
    ok8 = all(pr.returncode == 0 and o.strip() for pr, (o, e) in zip(procs, outs))
    rate_par = (8.0 * rows[1][1]) / wall_par
    print("cost: parallel probe 8 x T=30 wall=%.2fs aggregate=%.1f steps/s ok=%s"
          % (wall_par, rate_par, ok8), flush=True)

    tot = total_grid_steps()
    total = tot["main"] + tot["spot"] + tot["broken"]
    est_serial8 = total / rate / 8.0
    est_par = total / rate_par
    cut = est_par > 3.0 * 3600.0
    txt = []
    txt.append("COST ESTIMATE (W23 grid)")
    txt.append("measured cells: L2 kappa 1.0 F2 rho 4, dt 0.1, 16 seeds, 400 choices")
    for T, n, w, rt in rows:
        txt.append("  T=%g s: n_steps=%d wall=%.3f s rate=%.1f steps/s" % (T, n, w, rt))
    txt.append("combined single-process rate: %.1f steps/s" % rate)
    txt.append("parallel probe: 8 concurrent T=30 cells, wall %.2f s, aggregate %.1f steps/s"
               % (wall_par, rate_par))
    txt.append("total grid steps: main %d + spot %d + broken %d = %d"
               % (tot["main"], tot["spot"], tot["broken"], total))
    txt.append("estimate from single-cell rate / 8 processes: %.2f h" % (est_serial8 / 3600.0))
    txt.append("estimate from the 8-process aggregate rate:    %.2f h" % (est_par / 3600.0))
    txt.append("primary estimate (8-process aggregate): %.2f h; threshold 3 h"
               % (est_par / 3600.0))
    txt.append("DECISION: %s" % ("cut250" if cut else "full400"))
    if cut:
        txt.append("  -> n_choices = 250 for the T=300, rho=4 cells")
    open(COSTFILE, "w").write("\n".join(txt) + "\n")
    print("\n".join(txt), flush=True)
    return 0


# ---------------------------------------------------------------- grid
def finished_ids():
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["cell_id"])
            except Exception:
                continue
    return done


def run_pending(cells, concurrency=8):
    os.makedirs(LOGS, exist_ok=True)
    done = finished_ids()
    todo = [p for p in cells if toy.cell_id(p) not in done]
    print("grid: %d cells, %d already done, %d to run" % (len(cells), len(done), len(todo)),
          flush=True)
    runner = os.path.join(HERE, "run_cell.py")
    t_start = time.time()
    finished = 0
    running = []
    with open(OUT, "a") as fo:
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
                    print("grid: FAILED %s (see logs/fail_%s.log)" % (cid, tag), flush=True)
                    continue
                res = json.loads(out.strip().splitlines()[-1])
                res["cell_id"] = cid
                fo.write(json.dumps(res) + "\n")
                fo.flush()
                finished += 1
                print("grid: done %3d/%d  %6.1fs  %s"
                      % (finished, len(todo) + finished, time.time() - t0, cid), flush=True)
            running = still
    print("grid: finished %d cells in %.1f s" % (finished, time.time() - t_start), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cost", "plan", "run"])
    args = ap.parse_args()
    cut = read_cost_decision()
    cells = apply_cut(all_cells(), cut)
    if args.mode == "cost":
        return cost_mode()
    if args.mode == "plan":
        print("cells: %d; cut=%s" % (len(cells), cut))
        for c in cells:
            print(toy.cell_id(c))
        return 0
    run_pending(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
