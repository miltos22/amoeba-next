"""Grid driver (third group): 64 L4 cells, cost estimate, resumable 6-process runner."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
D = os.path.join(ROOT, "runs_c")
OUT = os.path.join(HERE, "out_c")
CELLS = os.path.join(OUT, "cells.jsonl")
COSTFILE = os.path.join(D, "cost_estimate.txt")
LOGS = os.path.join(D, "logs")


def cell(combo, m, beta, rho, T, contingent, sign=1, group="g1"):
    return dict(learner="L4", features="F1", rho=float(rho), T=float(T),
                sign=int(sign), background=True, contingent=bool(contingent),
                dt=0.1, m=float(m), eta_max=0.2, beta=float(beta),
                combo=combo, n_choices=800, variant=group)


def build_cells():
    cells = []
    # item 1: combo x m x rho x T x stream
    for combo in ("max", "mean"):
        for m in (10.0, 30.0):
            for rho in (4.0, 0.25):
                for T in (3.0, 30.0, 300.0):
                    for cont in (True, False):
                        cells.append(cell(combo, m, 4.0, rho, T, cont, group="g1"))
    # item 2: combo max, m 10, sign -1
    for rho in (4.0, 0.25):
        for T in (3.0, 30.0):
            for cont in (True, False):
                cells.append(cell("max", 10.0, 4.0, rho, T, cont, sign=-1,
                                  group="g2"))
    # item 3: combo max, m 10, beta 8, sign +1, T in {3, 300}
    for rho in (4.0, 0.25):
        for T in (3.0, 300.0):
            for cont in (True, False):
                cells.append(cell("max", 10.0, 8.0, rho, T, cont, group="g3"))
    cells.sort(key=lambda c: c["T"])  # stable: all T=3 first, then 30, then 300
    return cells


def run_kwargs(p):
    keys = ("learner", "features", "rho", "T", "sign", "background", "contingent",
            "dt", "m", "eta_max", "beta", "combo", "n_choices")
    return {k: p[k] for k in keys if k in p}


def read_decision():
    if not os.path.exists(COSTFILE):
        return "full"
    txt = open(COSTFILE).read()
    if "DECISION: skip_g3_T300_rho4" in txt:
        return "skip_g3_T300_rho4"
    return "full"


def apply_decision(cells, decision):
    if decision == "skip_g3_T300_rho4":
        return [c for c in cells
                if not (c["variant"] == "g3" and c["T"] == 300.0
                        and c["rho"] == 4.0)]
    return cells


# ---------------------------------------------------------------- cost
def cost_mode():
    os.makedirs(LOGS, exist_ok=True)
    p = cell("max", 10.0, 4.0, 4.0, 3.0, True)
    t0 = time.time()
    toy.run_cell(**run_kwargs(p), max_steps=20_000)
    wall = time.time() - t0
    per_step = wall / 20_000.0
    print("cost: L4 F1 20000 steps wall=%.2f s -> %.6f s/step"
          % (wall, per_step), flush=True)

    cells = build_cells()
    total = 0
    core_s = 0.0
    skip_core = 0.0
    for c in cells:
        st = toy.Stream(T=c["T"], rho=c["rho"], sign=c["sign"],
                        background=c["background"], contingent=c["contingent"],
                        seeds=range(toy.NS), dt=c["dt"], n_choices=c["n_choices"])
        total += st.n_steps
        cost = st.n_steps * per_step
        core_s += cost
        if not (c["variant"] == "g3" and c["T"] == 300.0 and c["rho"] == 4.0):
            skip_core += cost
    core_h = core_s / 3600.0
    skip_h = skip_core / 3600.0
    skip = core_h > 20.0
    txt = []
    txt.append("COST ESTIMATE (grid)")
    txt.append("measured: L4 F1, 20000 steps: %.6f s/step" % per_step)
    txt.append("total cells: %d" % len(cells))
    txt.append("total steps: %d" % total)
    txt.append("projected core-hours (all cells): %.2f" % core_h)
    txt.append("projected core-hours without g3 T=300 rho=4: %.2f" % skip_h)
    txt.append("threshold: 20 core-hours")
    txt.append("DECISION: %s" % ("skip_g3_T300_rho4" if skip else "full"))
    if skip:
        txt.append("  -> not run: the item-3 cells at T=300, rho=4")
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
                       sys.executable, runner, "--out", "out_c",
                       "--params", json.dumps(p)]
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
