"""Run one second-group cell, save its p record, print one JSON line."""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN_KEYS = ("learner", "features", "rho", "T", "sign", "background", "contingent",
            "dt", "alpha_p", "alpha_v", "kappa", "n_choices", "tau_r", "cv",
            "m", "eta_max", "beta", "combo")


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception as e:  # pragma: no cover
        return "NOT_FOUND: %s" % e


def safe_name(cid):
    return re.sub(r"[^A-Za-z0-9.-]", "_", cid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", default="out_b")
    args = ap.parse_args()
    outdir = os.path.join(HERE, args.out)
    pdir = os.path.join(outdir, "p")
    p = json.loads(args.params)
    kw = {k: p[k] for k in RUN_KEYS if k in p}
    kw.setdefault("n_choices", 800)
    if "tau_r" in p:
        kw["tau_r"] = p["tau_r"]
    if "cv" in p:
        kw["cv"] = p["cv"]
    if "mu_c" in p:
        kw["mu_c"] = p["mu_c"]

    t0 = time.time()
    res = toy.run_cell(**kw)
    wall = time.time() - t0

    p_rec = res.pop("p_rec")
    cid = toy.cell_id(res)
    fn = safe_name(cid) + ".npz"
    os.makedirs(pdir, exist_ok=True)
    np.savez_compressed(os.path.join(pdir, fn), p=p_rec,
                        seeds=np.asarray(res["seeds"], dtype=np.int64))
    res["p_file"] = "p/" + fn
    res["variant"] = p.get("variant")
    res["wall_s"] = wall
    res["git_commit"] = git_head()
    res["numpy"] = np.__version__
    res["cell_id"] = cid
    sys.stdout.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
