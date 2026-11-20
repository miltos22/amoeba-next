"""Run one cell and print one JSON object on stdout (design 7.5)."""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception as e:  # pragma: no cover
        return "NOT_FOUND: %s" % e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="JSON object of cell parameters")
    args = ap.parse_args()
    p = json.loads(args.params)

    t0 = time.time()
    res = toy.run_cell(
        learner=p["learner"],
        features=p["features"],
        rho=p.get("rho", 4.0),
        T=p.get("T", 30.0),
        sign=p.get("sign", 1),
        background=p.get("background", True),
        contingent=p.get("contingent", True),
        dt=p.get("dt", 0.1),
        alpha_p=p.get("alpha_p", 0.1),
        alpha_v=p.get("alpha_v", 0.02),
        kappa=p.get("kappa", 1.0),
        n_choices=p.get("n_choices", 400),
        I=p.get("I", None),
    )
    res["wall_s"] = time.time() - t0
    res["git_commit"] = git_head()
    res["numpy"] = np.__version__
    res["cell_id"] = toy.cell_id(res)
    res.pop("events", None)
    res.pop("p_rec", None)
    sys.stdout.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
