"""Selftest, second group: old behaviour, known answers, head contributions, mu_c=0.5."""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy  # noqa: E402

R = os.path.abspath(os.path.join(HERE, "..", ".."))
PRE = os.path.join(HERE, "check")
OLD = os.path.join(HERE, "out", "cells.jsonl")

RESULTS = []


def check(item, name, ok, detail):
    RESULTS.append((item, name, bool(ok), detail))
    print("[item %s] %-38s %s :: %s" % (item, name, "PASS" if ok else "FAIL", detail),
          flush=True)
    return ok


def old_row(learner, kappa, feats, rho, T, sign=1, background=True, contingent=True,
            dt=0.1, alpha_p=0.1, alpha_v=0.02, n_choices=400):
    for line in open(OLD):
        d = json.loads(line)
        if (d["learner"] == learner and d["features"] == feats and d["rho"] == rho
                and d["T"] == T and d["sign"] == sign
                and bool(d["background"]) == background
                and bool(d["contingent"]) == contingent and d["dt"] == dt
                and d["alpha_p"] == alpha_p and d["alpha_v"] == alpha_v
                and d["n_choices"] == n_choices
                and (d["kappa"] == kappa if learner == "L2" else True)):
            return d
    raise KeyError("old row not found")


def item1():
    ok = True
    for learner, kappa in (("L1", None), ("L2", 1.0), ("L3", None)):
        kk = 1.0 if kappa is None else kappa
        ref = old_row(learner, kappa, "F1", 4.0, 3.0)
        new = toy.run_cell(learner, "F1", rho=4.0, T=3.0, sign=1, background=True,
                           contingent=True, kappa=kk, n_choices=400)
        same_n = (new["N_star"] == ref["N_star"])
        same_p = np.array_equal(np.array(new["final_p"]),
                                np.array(ref["final_p"]))
        ok &= check("1", "old behaviour %s" % learner.replace("L2", "L2 k=1.0"),
                    same_n and same_p,
                    "N_star identical: %s; final_p bit-identical: %s"
                    % (same_n, same_p))
    return ok


def item2():
    pre = json.load(open(os.path.join(PRE, "precheck_T3.json")))
    ok = True
    for suffix, kw in (("c100", dict(tau_r=100.0)),
                       ("cs100", dict(tau_r=100.0, cv=0.3))):
        for cont, tag in ((False, "broken"), (True, "real")):
            key = "%s|T=3|%s" % (suffix, tag)
            ref = pre[key]
            new = toy.run_cell("L2", "F1", rho=4.0, T=3.0, sign=1, background=True,
                               contingent=cont, kappa=1.0, n_choices=400, **kw)
            same_n = (new["N_star"] == ref["N_star"])
            dfp = float(np.max(np.abs(np.array(new["final_p"]) -
                                      np.array(ref["final_p"]))))
            ok &= check("2", "known answer %s" % key,
                        same_n and dfp < 1e-9,
                        "N_star identical: %s; max |final_p diff| %.3e"
                        % (same_n, dfp))
    return ok


def item3():
    o = toy.run_cell("L2", "F1", rho=4.0, T=3.0, sign=1, background=True,
                     contingent=True, kappa=1.0, n_choices=400, tau_r=100.0)
    c = np.array(o["head_contrib"])
    th = np.array(o["theta_end"])
    err = float(np.max(np.abs(c.sum(0) - th)))
    return check("3", "sum head_contrib == theta", err <= 1e-9,
                 "max |sum_h head_contrib - theta| %.3e over 16 seeds" % err)


def item4():
    a = toy.run_cell("L2", "F1", rho=4.0, T=3.0, sign=1, background=True,
                     contingent=True, kappa=1.0, n_choices=400, tau_r=100.0,
                     mu_c=0.5)
    b = toy.run_cell("L2", "F1", rho=4.0, T=3.0, sign=1, background=True,
                     contingent=True, kappa=1.0, n_choices=400, tau_r=100.0,
                     mu_c=0.5, _force_mu_new=True)
    d = float(np.max(np.abs(np.array(a["final_p"]) - np.array(b["final_p"]))))
    return check("4", "mu_c=0.5 old vs new bound", d < 1e-9,
                 "max |final_p diff| %.3e" % d)


def main():
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    if "1" in which:
        item1()
    if "2" in which:
        item2()
    if "3" in which:
        item3()
    if "4" in which:
        item4()
    bad = [r for r in RESULTS if not r[2]]
    print("\nSELFTEST_B SUMMARY: %d checks, %d failed" % (len(RESULTS), len(bad)),
          flush=True)
    for r in bad:
        print("  FAILED item %s: %s :: %s" % r, flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
