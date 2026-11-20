"""Selftest for learner L4 (third group). Stops at the first failing item."""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toy  # noqa: E402

R = os.path.abspath(os.path.join(HERE, "..", ".."))
D = os.path.join(HERE, "runs_c")

RESULTS = []
_KNOWN = None


def check(item, name, ok, detail):
    RESULTS.append((item, name, bool(ok), detail))
    print("[item %s] %-46s %s :: %s"
          % (item, name, "PASS" if ok else "FAIL", detail), flush=True)
    return ok


def item1():
    p = os.path.join(HERE, "selftest_b.py")
    r = subprocess.run([sys.executable, p], capture_output=True, text=True)
    print("---- second-group selftest output (subprocess) ----", flush=True)
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stdout.write(r.stderr)
    print("---- end second-group selftest output ----", flush=True)
    ok = (r.returncode == 0) and ("9 checks, 0 failed" in r.stdout)
    return check("1", "second-group selftest still passes (9 checks, 0 failed)", ok,
                 "returncode %d; '9 checks, 0 failed' present: %s"
                 % (r.returncode, "9 checks, 0 failed" in r.stdout))


def item2():
    r = toy.run_cell("L4", "F1", rho=4.0, T=3.0, sign=0, background=False,
                     contingent=True, n_choices=200, _debug=True)
    p = r["p_rec"]
    q = np.array(r["q_end"])
    ok = bool(np.all(p == 0.5) and np.all(q == 0.0))
    return check("2", "no consequence, no background: p=0.5, Q=0", ok,
                 "p: %d values, all exactly 0.5: %s; Q: shape %s, max|Q| %.17g, "
                 "all exactly 0.0: %s"
                 % (p.size, bool(np.all(p == 0.5)), q.shape,
                    float(np.max(np.abs(q))), bool(np.all(q == 0.0))))


def known():
    global _KNOWN
    if _KNOWN is None:
        _KNOWN = toy.run_cell("L4", "F1", rho=400.0, T=3.0, sign=1,
                              background=False, contingent=True, m=10.0,
                              eta_max=0.2, beta=4.0, combo="max", n_choices=60,
                              force_action=1, _rb_off=True, _debug=True,
                              _tail_steps=31)
    return _KNOWN


def item3():
    r = known()
    q = np.array(r["q_end"])
    g = np.exp(-0.1 / toy.HORIZONS)
    k_off = 30  # Stream.offset for T=3, dt=0.1; read 30 steps after the choice
    flag = 1.0 - (1.0 - 0.2) ** 60
    ok = True
    print("[item 3] k_off = %d (Stream.offset = round(T/dt); the trace is decayed "
          "at the end of each of the %d steps between choice and read)"
          % (k_off, k_off), flush=True)
    for h in range(3):
        expv = flag * g[h] ** k_off
        err = float(np.max(np.abs(q[h, :, 1] - expv)))
        q0 = float(np.max(np.abs(q[h, :, 0])))
        good = (err < 1e-9) and (q0 == 0.0)
        ok &= good
        print("[item 3] head %d h=%.6f expected %.17g found(seed0) %.17g "
              "max|diff| %.3e max|Q0| %.3e %s"
              % (h, toy.HORIZONS[h], expv, q[h, 0, 1], err, q0,
                 "PASS" if good else "FAIL"), flush=True)
    return check("3", "known answer, three shortest heads (k_off=30)", ok,
                 "see the three head lines above")


def item4():
    r = known()
    q = np.array(r["q_end"])
    g = np.exp(-0.1 / toy.HORIZONS)
    k4 = 40  # the value T=4 would give
    flag = 1.0 - (1.0 - 0.2) ** 60
    diffs = [float(np.max(np.abs(q[h, :, 1] - flag * g[h] ** k4)))
             for h in range(3)]
    ok = max(diffs) > 1e-9
    return check("4", "the same check with T=4 must fail", ok,
                 "max|Q - expected(T=4)| per head = %s; max %.3e > 1e-9: %s"
                 % (["%.3e" % v for v in diffs], max(diffs), ok))


def item5():
    m, n_ch = 10.0, 60
    r = toy.run_cell("L4", "F1", rho=0.25, T=3.0, sign=1, background=False,
                     contingent=True, m=m, eta_max=0.2, beta=4.0, combo="max",
                     n_choices=n_ch, _debug=True)
    st = toy.Stream(T=3.0, rho=0.25, sign=1, background=False, contingent=True,
                    seeds=list(range(toy.NS)), dt=0.1, n_choices=n_ch)
    gaps = (st.choice_steps[:, -1] - st.choice_steps[:, -2]) * 0.1
    eta_last = np.array(r["eta_last"])  # (7, ns)
    exp = gaps / (m * toy.HORIZONS[6])
    err = float(np.max(np.abs(eta_last[6] - exp)))
    mean_eta = 0.75 / (m * toy.HORIZONS[6])
    cap = float(r["cap_frac_head"][6])
    ok = (err <= 1e-12) and (abs(cap - 1.0 / n_ch) <= 1e-15)
    print("[item 5] last gap s: seed0 %.6f, seed1 %.6f (min %.6f max %.6f); "
          "eta_last[6] seed0 %.17g seed1 %.17g"
          % (gaps[0], gaps[1], gaps.min(), gaps.max(),
             eta_last[6, 0], eta_last[6, 1]), flush=True)
    print("[item 5] 0.75/(m*H6) = %.17g (the stream gives jittered gaps, so this "
          "is not the value used)" % mean_eta, flush=True)
    return check("5", "elapsed-time weight and cap fraction", ok,
                 "max|eta_last[6] - gap/(m*H6)| %.3e; cap_frac_head[6] %.17g "
                 "(1/n_choices %.17g)" % (err, cap, 1.0 / n_ch))


def item6():
    out = {}
    for dt in (0.1, 0.05):
        r = toy.run_cell("L4", "F1", rho=4.0, T=3.0, sign=1, background=True,
                         contingent=True, dt=dt, m=10.0, eta_max=0.2,
                         beta=4.0, combo="max", n_choices=400)
        out[dt] = (r["median_N"], r["n_reached"], r["median_final_p"],
                   r["n_steps"])
    print("[item 6] dt 0.10: median N* %s reached %d/16 median final p %.6f "
          "steps %d" % out[0.1], flush=True)
    print("[item 6] dt 0.05: median N* %s reached %d/16 median final p %.6f "
          "steps %d" % out[0.05], flush=True)
    return check("6", "dt 0.05 vs 0.1 median N* printed", True,
                 "dt 0.10 median N* %s; dt 0.05 median N* %s"
                 % (out[0.1][0], out[0.05][0]))


def main():
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    items = {"1": item1, "2": item2, "3": item3, "4": item4, "5": item5,
             "6": item6}
    for key in which:
        ok = items[key]()
        if not ok:
            print("\nSELFTEST_C STOPPED at item %s (failure)" % key, flush=True)
            return 1
    print("\nSELFTEST_C SUMMARY: %d checks, 0 failed" % len(RESULTS), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
