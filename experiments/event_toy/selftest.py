"""Selftest items 1-6 of the W23 brief.  Prints every number it asserts."""

import sys
import time

import numpy as np

import toy
from toy import HORIZONS, Stream, run_cell

RESULTS = []


def check(item, name, ok, detail):
    RESULTS.append((item, name, bool(ok), detail))
    print("[item %s] %-34s %s :: %s" % (item, name, "PASS" if ok else "FAIL", detail),
          flush=True)
    return ok


# ---------------------------------------------------------------- item 1
def item1():
    ok_all = True
    for sign, label in ((1, "+1"), (-1, "-1")):
        t0 = time.time()
        r = run_cell("L1", "F1", rho=1.0, T=0.0, I=10.0, sign=sign,
                     background=False, contingent=True)
        reached300 = int(np.sum(np.asarray(r["N_star"]) <= 300))
        med = float(np.median(r["N_star"]))
        ok = reached300 >= 15
        ok_all &= check("1", "MUST LEARN sign %s" % label, ok,
                        "reached<=300: %d/16, median N*: %s, median final p: %.6f, wall %.1fs"
                        % (reached300, med, r["median_final_p"], time.time() - t0))
    return ok_all


# ---------------------------------------------------------------- item 2
def item2():
    t0 = time.time()
    r = run_cell("L2", "F1", rho=4.0, T=30.0, sign=1, background=True,
                 contingent=False, kappa=1.0)
    nre = r["n_reached"]
    mfp = r["median_final_p"]
    ok = (nre <= 2) and (0.25 <= mfp <= 0.75)
    return check("2", "MUST NOT LEARN broken stream", ok,
                 "reached: %d/16, median final p: %.6f, median N*: %s, wall %.1fs"
                 % (nre, mfp, r["median_N"], time.time() - t0))


# ---------------------------------------------------------------- item 3
def item3():
    t0 = time.time()
    r = run_cell("L1", "F1", rho=4.0, T=300.0, sign=1, background=False,
                 contingent=True)
    nre = r["n_reached"]
    ok = nre <= 1
    return check("3", "MUST NOT LEARN out of reach", ok,
                 "reached: %d/16, median final p: %.6f, median N*: %s, wall %.1fs"
                 % (nre, r["median_final_p"], r["median_N"], time.time() - t0))


# ---------------------------------------------------------------- item 4
def item4():
    ok = True
    for dt in (0.1, 0.05):
        g = np.exp(-dt / HORIZONS)
        err = np.abs(g ** (HORIZONS / dt) - np.exp(-1.0))
        ok &= check("4", "gamma_i^(h_i/dt)=exp(-1) dt=%.2f" % dt,
                    bool(np.all(err <= 1e-9)),
                    "max abs deviation %.3e (7 heads)" % err.max())
    for kappa in (0.043, 1.0):
        g = np.exp(-0.1 / HORIZONS)
        d = np.exp(-0.1 / (kappa * HORIZONS))
        print("[item 4] dt=0.1 kappa=%.3f gamma_i=%s" %
              (kappa, np.array2string(g, precision=12, separator=", ")), flush=True)
        print("[item 4] dt=0.1 kappa=%.3f decay_i=%s" %
              (kappa, np.array2string(d, precision=12, separator=", ")), flush=True)
    return ok


# ---------------------------------------------------------------- item 5
def item5():
    T, rho, seed = 30.0, 4.0, 0
    NMAX = 2_000_000
    ok = True

    # (i) intervals, with background on
    st = Stream(T=T, rho=rho, sign=1, background=True, contingent=True,
                seeds=[seed], dt=0.1, n_choices=2400)
    nsteps = min(st.n_steps, NMAX)
    off, L = st.offset, st.offset + 1
    buf = np.zeros(L)
    nextc = int(st.choice_steps[0, 0])
    ptr = 0
    k = st.K
    dt_t = st.dist_t[:, 0, :]
    dt_ptr = st.dist_ptr[0].copy()
    bt = st.bg_t[0, :]
    bv = st.bg_v[0, :]
    bp = 0
    choices = []
    rewards = np.zeros(nsteps)
    kk = np.arange(k)
    t0 = time.time()
    for t in range(nsteps):
        cd = dt_t[kk, dt_ptr]
        hit = cd == t
        dt_ptr += hit
        bhit = (bp < len(bt)) and (bt[bp] == t)
        r = buf[t % L]
        buf[t % L] = 0.0
        if bhit:
            r += bv[bp]
            bp += 1
        if nextc == t:
            choices.append(t)
            buf[(t + off) % L] += 1.0
            ptr += 1
            nextc = int(st.choice_steps[0, ptr]) if ptr < st.n_choices else toy.BIG
        rewards[t] = r
    lc = np.asarray(choices, dtype=np.int64)
    gap = float(np.diff(lc).mean() * 0.1)
    ok &= check("5", "mean choice gap", abs(gap - 120.0) <= 0.03 * 120.0,
                "%.4f s (expect 120 within 3%%), choices=%d, steps=%d, wall %.1fs"
                % (gap, len(lc), nsteps, time.time() - t0))
    for kk2 in range(k):
        ix = dt_t[kk2]
        ix = ix[ix < st.n_steps + 2]
        m = float(np.diff(ix).mean() * 0.1)
        ok &= check("5", "mean distractor interval k=%d" % (kk2 + 1),
                    abs(m - 120.0) <= 0.05 * 120.0,
                    "%.4f s (expect 120 within 5%%), n=%d" % (m, len(ix)))
    ix = bt[bt < st.n_steps + 2]
    m = float(np.diff(ix).mean() * 0.1)
    ok &= check("5", "mean background interval", abs(m - 120.0) <= 0.05 * 120.0,
                "%.4f s (expect 120 within 5%%), n=%d" % (m, len(ix)))

    # (ii) consequence offset, background off, a always 1
    st2 = Stream(T=T, rho=rho, sign=1, background=False, contingent=True,
                 seeds=[seed], dt=0.1, n_choices=2400)
    nsteps2 = min(st2.n_steps, NMAX)
    off2, L2 = st2.offset, st2.offset + 1
    buf = np.zeros(L2)
    nextc = int(st2.choice_steps[0, 0])
    ptr = 0
    dt_t = st2.dist_t[:, 0, :]
    dt_ptr = st2.dist_ptr[0].copy()
    choices2 = []
    rewards2 = np.zeros(nsteps2)
    for t in range(nsteps2):
        cd = dt_t[kk, dt_ptr]
        dt_ptr += (cd == t)
        r = buf[t % L2]
        buf[t % L2] = 0.0
        if nextc == t:
            choices2.append(t)
            buf[(t + off2) % L2] += 1.0
            ptr += 1
            nextc = int(st2.choice_steps[0, ptr]) if ptr < st2.n_choices else toy.BIG
        rewards2[t] = r
    assert off2 == 300, off2
    mism = 0
    checked = 0
    for tc in choices2:
        if tc + off2 < nsteps2:
            checked += 1
            if rewards2[tc + off2] != 1.0:
                mism += 1
    delivered = float(rewards2.sum())
    ok &= check("5", "consequence +1 at choice+300", mism == 0 and delivered == checked,
                "choices=%d, checked=%d, mismatches=%d, delivered=%.1f"
                % (len(choices2), checked, mism, delivered))
    return ok


# ---------------------------------------------------------------- item 6
def item6():
    t0 = time.time()
    a = run_cell("L1", "F1", rho=4.0, T=30.0, sign=1, background=True,
                 contingent=True, record_events=True, max_steps=100_000)
    b = run_cell("L3", "F1", rho=4.0, T=30.0, sign=1, background=True,
                 contingent=True, record_events=True, max_steps=100_000)
    same = True
    nd = 0
    for k in range(4):
        ea, eb = a["events"]["dist"][k], b["events"]["dist"][k]
        nd += len(ea)
        same &= (ea == eb)
    nb = len(a["events"]["bg"])
    same &= (a["events"]["bg"] == b["events"]["bg"])
    return check("6", "SAME STREAM L1 vs L3", same,
                 "identical distractor and background indices over 100000 steps "
                 "(distractor events=%d, background events=%d), wall %.1fs"
                 % (nd, nb, time.time() - t0))


def main():
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    if "1" in which:
        item1()
    if "2" in which:
        item2()
    if "3" in which:
        item3()
    if "4" in which:
        item4()
    if "5" in which:
        item5()
    if "6" in which:
        item6()
    bad = [r for r in RESULTS if not r[2]]
    print("\nSELFTEST SUMMARY: %d checks, %d failed" % (len(RESULTS), len(bad)), flush=True)
    for r in bad:
        print("  FAILED item %s: %s :: %s" % r, flush=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
