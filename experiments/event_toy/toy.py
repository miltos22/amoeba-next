"""Event-stream toy: one unbroken stream, four learners, F1/F2 features.

Self-contained numpy. See sections 2-5 of the design.

Update order inside one step t (design 4.6), identical for all learners:
  1. read x_t and the reward r_t delivered at step t; if t is a choice step
     p = sigmoid(theta), sample a, g_t = a - p, else g_t = 0.
  2. V_t = w . x_t with the weights as they are now.
  3. if t > 0: delta = r_t + gamma*V_t - V_prev  (L3: r_t - rbar + V_t - V_prev).
     Apply critic and policy updates with the traces as they stand now (they
     contain deposits up to step t-1 only).  For L3 then update rbar.
  4. traces: z <- decay * z + deposit of step t (x_t critic, g_t policy).
  5. store V_prev = w . x_t computed with the UPDATED weights.
  6. F2 only: update the memories with step t's events (so the choice made at
     step t appears in x from step t+1).

Bounded step (design 4.5), applied separately to each trace vector:
  for param += alpha * delta * z, let m = alpha * 2.0 * max(|delta|,1) * sum|z|;
  if m > 1 use alpha/m in place of alpha, else alpha.
"""

import math

import numpy as np

NS = 16

# 7 horizons of L2 / L3 time constants: 10, 31.62, ..., 10000 s
HORIZONS = 10.0 ** (0.5 * np.arange(1.0, 8.0))
# 7 memory time constants of F2: 1, 3.162, ..., 1000 s
TAUS = 10.0 ** (0.5 * np.arange(0.0, 7.0))

BIG = np.int64(1) << 60


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _poisson_indices(rng, n_steps, p, chunk=8192):
    """Indices of a Bernoulli(p)-per-step process over steps 0..n_steps-1."""
    last = -1
    parts = []
    while True:
        g = rng.geometric(p, size=chunk)
        cs = last + np.cumsum(g)
        ok = cs < n_steps
        if ok.all():
            parts.append(cs)
            last = int(cs[-1])
        else:
            if ok.any():
                parts.append(cs[ok])
            break
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


class Stream:
    """The shared stream of design section 2.

    Everything is generated on the 0.1 s coarse grid; at dt = 0.05 every coarse
    step index is doubled (same stream, twice the resolution).
    """

    def __init__(self, T, rho, sign, background, contingent, seeds, dt=0.1,
                 I=None, n_choices=400):
        self.dt = float(dt)
        M = int(round(0.1 / self.dt))
        assert M in (1, 2), "dt must be 0.1 or 0.05"
        self.M = M
        base = 0.1
        self.I = float(I) if I is not None else float(rho) * float(T)
        assert self.I > 0.0
        ns = len(seeds)
        self.ns = ns
        self.sign = float(sign)
        self.background = bool(background)
        self.contingent = bool(contingent)
        self.T = float(T)
        self.rho = float(rho)
        self.n_choices = int(n_choices)
        # offset of the consequence in steps; T == 0 means "on the very next step"
        off_c = int(round(T / base))
        if off_c < 1:
            off_c = 1
        self.offset_coarse = off_c
        self.offset = off_c * M
        p = base / self.I
        K = 4
        self.K = K

        rngs = [np.random.default_rng(int(s)) for s in seeds]
        choice_c = np.empty((ns, self.n_choices), dtype=np.int64)
        for si in range(ns):
            gaps_s = rngs[si].uniform(0.5 * self.I, 1.5 * self.I, size=self.n_choices)
            gstep = np.maximum(1, np.rint(gaps_s / base).astype(np.int64))
            choice_c[si] = np.cumsum(gstep)
        n_coarse = int(choice_c[:, -1].max()) + 1
        self.n_coarse = n_coarse

        dist = [[None] * ns for _ in range(K)]
        bg = [None] * ns
        bgv = [None] * ns
        for si in range(ns):
            for k in range(K):
                dist[k][si] = _poisson_indices(rngs[si], n_coarse, p)
            bg[si] = _poisson_indices(rngs[si], n_coarse, p)
            bgv[si] = rngs[si].choice(np.array([-1.0, 1.0]), size=len(bg[si]))

        self.choice_steps = choice_c * M  # (ns, n_choices), fine steps
        self.n_steps = int(self.choice_steps[:, -1].max()) + 1
        sentinel = np.int64(BIG)  # never matches a step, also beyond a test tail

        maxn = max(len(x) for k in range(K) for x in dist[k]) + 1
        self.dist_t = np.full((K, ns, maxn), sentinel, dtype=np.int64)
        self.dist_v = np.zeros((K, ns, maxn), dtype=np.float64)
        for k in range(K):
            for si in range(ns):
                ix = dist[k][si]
                if len(ix):
                    self.dist_t[k, si, :len(ix)] = ix * M
                    self.dist_v[k, si, :len(ix)] = 1.0
        self.dist_ptr = np.zeros((ns, K), dtype=np.int64)
        self.maxn_dist = maxn

        maxb = max(len(x) for x in bg) + 1
        self.bg_t = np.full((ns, maxb), sentinel, dtype=np.int64)
        self.bg_v = np.zeros((ns, maxb), dtype=np.float64)
        for si in range(ns):
            ix = bg[si]
            if len(ix):
                self.bg_t[si, :len(ix)] = ix * M
                self.bg_v[si, :len(ix)] = bgv[si]
        self.bg_ptr = np.zeros(ns, dtype=np.int64)
        self.maxn_bg = maxb

    def count_summary(self):
        return {
            "n_choice_events": int(self.ns * self.n_choices),
            "n_distractor_events": int((self.dist_v > 0).sum()),
            "n_background_events": int((self.bg_v != 0).sum()),
            "n_coarse_steps": int(self.n_coarse),
            "n_steps": int(self.n_steps),
        }


def _nstar(prow, sign, n_choices):
    """Smallest 1-based k with mean(p_k..p_{k+19}) >= 0.9 (sign +1) or <= 0.1."""
    if not np.all(np.isfinite(prow)):
        return n_choices + 1
    cs = np.concatenate(([0.0], np.cumsum(prow)))
    w = (cs[20:] - cs[:-20]) / 20.0
    if sign > 0:
        hit = np.nonzero(w >= 0.9)[0]
    else:
        hit = np.nonzero(w <= 0.1)[0]
    if len(hit):
        return int(hit[0] + 1)
    return n_choices + 1


def run_cell(learner, features, rho, T, sign=1, background=True, contingent=True,
             dt=0.1, alpha_p=0.1, alpha_v=0.02, kappa=1.0, n_choices=400,
             seeds=range(NS), I=None, record_events=False, max_steps=None,
             force_action=None, tau_r=None, cv=None, mu_c=0.5,
             _force_mu_new=False, m=10.0, eta_max=0.2, beta=4.0,
             combo="max", _rb_off=False, _debug=False, _tail_steps=0):
    """Run one cell (one learner/features/stream configuration, 16 seeds).

    Returns a dict of results (JSON-serialisable except numpy scalars, which are
    converted).
    """
    seeds = list(seeds)
    ns = len(seeds)
    stream = Stream(T=T, rho=rho, sign=sign, background=background,
                    contingent=contingent, seeds=seeds, dt=dt, I=I,
                    n_choices=n_choices)
    F2 = (features == "F2")
    has_bg = bool(background) and stream.count_summary()["n_background_events"] > 0
    D = 48 if F2 else 6
    alpha_vp = alpha_v * (dt / 0.1)  # critic step sizes scale with dt; policy does not
    eta = 0.01 * (dt / 0.1)
    K = 4
    H = 7
    off = stream.offset
    L = off + 1
    buf = np.zeros((ns, L), dtype=np.float64)
    signf = float(sign)

    choice_steps = stream.choice_steps
    next_choice = choice_steps[:, 0].astype(np.int64).copy()
    cptr = np.zeros(ns, dtype=np.int64)
    dist_t = stream.dist_t
    dist_ptr = stream.dist_ptr
    bg_t = stream.bg_t
    bg_v = stream.bg_v
    bg_ptr = stream.bg_ptr
    seed_ar = np.arange(ns)
    k_ar = np.arange(K)[None, :]
    seed_col = seed_ar[:, None]

    rngs_a = [np.random.default_rng(int(s) + 10_000) for s in seeds]

    p_rec = np.full((ns, n_choices), np.nan)
    done = np.zeros(ns, dtype=bool)

    x = np.empty((ns, D), dtype=np.float64)
    x[:, 0] = 1.0
    if F2:
        m = np.zeros((ns, 6, 7), dtype=np.float64)
        mdecay = np.exp(-dt / TAUS)  # (7,)
        ev = np.empty((ns, 6), dtype=np.float64)
    a = np.zeros(ns, dtype=np.float64)
    g = np.zeros(ns, dtype=np.float64)

    # ---- learner state -------------------------------------------------
    if learner == "L1":
        gamma = 0.99 ** (dt / 0.1)
        lam = 0.8 ** (dt / 0.1)
        decay = gamma * lam
        w = np.zeros((ns, D))
        zv = np.zeros((ns, D))
        zp = np.zeros(ns)
        theta = np.zeros(ns)
    elif learner == "L2":
        gammas = np.exp(-dt / HORIZONS)
        decays = np.exp(-dt / (float(kappa) * HORIZONS))
        W = np.zeros((H, ns, D))
        Zv = np.zeros((H, ns, D))
        Zp = np.zeros((H, ns))
        theta = np.zeros(ns)
        rbar = np.zeros(ns)            # one running average, shared by all heads
        alpha_i = (cv * dt / HORIZONS).reshape(H, 1) if cv is not None else None
        bch = np.zeros(H)              # bound-active counts per head
        contrib = np.zeros((H, ns))    # per-head policy contributions
    elif learner == "L3":
        decays = np.exp(-dt / HORIZONS)
        w = np.zeros((ns, D))
        Zv = np.zeros((H, ns, D))
        Zp = np.zeros((H, ns))
        theta = np.zeros(ns)
        rbar = np.zeros(ns)
    elif learner == "L4":
        assert features == "F1", "L4 uses F1 only"
        gammas = np.exp(-dt / HORIZONS)
        Q = np.zeros((H, ns, 2))
        E = np.zeros((H, ns, 2))
        rb = np.zeros((H, ns))
        t_last = np.full(ns, -1, dtype=np.int64)
        cap_count = np.zeros(H)
        cap_choices = 0
        eta_last = np.zeros((H, ns))
        gap_last = np.zeros(ns)
    else:
        raise ValueError(learner)

    Vprev = np.zeros((H, ns)) if learner == "L2" else np.zeros(ns)
    bound_c = 0
    bound_p = 0
    tot_c = 0
    tot_p = 0
    diverged = np.zeros(ns, dtype=bool)

    n_steps = stream.n_steps
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    if _tail_steps:  # test only: keep stepping so pending consequences are read
        n_steps += int(_tail_steps)

    rec = {"dist": [[] for _ in range(K)], "bg": []} if record_events else None

    err = np.errstate(divide="ignore", over="ignore")
    err.__enter__()
    for t in range(n_steps):
        # ---- 1. read x_t and r_t --------------------------------------
        fire = next_choice == t
        # distractors of step t
        curd = dist_t[k_ar, seed_col, dist_ptr]
        hitd = curd == t
        d = hitd
        dist_ptr += hitd
        # background rewards of step t
        curb = bg_t[seed_ar, bg_ptr]
        bghit = curb == t
        r = buf[:, t % L].copy()
        buf[:, t % L] = 0.0
        if has_bg and bghit.any():
            r += bg_v[seed_ar, bg_ptr] * bghit
        bg_ptr += bghit
        if rec is not None:
            for k in range(K):
                ii = np.nonzero(d[:, k])[0]
                for i in ii:
                    rec["dist"][k].append((int(seeds[i]), int(t)))
            ii = np.nonzero(bghit)[0]
            for i in ii:
                rec["bg"].append((int(seeds[i]), int(t)))

        # x_t is built from the memories as they stand (through step t-1)
        x[:, 1] = fire
        x[:, 2:6] = d
        if F2:
            x[:, 6:] = m.reshape(ns, 42)

        if fire.any():
            idxs = np.nonzero(fire)[0]
            if learner == "L4":
                Delta_now = Q[:, :, 1] - Q[:, :, 0]
                if combo == "max":
                    hsel = np.argmax(np.abs(Delta_now), axis=0)
                    d_now = Delta_now[hsel, np.arange(ns)]
                else:
                    d_now = Delta_now.mean(0)
                p = sigmoid(beta * d_now)
            else:
                p = sigmoid(theta)
            if force_action is None:
                for i in idxs:
                    a[i] = 1.0 if rngs_a[i].random() < p[i] else 0.0
            else:
                a[idxs] = float(force_action)
            if learner != "L4":
                g[idxs] = a[idxs] - p[idxs]
            p_rec[idxs, cptr[idxs]] = p[idxs]
            cptr[idxs] += 1
            nxt = cptr[idxs]
            fin = nxt >= n_choices
            nxtc = np.where(fin, BIG,
                            choice_steps[idxs, np.minimum(nxt, n_choices - 1)])
            next_choice[idxs] = nxtc
            # schedule the consequence
            if stream.contingent:
                sched = idxs[a[idxs] > 0.5]
            else:
                sched = idxs
            if len(sched):
                buf[sched, (t + off) % L] += signf
            if learner == "L4":
                # elapsed-time weight and the Q/E state update (design step 2.4-2.5)
                nf = len(idxs)
                el = np.where(t_last[idxs] < 0, np.inf,
                              (t - t_last[idxs]) * dt)
                raw = el[None, :] / (m * HORIZONS)[:, None]
                eta_f = np.minimum(eta_max, raw)
                cap_count += (raw > eta_max).sum(1)
                cap_choices += nf
                onehot = np.zeros((nf, 2))
                onehot[np.arange(nf), a[idxs].astype(np.int64)] = 1.0
                fac = 1.0 - eta_f[:, :, None] * onehot[None, :, :]
                Q[:, idxs, :] = Q[:, idxs, :] * fac
                E[:, idxs, :] = (E[:, idxs, :]
                                 + eta_f[:, :, None] * onehot[None, :, :])
                t_last[idxs] = t
                eta_last[:, idxs] = eta_f
                gap_last[idxs] = el

        # ---- 2. V_t ----------------------------------------------------
        if learner == "L4":
            pass
        elif learner in ("L1", "L3"):
            V = np.einsum("nd,nd->n", w, x)
        else:
            V = np.einsum("hnd,nd->hn", W, x)

        # ---- 3. updates with traces as they stand ----------------------
        if learner == "L4":
            # action-outcome averages; reward of step t is read here, after
            # the choice update of step 2, so a same-step reward is credited
            rc = r[None, :] - rb
            Q += E * rc[:, :, None]
            E *= gammas[:, None, None]
            if not _rb_off:
                rb += (dt / (m * HORIZONS))[:, None] * (r[None, :] - rb)
        if learner != "L4" and t > 0:
            if learner == "L1":
                delta = r + gamma * V - Vprev
                mx = np.maximum(np.abs(delta), 1.0)
                mval = alpha_vp * 2.0 * mx * zv.sum(1)
                active = mval > 1.0
                ae = np.where(active, alpha_vp / mval, alpha_vp)
                w += (ae * delta)[:, None] * zv
                bound_c += int(active.sum())
                tot_c += ns
                mval = alpha_p * 2.0 * mx * np.abs(zp)
                active = mval > 1.0
                aep = np.where(active, alpha_p / mval, alpha_p)
                theta += aep * delta * zp
                bound_p += int(active.sum())
                tot_p += ns
            elif learner == "L2":
                if tau_r is None:
                    delta = r[None, :] + gammas[:, None] * V - Vprev
                else:
                    delta = (r - rbar)[None, :] + gammas[:, None] * V - Vprev
                mx = np.maximum(np.abs(delta), 1.0)
                if mu_c == 0.5 and not _force_mu_new:
                    # old bounded step; bit-identical to the earlier code when cv is None
                    if cv is None:
                        mval = alpha_vp * 2.0 * mx * Zv.sum(2)
                        active = mval > 1.0
                        ae = np.where(active, alpha_vp / mval, alpha_vp)
                    else:
                        mval = alpha_i * 2.0 * mx * Zv.sum(2)
                        active = mval > 1.0
                        ae = np.where(active, alpha_i / mval, alpha_i)
                else:
                    s = mx * Zv.sum(2)
                    if cv is None:
                        mval = alpha_vp * s
                        active = mval > mu_c
                        ae = np.where(active, mu_c / s, alpha_vp)
                    else:
                        mval = alpha_i * s
                        active = mval > mu_c
                        ae = np.where(active, mu_c / s, alpha_i)
                W += (ae * delta)[:, :, None] * Zv
                bound_c += int(active.sum())
                bch += active.sum(1)
                tot_c += H * ns
                mval = alpha_p * 2.0 * mx * np.abs(Zp)
                active = mval > 1.0
                aep = np.where(active, alpha_p / mval, alpha_p)
                theta += (aep * delta * Zp).sum(0) / H
                contrib += (aep * delta * Zp) / H
                bound_p += int(active.sum())
                tot_p += H * ns
                if tau_r is not None:
                    rbar += (dt / tau_r) * (r - rbar)
            else:  # L3
                delta = r - rbar + V - Vprev
                mx = np.maximum(np.abs(delta), 1.0)
                mval = alpha_vp * 2.0 * mx * Zv.sum(2)
                active = mval > 1.0
                ae = np.where(active, alpha_vp / mval, alpha_vp)
                w += ((ae * delta)[:, :, None] * Zv).sum(0) / H
                bound_c += int(active.sum())
                tot_c += H * ns
                mval = alpha_p * 2.0 * mx * np.abs(Zp)
                active = mval > 1.0
                aep = np.where(active, alpha_p / mval, alpha_p)
                theta += (aep * delta * Zp).sum(0) / H
                bound_p += int(active.sum())
                tot_p += H * ns
                rbar += eta * delta
                if t % 4096 == 0:
                    diverged |= ~(np.isfinite(rbar) & np.isfinite(theta))

        # ---- 4. traces ------------------------------------------------
        if learner == "L1":
            zv *= decay
            zv += x
            zp *= decay
            zp += g
        elif learner == "L2":
            Zv *= decays[:, None, None]
            Zv += x[None, :, :]
            Zp *= decays[:, None]
            Zp += g[None, :]
        elif learner == "L3":
            Zv *= decays[:, None, None]
            Zv += x[None, :, :]
            Zp *= decays[:, None]
            Zp += g[None, :]

        if fire.any():
            g[fire] = 0.0

        # ---- 5. V_prev with the UPDATED weights ------------------------
        if learner in ("L1", "L3"):
            Vprev = np.einsum("nd,nd->n", w, x)
        elif learner == "L2":
            Vprev = np.einsum("hnd,nd->hn", W, x)

        # ---- 6. F2 memories with step t's events -----------------------
        if F2:
            ev[:, 0] = fire * (a > 0.5)
            ev[:, 1] = fire * (a <= 0.5)
            ev[:, 2:6] = d
            m *= mdecay[None, None, :]
            np.maximum(m, ev[:, :, None], out=m)

        if t % 4096 == 0:
            if learner == "L4":
                diverged |= ~np.isfinite(Q).all(axis=(0, 2))
            elif learner == "L2":
                diverged |= ~(np.isfinite(theta) &
                              np.isfinite(W).all(axis=(0, 2)))
            else:
                diverged |= ~(np.isfinite(theta) & np.isfinite(w).all(1))

    err.__exit__(None, None, None)

    # ---------------- measure -------------------------------------------
    N_star = np.empty(ns, dtype=np.int64)
    final_p = np.empty(ns)
    last = n_choices
    first = max(0, last - 20)
    for si in range(ns):
        row = p_rec[si]
        bad = (not np.all(np.isfinite(row)))
        if learner == "L2":
            bad = bad or (not np.all(np.isfinite(W[:, si, :])))
        elif learner == "L4":
            bad = bad or (not np.all(np.isfinite(Q[:, si, :])))
        else:
            bad = bad or (not np.all(np.isfinite(w[si])))
        if learner != "L4":
            bad = bad or (not np.isfinite(theta[si]))
        if bad:
            diverged[si] = True
        N_star[si] = (n_choices + 1) if diverged[si] else _nstar(row, sign, n_choices)
        seg = row[np.isfinite(row)]
        if len(seg) >= 20:
            final_p[si] = float(np.mean(seg[-20:]))
        elif len(seg) > 0:
            final_p[si] = float(np.mean(seg))
        else:
            final_p[si] = float("nan")

    n_star_list = [int(v) for v in N_star]
    reached = int(np.sum(N_star <= n_choices))
    q25, q50, q75 = np.percentile(N_star, [25, 50, 75])

    out = {
        "learner": learner,
        "kappa": (float(kappa) if learner == "L2" else None),
        "features": features,
        "rho": float(rho),
        "T": float(T),
        "sign": int(sign),
        "background": bool(background),
        "contingent": bool(contingent),
        "dt": float(dt),
        "alpha_p": float(alpha_p),
        "alpha_v": float(alpha_v),
        "I": float(stream.I),
        "n_choices": int(n_choices),
        "tau_r": (None if tau_r is None else float(tau_r)),
        "cv": (None if cv is None else float(cv)),
        "mu_c": float(mu_c),
        "seeds": [int(s) for s in seeds],
        "N_star": n_star_list,
        "final_p": [float(v) for v in final_p],
        "diverged": [bool(v) for v in diverged],
        "median_N": float(q50),
        "q25_N": float(q25),
        "q75_N": float(q75),
        "n_reached": reached,
        "median_final_p": float(np.median(final_p)),
        "n_diverged": int(diverged.sum()),
        "bound_frac_critic": (float(bound_c) / tot_c) if tot_c else 0.0,
        "bound_frac_policy": (float(bound_p) / tot_p) if tot_p else 0.0,
        "n_steps": int(n_steps),
        "stream": stream.count_summary(),
    }
    if learner == "L4":
        out["m"] = float(m)
        out["eta_max"] = float(eta_max)
        out["beta"] = float(beta)
        out["combo"] = combo
        out["delta_end"] = [[float(Q[i, si, 1] - Q[i, si, 0])
                             for si in range(ns)] for i in range(H)]
        out["cap_frac_head"] = [float(v) for v in
                                (cap_count / cap_choices if cap_choices
                                 else np.zeros(H))]
        out["rb_end"] = [[float(v) for v in rb[i]] for i in range(H)]
        if _debug:
            out["q_end"] = [[[float(v) for v in Q[i, si, :]]
                             for si in range(ns)] for i in range(H)]
            out["eta_last"] = [[float(v) for v in eta_last[i]]
                               for i in range(H)]
            out["gap_last"] = [float(v) for v in gap_last]
    if learner in ("L2", "L4"):
        out["n_high"] = int(np.sum(final_p >= 0.9))
        out["n_low"] = int(np.sum(final_p <= 0.1))
        stuck = 0
        for si in range(ns):
            tail = p_rec[si][-20:]
            if np.all(tail >= 0.999) or np.all(tail <= 0.001):
                stuck += 1
        out["n_stuck"] = stuck
        out["p_rec"] = p_rec
    if learner == "L2":
        den = float(tot_c) / H if tot_c else 0.0
        out["bound_frac_head"] = [float(v) for v in
                                  (bch / den if den else np.zeros(H))]
        out["head_contrib"] = [[float(v) for v in contrib[i]] for i in range(H)]
        out["rbar_end"] = [float(v) for v in rbar]
        out["theta_end"] = [float(v) for v in theta]
    if rec is not None:
        out["events"] = rec
    return out


def cell_id(p):
    if p["learner"] == "L4":
        return ("L4|m=%s|eta=%s|beta=%s|combo=%s|rho=%s|T=%s|sign=%s|bg=%d"
                "|cont=%d|dt=%s|n=%s"
                % (p["m"], p["eta_max"], p["beta"], p["combo"], p["rho"],
                   p["T"], p["sign"], int(p["background"]),
                   int(p["contingent"]), p["dt"], p["n_choices"]))
    return "|".join([
        str(p["learner"]),
        "k=%s" % ("NA" if (p["learner"] != "L2" or p.get("kappa") is None)
                  else p["kappa"]),
        str(p["features"]),
        "rho=%s" % p["rho"],
        "T=%s" % p["T"],
        "sign=%s" % p["sign"],
        "bg=%d" % int(p["background"]),
        "cont=%d" % int(p["contingent"]),
        "dt=%s" % p["dt"],
        "ap=%s" % p["alpha_p"],
        "av=%s" % p["alpha_v"],
        "tr=%s" % ("NA" if p.get("tau_r") is None else p["tau_r"]),
        "cv=%s" % ("NA" if p.get("cv") is None else p["cv"]),
        "mu=%s" % p.get("mu_c", 0.5),
        "n=%s" % p["n_choices"],
    ])
