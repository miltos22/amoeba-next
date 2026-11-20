"""Far-value readout on the unit-buffer record stream (revision 6).

Records are indexed 1..M in the definitions; arrays here are 0-based
positions 0..M-1 (position p = record p+1).  A retrieved item is a
(anchor position, band k, match weight) triple; bands cover positions
p+1+2^(k-1) .. p+2^k (count 2^(k-1)) and are mature iff the end position is
<= M-1 (p <= M-2^k-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
KMAX = 14


def energy_bin(E):
    b = np.searchsorted(EDGES, np.asarray(E, dtype=np.float64), side="right") - 1
    return np.clip(b, 0, 3)


@dataclass
class Stream:
    U: np.ndarray
    dP: np.ndarray
    E_start: np.ndarray
    t_start: np.ndarray
    t_end: np.ndarray

    def __post_init__(self):
        self.U = np.asarray(self.U, dtype=np.float64)
        self.dP = np.asarray(self.dP, dtype=np.float64)
        self.E_start = np.asarray(self.E_start, dtype=np.float64)
        self.t_start = np.asarray(self.t_start, dtype=np.float64)
        self.t_end = np.asarray(self.t_end, dtype=np.float64)
        self.M = len(self.U)
        self.prefU = np.concatenate(([0.0], np.cumsum(self.U)))
        self.prefDP = np.concatenate(([0.0], np.cumsum(self.dP)))
        self.bin = energy_bin(self.E_start)
        self.median_dP = float(np.median(self.dP)) if self.M else 0.0

    def with_U(self, U):
        return Stream(U, self.dP, self.E_start, self.t_start, self.t_end)


def band_Y(stream, k):
    q = 2 ** k
    half = 2 ** (k - 1)
    Mv = stream.M - q
    if Mv <= 0:
        return np.empty(0), np.empty(0)
    i = np.arange(Mv, dtype=np.int64)
    Y = stream.prefU[i + q + 1] - stream.prefU[i + half + 1]
    dP = stream.prefDP[i + q + 1] - stream.prefDP[i + half + 1]
    return Y, dP


def compute_ubar(stream, k, targets, mode="phys"):
    q = 2 ** k
    half = 2 ** (k - 1)
    Mv = stream.M - q
    targets = np.asarray(targets, dtype=np.int64)
    if mode == "none":
        Yt = stream.prefU[targets + q + 1] - stream.prefU[targets + half + 1]
        dPt = stream.prefDP[targets + q + 1] - stream.prefDP[targets + half + 1]
        z = np.zeros(len(targets))
        return z, Yt, dPt
    if Mv <= 0:
        z = np.zeros(len(targets))
        return z, z.copy(), z.copy()
    i = np.arange(Mv, dtype=np.int64)
    Y = stream.prefU[i + q + 1] - stream.prefU[i + half + 1]
    if mode == "phi":
        dP = np.full(Mv, float(half))
        median_dP = 1.0
    else:
        dP = stream.prefDP[i + q + 1] - stream.prefDP[i + half + 1]
        median_dP = stream.median_dP
    bin_i = stream.bin[:Mv]
    PB = []
    DB = []
    for b in range(4):
        m = (bin_i == b)
        PB.append(np.concatenate(([0.0], np.cumsum(np.where(m, Y, 0.0)))))
        DB.append(np.concatenate(([0.0], np.cumsum(np.where(m, dP, 0.0)))))
    PtY = np.concatenate(([0.0], np.cumsum(Y)))
    PtD = np.concatenate(([0.0], np.cumsum(dP)))
    D0 = half * median_dP * 10.0
    ub = np.empty(len(targets), dtype=np.float64)
    for t in range(len(targets)):
        p = int(targets[t])
        b = int(stream.bin[p])
        left_end = min(p - half, Mv - 1)
        right_start = p + half
        right_end = Mv - 1

        def sums(arr):
            s = 0.0
            if left_end >= 0:
                s += arr[left_end + 1]
            if right_start <= right_end:
                s += arr[right_end + 1] - arr[right_start]
            return s

        sy = sums(PB[b])
        sd = sums(DB[b])
        sy_all = sums(PtY)
        sd_all = sums(PtD)
        ubar_all = sy_all / sd_all if sd_all > 0 else 0.0
        den = sd + D0
        ub[t] = (sy + D0 * ubar_all) / den if den > 0 else 0.0
    Yt = stream.prefU[targets + q + 1] - stream.prefU[targets + half + 1]
    if mode == "phi":
        dPt = np.full(len(targets), float(half))
    else:
        dPt = stream.prefDP[targets + q + 1] - stream.prefDP[targets + half + 1]
    return ub, Yt, dPt


def ubar_dict(stream, anchors, ks, mode="phys"):
    out = {}
    anchors = np.asarray(anchors, dtype=np.int64)
    ks = np.asarray(ks, dtype=np.int64)
    for k in sorted(set(ks.tolist())):
        sel = anchors[(ks == k) & (anchors <= stream.M - 2 ** k - 1)]
        if len(sel):
            ub, _, _ = compute_ubar(stream, k, sel, mode)
            for p, v in zip(sel.tolist(), ub.tolist()):
                out[(int(k), int(p))] = float(v)
    return out


def _prep(stream, anchors, ks, weights):
    anchors = np.asarray(anchors, dtype=np.int64)
    ks = np.asarray(ks, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    return anchors, ks, weights


def _W_by_k(anchors, ks, weights, stream, only_k=None):
    W = {}
    for k in sorted(set(ks.tolist())):
        if only_k is not None and k != only_k:
            continue
        m = (ks == k) & (anchors <= stream.M - 2 ** k - 1)
        W[k] = float(weights[m].sum())
    return W


def _band_inputs(stream, p, k, mode):
    q = 2 ** k
    half = 2 ** (k - 1)
    Y = float(stream.prefU[p + q + 1] - stream.prefU[p + half + 1])
    if mode == "phi":
        dP = float(half)
    else:
        dP = float(stream.prefDP[p + q + 1] - stream.prefDP[p + half + 1])
    return Y, dP


def v_unp(stream, anchors, ks, weights, n0, ub, mode="phys"):
    anchors, ks, weights = _prep(stream, anchors, ks, weights)
    W = _W_by_k(anchors, ks, weights, stream)
    per = {}
    total = 0.0
    for k in sorted(set(ks.tolist())):
        m = (ks == k) & (anchors <= stream.M - 2 ** k - 1)
        if not np.any(m):
            per[k] = 0.0
            continue
        w = weights[m] / (n0 + W[k])
        vk = 0.0
        for p, wk in zip(anchors[m].tolist(), w.tolist()):
            Y, dP = _band_inputs(stream, p, k, mode)
            if mode == "none":
                R = Y
            else:
                R = Y - ub.get((k, p), 0.0) * dP
            vk += wk * R
        per[k] = vk
        total += vk
    return total, per


def v_b1(stream, anchors, ks, weights, n0, ub, mode="phys", kband=None,
         broken=False):
    anchors, ks, weights = _prep(stream, anchors, ks, weights)
    W = _W_by_k(anchors, ks, weights, stream)
    pos, da, db = [], [], []
    for p, k, e in zip(anchors.tolist(), ks.tolist(), weights.tolist()):
        if kband is not None and k != kband:
            continue
        if p > stream.M - 2 ** k - 1:
            continue
        q = 2 ** k
        half = 2 ** (k - 1)
        w = e / (n0 + W[k])
        u = 0.0 if mode == "none" else ub.get((k, p), 0.0)
        l = p + 1 + half
        r = p + q + 1
        pos.append(l)
        da.append(w)
        db.append(w * u)
        pos.append(r)
        da.append(-w)
        db.append(-w * u)
    if not pos:
        return 0.0
    pos = np.asarray(pos, dtype=np.int64)
    da = np.asarray(da, dtype=np.float64)
    db = np.asarray(db, dtype=np.float64)
    order = np.argsort(pos, kind="stable")
    pos, da, db = pos[order], da[order], db[order]
    if mode == "phi":
        prefDP = np.arange(stream.M + 1, dtype=np.float64)
    elif mode == "none":
        prefDP = None
    else:
        prefDP = stream.prefDP
    V = 0.0
    a = 0.0
    b = 0.0
    prev = 0
    i = 0
    n = len(pos)
    while i < n:
        x = int(pos[i])
        if x > prev:
            Up = stream.prefU[x] - stream.prefU[prev]
            Dp = 0.0 if prefDP is None else prefDP[x] - prefDP[prev]
            if a > 0.0:
                s = 1.0 if broken else min(1.0, 1.0 / a)
                V += s * (a * Up - b * Dp)
        while i < n and int(pos[i]) == x:
            a += da[i]
            b += db[i]
            i += 1
        prev = x
    if prev < stream.M:
        Up = stream.prefU[stream.M] - stream.prefU[prev]
        Dp = 0.0 if prefDP is None else prefDP[stream.M] - prefDP[prev]
        if a > 0.0:
            s = 1.0 if broken else min(1.0, 1.0 / a)
            V += s * (a * Up - b * Dp)
    return V




def v_b1_fast(stream, anchors, ks, weights, n0, ub, mode="phys",
              kband=None, broken=False):
    """Equivalent to v_b1 using difference arrays instead of an event sweep."""
    anchors, ks, weights = _prep(stream, anchors, ks, weights)
    W = _W_by_k(anchors, ks, weights, stream)
    M = stream.M
    dl, dr, wl, wr, ul, ur = [], [], [], [], [], []
    for p, k, e in zip(anchors.tolist(), ks.tolist(), weights.tolist()):
        if kband is not None and k != kband:
            continue
        if p > M - 2 ** k - 1:
            continue
        q = 2 ** k
        half = 2 ** (k - 1)
        w = e / (n0 + W[k])
        u = 0.0 if mode == "none" else ub.get((k, p), 0.0)
        l, r = p + 1 + half, p + q + 1
        dl.append(l); dr.append(r)
        wl.append(w); wr.append(-w)
        ul.append(w * u); ur.append(-w * u)
    if not dl:
        return 0.0
    da = (np.bincount(np.asarray(dl), weights=np.asarray(wl), minlength=M + 1)
          + np.bincount(np.asarray(dr), weights=np.asarray(wr), minlength=M + 1))
    db = (np.bincount(np.asarray(dl), weights=np.asarray(ul), minlength=M + 1)
          + np.bincount(np.asarray(dr), weights=np.asarray(ur), minlength=M + 1))
    a = np.cumsum(da[:M])
    b = np.cumsum(db[:M])
    if broken:
        s = np.where(a > 0, 1.0, 0.0)
    else:
        s = np.where(a > 0, np.minimum(1.0, 1.0 / np.maximum(a, 1e-300)), 0.0)
    if mode == "phi":
        dP = np.ones(M)
    elif mode == "none":
        dP = np.zeros(M)
    else:
        dP = stream.dP
    return float(np.sum(s * (a * stream.U - b * dP)))


def leverage(stream, target, anchors, ks, weights, n0, mode="phys",
             readout="b1", broken=False):
    """L_direct and L_total for one record, U_r perturbed by +-1."""
    ub = ubar_dict(stream, anchors, ks, mode)
    tgt = int(target)
    vals = {}
    for tag, sgn in (("plus", 1.0), ("minus", -1.0)):
        U2 = stream.U.copy()
        U2[tgt] += sgn
        st2 = stream.with_U(U2)
        ub2 = ubar_dict(st2, anchors, ks, mode)
        if readout == "b1":
            vtot = v_b1(st2, anchors, ks, weights, n0, ub2, mode, broken=broken)
            vfroz = v_b1(st2, anchors, ks, weights, n0, ub, mode, broken=broken)
        else:
            vtot, _ = v_unp(st2, anchors, ks, weights, n0, ub2, mode)
            vfroz, _ = v_unp(st2, anchors, ks, weights, n0, ub, mode)
        vals[tag] = (vtot, vfroz)
    L_total = (vals["plus"][0] - vals["minus"][0]) / 2.0
    L_direct = (vals["plus"][1] - vals["minus"][1]) / 2.0
    return {"L_total": float(L_total), "L_direct": float(L_direct)}


def v_b1_windows(U, dP, windows):
    """Hand-oracle B1 on raw window tuples (lo, hi, w, ubar), [lo, hi)."""
    U = np.asarray(U, dtype=np.float64)
    dP = np.asarray(dP, dtype=np.float64)
    M = len(U)
    a = np.zeros(M)
    b = np.zeros(M)
    for lo, hi, w, ub in windows:
        a[lo:hi] += w
        b[lo:hi] += w * ub
    s = np.where(a > 0.0, np.minimum(1.0, 1.0 / np.maximum(a, 1e-300)), 0.0)
    return float(np.sum(s * (a * U - b * dP)))


def v_maxov_windows(U, dP, windows):
    """Max-overlap discount: each window's w is divided by max(1, max a in it)."""
    U = np.asarray(U, dtype=np.float64)
    dP = np.asarray(dP, dtype=np.float64)
    M = len(U)
    a = np.zeros(M)
    for lo, hi, w, _ub in windows:
        a[lo:hi] += w
    V = 0.0
    for lo, hi, w, ub in windows:
        if hi <= lo:
            continue
        den = max(1.0, float(a[lo:hi].max()))
        ww = w / den
        V += ww * float(np.sum(U[lo:hi] - ub * dP[lo:hi]))
    return float(V)
