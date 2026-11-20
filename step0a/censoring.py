"""Interval-censoring primitives for H6 (revision 3).

Atoms are the m >= 1 look atoms (coalesced): atom i has Phi endpoints
(start_i, end_i] and physical utility integrals (I_plus_i, I_minus_i).
A band of anchor j is the Phi interval (Phi_j + 2^(k-1), Phi_j + 2^k].
Only atom endpoints are exact; a band boundary that falls strictly inside an
atom makes the inside integral uncertain.  Lower bound: put all of the atom's
I_minus inside and none of I_plus; upper bound: the opposite.  Never
interpolate.
"""

from __future__ import annotations

import numpy as np

TAU_CENS = 0.5  # 0.5 * S with S = 1.0


# ---------------------------------------------------------------------------
# scalar band bounds
# ---------------------------------------------------------------------------

def band_bounds(a: float, b: float, ends: np.ndarray,
                ip: np.ndarray, im: np.ndarray,
                prefix_net: np.ndarray | None = None):
    """Bounds on the integral over the Phi interval (a, b].

    ``ends[i]`` is atom i's end Phi; atom i starts at ``ends[i-1]`` (or 0).
    Returns (y_min, y_max, overlap_first, overlap_last).
    """
    ends = np.asarray(ends)
    ip = np.asarray(ip)
    im = np.asarray(im)
    A = len(ends)
    if prefix_net is None:
        prefix_net = np.concatenate(([0.0], np.cumsum(ip - im)))
    s = int(np.searchsorted(ends, a, side="left"))
    e = int(np.searchsorted(ends, b, side="left"))
    if e >= A:
        e = A - 1
    if s >= A:
        return 0.0, 0.0, A, A - 1
    start_partial = bool(ends[s] > a)
    end_partial = bool(ends[e] > b)
    first_whole = s + 1
    last_whole = e - 1 if end_partial else e
    if first_whole <= last_whole:
        whole = float(prefix_net[last_whole + 1] - prefix_net[first_whole])
    else:
        whole = 0.0
    lo = whole
    hi = whole
    if start_partial:
        lo -= float(im[s])
        hi += float(ip[s])
    if end_partial and e != s:
        lo -= float(im[e])
        hi += float(ip[e])
    first_overlap = s if start_partial else s + 1
    last_overlap = e
    return lo, hi, first_overlap, last_overlap


# ---------------------------------------------------------------------------
# vectorised band bounds (for the H6(d) sweep)
# ---------------------------------------------------------------------------

def batch_band_bounds(a: np.ndarray, b: np.ndarray, ends: np.ndarray,
                      ip: np.ndarray, im: np.ndarray):
    """Vectorised version of :func:`band_bounds` for matured bands.

    a, b are arrays of equal length; every b must be <= ends[-1].
    Returns (y_min, y_max, first_overlap, last_overlap).
    """
    ends = np.asarray(ends)
    ip = np.asarray(ip)
    im = np.asarray(im)
    A = len(ends)
    net = ip - im
    P = np.concatenate(([0.0], np.cumsum(net)))
    s = np.searchsorted(ends, a, side="left")
    e = np.searchsorted(ends, b, side="left")
    e = np.minimum(e, A - 1)
    start_partial = ends[s] > a
    end_partial = ends[e] > b
    first_whole = s + 1
    last_whole = np.where(end_partial, e - 1, e)
    has_whole = first_whole <= last_whole
    whole = np.where(has_whole,
                     P[np.clip(last_whole + 1, 0, A)]
                     - P[np.clip(first_whole, 0, A)], 0.0)
    lo = whole.copy()
    hi = whole.copy()
    lo[start_partial] -= im[s[start_partial]]
    hi[start_partial] += ip[s[start_partial]]
    second = end_partial & (e != s)
    lo[second] -= im[e[second]]
    hi[second] += ip[e[second]]
    first_overlap = np.where(start_partial, s, s + 1)
    last_overlap = e
    return lo, hi, first_overlap, last_overlap


def band_width(ip: np.ndarray, im: np.ndarray,
               ends: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Censoring width (Y_max - Y_min) without the exact part."""
    lo, hi, _, _ = batch_band_bounds(a, b, ends, ip, im)
    return hi - lo


# ---------------------------------------------------------------------------
# atoms from a RECEPTOR readout
# ---------------------------------------------------------------------------

def atoms_from_readout(readout: dict, h: float):
    """Atom arrays from ``read_organ(..., record_anchors=True)``.

    Atom i spans (phi_after[i-1], phi_after[i]] and (t[i-1], t[i]], with the
    anchor's coalesced integrals.  Its block label is time // 1000 s mod 2.
    """
    an = readout["anchors"]
    phi_end = np.asarray(an["phi_after"], dtype=np.float64)
    t_end = np.asarray(an["time"], dtype=np.float64)
    ip = np.asarray(an["I_plus"], dtype=np.float64)
    im = np.asarray(an["I_minus"], dtype=np.float64)
    m = np.asarray(an["m"], dtype=np.int64)
    phi_start = np.concatenate(([0.0], phi_end[:-1]))
    t_start = np.concatenate(([0.0], t_end[:-1]))
    return {"phi_start": phi_start, "phi_end": phi_end,
            "t_start": t_start, "t_end": t_end,
            "I_plus": ip, "I_minus": im, "m": m}


def atom_impulse_flags(atoms: dict, events, h: float):
    """True for atoms whose world-time interval overlaps an event support."""
    t0 = atoms["t_start"]
    t1 = atoms["t_end"]
    flags = np.zeros(len(t0), dtype=bool)
    for e0, e1, _v in events:
        flags |= (t0 < e1) & (t1 > e0)
    return flags


def mark_bearing(flag_first: np.ndarray, flag_last: np.ndarray,
                 atom_impulse: np.ndarray):
    """Bands whose overlapping atom range contains any impulse atom."""
    P = np.concatenate(([0], np.cumsum(atom_impulse.astype(np.int64))))
    last = np.minimum(flag_last, len(atom_impulse) - 1)
    first = np.clip(flag_first, 0, len(atom_impulse))
    return (P[last + 1] - P[first]) > 0


# ---------------------------------------------------------------------------
# H6(b): coalescing readout (including a deliberately broken variant)
# ---------------------------------------------------------------------------

def read_atoms(organ: dict, idx: np.ndarray,
               utility_ip: np.ndarray | None = None,
               utility_im: np.ndarray | None = None,
               anchor_every_look: bool = False):
    """Read anchors from an organ stream.

    Normal: only looks with m >= 1 create anchors and the m = 0 integrals are
    coalesced into the next anchor.  Broken: an anchor is created at every
    look, each with that look's own integrals (no coalescing).
    """
    cum = np.cumsum(organ["m_grid"], dtype=np.int64)
    phi_look = cum[idx]
    m_look = np.zeros(len(idx), dtype=np.int64)
    if len(idx) > 1:
        m_look[1:] = np.diff(phi_look)
    ip = np.zeros(len(idx)) if utility_ip is None else np.asarray(utility_ip)
    im = np.zeros(len(idx)) if utility_im is None else np.asarray(utility_im)
    ki, gi, tt, mm, pp, ap, am = [], [], [], [], [], [], []
    pend_p = pend_m = 0.0
    zero_anchor = 0
    for k in range(len(idx)):
        if anchor_every_look:
            ki.append(k); gi.append(int(idx[k])); tt.append(float(idx[k]) * organ["h"])
            mm.append(int(m_look[k])); pp.append(float(phi_look[k]))
            ap.append(float(ip[k])); am.append(float(im[k]))
            if m_look[k] == 0:
                zero_anchor += 1
        elif m_look[k] >= 1:
            ki.append(k); gi.append(int(idx[k])); tt.append(float(idx[k]) * organ["h"])
            mm.append(int(m_look[k])); pp.append(float(phi_look[k]))
            ap.append(pend_p + float(ip[k])); am.append(pend_m + float(im[k]))
            pend_p = pend_m = 0.0
        else:
            pend_p += float(ip[k]); pend_m += float(im[k])
    return {
        "look_index": np.asarray(ki, dtype=np.int64),
        "grid_index": np.asarray(gi, dtype=np.int64),
        "time": np.asarray(tt, dtype=np.float64),
        "m": np.asarray(mm, dtype=np.int64),
        "phi_after": np.asarray(pp, dtype=np.float64),
        "I_plus": np.asarray(ap, dtype=np.float64),
        "I_minus": np.asarray(am, dtype=np.float64),
        "pending_I_plus": pend_p, "pending_I_minus": pend_m,
        "anchors_at_m0": zero_anchor,
        "n_m0_looks": int(np.sum(m_look == 0)),
    }


def censored(lo: np.ndarray, hi: np.ndarray, tau: float = TAU_CENS):
    return (hi - lo) > tau
