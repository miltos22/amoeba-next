"""Unit buffer inside the sense organ (revision 4).

The organ counts Phi exactly as RECEPTOR mode does on h_w.  Whenever a grid
step raises the integer Phi count by c >= 1, c unit records close.  The
utility integrals accumulated since the previous close (this grid step
included) go to the first of the c records; the other c-1 get zero.  Every
record carries its integer Phi index, I_u+, I_u-, a float32 snapshot of the
sensory receptor states at the end of that grid step, and P_raw at that same
grid step.  Integrals after the last crossing stay in an open record.

Look times never enter any function in this module.
"""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

from .experience_time import (
    SIGMA_HAT_P, SIGMA_MIN, OMEGA_G, active_groups,
)
from .pacemaker import K_P, Pacemaker
from .receptors import exact_filter
from .world_signals import H_W, grid_times


# ---------------------------------------------------------------------------
# streams and P_raw
# ---------------------------------------------------------------------------

def build_streams(pacemaker, groups, n: int, h: float = H_W,
                  drop_zero_groups: bool = True):
    """The organ's coordinate streams, same order as organ_stream.

    Returns (xs, basal_count).  xs[basal_count:] are the sensory receptor
    states, which are what the record snapshots store.
    """
    act = active_groups(groups)
    xs = []
    for j in range(K_P):
        sin, cos = pacemaker.pair_stream(j, n, h)
        xs.append(sin)
        xs.append(cos)
    basal = len(xs)
    for g in act:
        if g.xs is None:
            continue
        if drop_zero_groups and not np.any(g.xs):
            continue
        for c in range(g.xs.shape[0]):
            xs.append(g.xs[c])
    return xs, basal


def p_raw_stream(pacemaker, n: int, h: float = H_W):
    """P_raw after every grid step (cycles); P_raw[0] = 0."""
    ph = pacemaker.phases(n, h)
    idx = np.arange(n, dtype=np.int64)
    d = Pacemaker.dtheta(ph, idx)
    return np.cumsum(d.sum(axis=0) / (2.0 * math.pi))


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def build_records(organ: dict, utility_cp, utility_cm,
                  p_raw=None, h: float = H_W,
                  streams=None, basal_count: int = 0,
                  snapshots: bool = False):
    """Close one record per unit Phi crossing.

    utility_cp/utility_cm are cumulative integrals at grid points.  streams
    and basal_count are required only for snapshots.  Returns a dict of
    arrays; snapshots (if any) are float32 rows of the sensory states.
    """
    m = organ["m_grid"]
    n = len(m)
    R = int(organ["phi"])
    n_sens = 0 if streams is None else len(streams) - basal_count
    rec = {
        "index": np.empty(R, dtype=np.int64),
        "I_plus": np.empty(R, dtype=np.float64),
        "I_minus": np.empty(R, dtype=np.float64),
        "P_raw": np.zeros(R, dtype=np.float64) if p_raw is None
                 else np.empty(R, dtype=np.float64),
        "grid": np.empty(R, dtype=np.int64),
        "prev_grid": np.empty(R, dtype=np.int64),
        "n_sens": n_sens,
    }
    snap = np.empty((R, n_sens), dtype=np.float32) if snapshots else None
    cum = np.cumsum(m, dtype=np.int64)
    r = 0
    prev_close = 0
    for k in np.nonzero(m)[0]:
        k = int(k)
        c = int(m[k])
        phi_end = int(cum[k])
        phi_start = phi_end - c
        i0 = float(utility_cp[k] - utility_cp[prev_close])
        i1 = float(utility_cm[k] - utility_cm[prev_close])
        sl = slice(r, r + c)
        rec["index"][sl] = np.arange(phi_start + 1, phi_end + 1, dtype=np.int64)
        rec["I_plus"][sl] = 0.0
        rec["I_minus"][sl] = 0.0
        rec["I_plus"][r] = i0
        rec["I_minus"][r] = i1
        if p_raw is not None:
            rec["P_raw"][sl] = p_raw[k]
        rec["grid"][sl] = k
        rec["prev_grid"][sl] = k
        rec["prev_grid"][r] = prev_close
        if snapshots:
            row = np.array([x[k] for x in streams[basal_count:]],
                           dtype=np.float32)
            snap[sl] = row
        r += c
        prev_close = k
    if r != R:
        raise RuntimeError(f"closed {r} records, organ phi {R}")
    rec["snap"] = snap
    return rec


def build_record_integrals(organ: dict, utility_cp, utility_cm):
    """Light record build: index and integrals only (no snapshots/times)."""
    rec = build_records(organ, utility_cp, utility_cm, p_raw=None,
                        streams=None, snapshots=False)
    return rec


# ---------------------------------------------------------------------------
# reads, hashes, bands
# ---------------------------------------------------------------------------

def read_counts(rec: dict, idx: np.ndarray):
    """Number of records read at each look (closed since the previous look)."""
    g = rec["grid"]
    pos = np.searchsorted(g, idx, side="right")
    prev = np.concatenate(([0], pos[:-1]))
    return pos - prev


def records_read_at(rec: dict, idx: np.ndarray):
    """Highest record position read by each look (exclusive count)."""
    return np.searchsorted(rec["grid"], idx, side="right")


def records_hash(rec: dict, chunk: int = 200_000) -> str:
    """sha256 over (index, I_plus, I_minus, snapshot bytes, P_raw) per record."""
    h = hashlib.sha256()
    n_sens = rec["n_sens"]
    dt = np.dtype([("index", "<i8"), ("ip", "<f8"), ("im", "<f8"),
                   ("snap", "<f4", (n_sens,)), ("p", "<f8")])
    R = len(rec["index"])
    for start in range(0, R, chunk):
        end = min(R, start + chunk)
        a = np.empty(end - start, dtype=dt)
        a["index"] = rec["index"][start:end]
        a["ip"] = rec["I_plus"][start:end]
        a["im"] = rec["I_minus"][start:end]
        if n_sens:
            a["snap"] = rec["snap"][start:end]
        a["p"] = rec["P_raw"][start:end]
        h.update(a.tobytes())
    return h.hexdigest()


def band_prefix(rec: dict):
    net = rec["I_plus"] - rec["I_minus"]
    return np.concatenate(([0.0], np.cumsum(net)))


def band_Y(prefix: np.ndarray, R: int, k: int):
    """Y_jk for every record j whose band matures (positions 0..R-2^k-1).

    Bands are exact at unit resolution: (j + 2^(k-1), j + 2^k] in Phi.
    """
    q = 2 ** k
    half = 2 ** (k - 1)
    n_valid = R - q
    if n_valid <= 0:
        return np.empty(0)
    p = np.arange(n_valid, dtype=np.int64)
    return prefix[p + 1 + q] - prefix[p + 1 + half]


def record_impulse_flags(rec: dict, events, h: float = H_W):
    """True where the record's world-time interval overlaps an event."""
    R = len(rec["grid"])
    t0 = rec["prev_grid"].astype(np.float64) * h
    t1 = rec["grid"].astype(np.float64) * h
    valid = rec["grid"] > rec["prev_grid"]
    flag = np.zeros(R, dtype=bool)
    for e0, e1, _v in events:
        flag |= valid & (t0 < e1) & (t1 > e0)
    return flag


def band_bearing(flag: np.ndarray, R: int, k: int):
    """For every matured record j, whether any record in its band is flagged."""
    q = 2 ** k
    half = 2 ** (k - 1)
    P = np.concatenate(([0], np.cumsum(flag.astype(np.int64))))
    n_valid = R - q
    if n_valid <= 0:
        return np.zeros(0, dtype=bool)
    p = np.arange(n_valid, dtype=np.int64)
    return (P[p + 1 + q] - P[p + 1 + half]) > 0


def record_trace(rec: dict, lambdas):
    """Per-record extensive trace, in record index order."""
    dU = rec["I_plus"] - rec["I_minus"]
    R = len(dU)
    out = np.empty((R, len(lambdas)), dtype=np.float64)
    padded = np.concatenate((dU, [0.0]))
    for a, lam in enumerate(lambdas):
        e = math.exp(-1.0 / lam)
        c = math.exp(-1.0 / (2.0 * lam))
        y = exact_filter(padded, e, c, x0=0.0)
        out[:, a] = y[1:R + 1]
    return out


def trace_at(rec: dict, T: np.ndarray, target_time: float, h: float = H_W):
    """T_U after the last record whose grid time is <= target_time."""
    g = rec["grid"].astype(np.float64) * h
    pos = int(np.searchsorted(g, target_time, side="right")) - 1
    if pos < 0:
        return T[0]
    return T[pos]


# ---------------------------------------------------------------------------
# broken variants used only by tests / controls
# ---------------------------------------------------------------------------

def build_look_records(organ: dict, idx: np.ndarray, utility_ip, utility_im,
                       h: float = H_W):
    """BROKEN: one record per look, with that look's own integrals."""
    cum = np.cumsum(organ["m_grid"], dtype=np.int64)
    R = len(idx)
    return {
        "index": cum[idx].astype(np.int64),
        "I_plus": np.asarray(utility_ip, dtype=np.float64),
        "I_minus": np.asarray(utility_im, dtype=np.float64),
        "P_raw": np.zeros(R),
        "grid": np.asarray(idx, dtype=np.int64),
        "prev_grid": np.concatenate(([0], np.asarray(idx[:-1], dtype=np.int64))),
        "n_sens": 0,
        "snap": None,
    }
