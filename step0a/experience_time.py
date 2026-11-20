"""Experience time (sections 1.4, 1.5 and 1.8 of the design).

Operator 0.1, CHORD mode (the default candidate)
------------------------------------------------
Groups: the basal group p = all 2 K_p pacemaker coordinates; each sensory
group g = a list of receptor coordinates.  With fixed reference constants

    z_g  = (x_g - r_g) / max(sigma_hat_g, sigma_min_g)
    d^2  = (omega_p / Z_p) ||z_p||^2 + (1 / Z_ref) sum_g omega_g ||z_g||^2

and at every look, if d >= delta:  m = floor(d/delta), Phi += m,
r <- r + (m delta / d)(x - r) jointly over all coordinates.

ARC mode (Pivot 0.1, concrete form)
-------------------------------------------
Basal accumulator B receives the unwrapped phase-arc length of the observed
pacemaker increments; the sensory distance uses its own remainder r_sens:

    B += sqrt(omega_p / Z_p) / sigma_hat_p * ||dtheta||
    d  = B + d_sens,   m = floor(d/delta), Phi += m,
    B *= (1 - m delta / d),  r_sens <- r_sens + (m delta / d)(x_sens - r_sens).

Broken variants
---------------
``grow_Z``  : the old dilution bug -- the whole d^2 is divided by the number
              of attached sensory groups (so silent groups dilute the basal
              chord and the non-freezing guarantee is lost).
``per_look``: Phi += 1 at every look (time tied to the thinking rate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .pacemaker import K_P
from .world_signals import H_W

SIGMA_HAT_P = 0.70710678
DELTA = 0.25
Z_P = 1.0
OMEGA_P = 1.0
Z_REF = 4.0
OMEGA_G = 1.0
SIGMA_MIN = 1e-3


# --------------------------------------------------------------------------
# groups
# --------------------------------------------------------------------------

@dataclass
class SensoryGroup:
    """A sensory group.

    xs:        array (n_coords, n_looks), receptor state sampled at looks.
    sigma_hat: array (n_coords,) frozen calibration MAD, in channel units.
    sigma_min: array (n_coords,) floor, in channel units.
    omega:     group weight.
    copies_of: if set, this group is a declared exact copy of the named group
               and is dropped from d (it shares the original group).
    """

    name: str
    xs: np.ndarray
    sigma_hat: np.ndarray
    sigma_min: np.ndarray
    omega: float = OMEGA_G
    copies_of: str | None = None

    def __post_init__(self):
        self.xs = np.asarray(self.xs, dtype=np.float64)
        self.sigma_hat = np.atleast_1d(np.asarray(self.sigma_hat, dtype=np.float64))
        self.sigma_min = np.atleast_1d(np.asarray(self.sigma_min, dtype=np.float64))
        if self.xs.ndim != 2:
            raise ValueError("xs must be (n_coords, n_looks)")
        if len(self.sigma_hat) != self.xs.shape[0]:
            raise ValueError("sigma_hat length must match coords")
        if len(self.sigma_min) != self.xs.shape[0]:
            raise ValueError("sigma_min length must match coords")


def basal_coords(pacemaker, idx: np.ndarray, n: int,
                 h: float = H_W):
    """The 2 K_p pacemaker coordinates at the looks, and their scales.

    Coordinates are ordered (sin theta_0, cos theta_0, sin theta_1, ...).
    """
    ph = pacemaker.phases(n, h)
    sin, cos = pacemaker.pairs_at(ph, idx)
    xs = np.empty((2 * K_P, len(idx)), dtype=np.float64)
    xs[0::2] = sin
    xs[1::2] = cos
    sig = np.full(2 * K_P, SIGMA_HAT_P, dtype=np.float64)
    smin = np.full(2 * K_P, SIGMA_MIN, dtype=np.float64)
    return xs, sig, smin


def active_groups(groups: list[SensoryGroup]) -> list[SensoryGroup]:
    """Drop declared copies; validate the declarations first."""
    names = {g.name for g in groups}
    active: list[SensoryGroup] = []
    for g in groups:
        if g.copies_of is not None:
            if g.copies_of not in names:
                raise ValueError(f"copy {g.name!r} of unknown group {g.copies_of!r}")
            if g.copies_of == g.name:
                raise ValueError("a group cannot copy itself")
            continue
        active.append(g)
    return active


# --------------------------------------------------------------------------
# non-freezing guarantee
# --------------------------------------------------------------------------

def non_freezing_lhs(delta: float = DELTA, z_p: float = Z_P,
                     omega_p: float = OMEGA_P,
                     sigma_p: float = SIGMA_HAT_P) -> float:
    return math.sqrt(omega_p / z_p) * (2.0 / sigma_p)


def assert_non_freezing(delta: float = DELTA, z_p: float = Z_P,
                        omega_p: float = OMEGA_P,
                        sigma_p: float = SIGMA_HAT_P) -> None:
    lhs = non_freezing_lhs(delta, z_p, omega_p, sigma_p)
    assert lhs >= 2.0 * delta, (
        f"non-freezing guarantee violated: LHS={lhs:.6f} < 2*delta={2.0 * delta:.6f}")
    return lhs


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

def _anchor_dict(ki, tt, mm, phis, ips, ims, dps, gi=None):
    return {
        "look_index": np.asarray(ki, dtype=np.int64),
        "grid_index": np.asarray(gi if gi is not None else ki, dtype=np.int64),
        "time": np.asarray(tt, dtype=np.float64),
        "m": np.asarray(mm, dtype=np.int64),
        "phi_after": np.asarray(phis, dtype=np.float64),
        "I_plus": np.asarray(ips, dtype=np.float64),
        "I_minus": np.asarray(ims, dtype=np.float64),
        "dP_raw": np.asarray(dps, dtype=np.float64),
    }


def simulate(pacemaker,
             groups: list[SensoryGroup],
             idx: np.ndarray,
             n: int,
             mode: str = "chord",
             delta: float = DELTA,
             z_p: float = Z_P,
             omega_p: float = OMEGA_P,
             z_ref: float = Z_REF,
             h: float = H_W,
             utility_ip: np.ndarray | None = None,
             utility_im: np.ndarray | None = None,
             dp_raw: np.ndarray | None = None,
             record_anchors: bool = False,
             store_trajectory: bool = False,
             check_non_freezing: bool = True,
             drop_zero_groups: bool = True):
    """Run one life and return Phi plus optional anchors.

    Returns a dict with at least: phi, n_looks, pending_I_plus,
    pending_I_minus, pending_dP_raw, total_I_plus, total_I_minus,
    total_dP_raw, and (record_anchors) anchors.
    """
    L = len(idx)
    if L == 0:
        raise ValueError("empty look schedule")
    if check_non_freezing and mode in ("chord", "grow_Z"):
        assert_non_freezing(delta, z_p, omega_p, SIGMA_HAT_P)
    if mode == "per_look":
        anchor = _anchor_dict([], [], [], [], [], [], [])
        return {
            "phi": float(L), "n_looks": L, "mode": mode,
            "anchors": anchor,
            "pending_I_plus": 0.0, "pending_I_minus": 0.0, "pending_dP_raw": 0.0,
            "total_I_plus": float(np.sum(utility_ip)) if utility_ip is not None else 0.0,
            "total_I_minus": float(np.sum(utility_im)) if utility_im is not None else 0.0,
            "total_dP_raw": float(np.sum(dp_raw)) if dp_raw is not None else 0.0,
        }

    act = active_groups(groups)
    n_sens_groups = len(act)

    # ---------------- coordinate arrays and factors
    basal_xs, basal_sig, basal_smin = basal_coords(pacemaker, idx, n, h)
    xs_list: list[np.ndarray] = [basal_xs[c] for c in range(basal_xs.shape[0])]
    fac_list: list[float] = [
        (omega_p / z_p) / (max(basal_sig[c], basal_smin[c]) ** 2)
        for c in range(basal_xs.shape[0])
    ]
    sens_xs: list[np.ndarray] = []
    for g in act:
        if drop_zero_groups and not np.any(g.xs):
            continue
        for c in range(g.xs.shape[0]):
            xs_list.append(g.xs[c])
            sens_xs.append(g.xs[c])
            sig = max(float(g.sigma_hat[c]), float(g.sigma_min[c]))
            fac_list.append((g.omega / z_ref) / (sig * sig))
    nc = len(xs_list)

    # ARC precomputations
    if mode == "arc":
        ph = pacemaker.phases(n, h)
        dth = pacemaker.dtheta(ph, idx)
        pacemaker.assert_positive(dth)
        basal_scale = math.sqrt(omega_p / z_p) / SIGMA_HAT_P
        sens_fac = [
            (g.omega / z_ref) / (max(float(g.sigma_hat[c]), float(g.sigma_min[c])) ** 2)
            for g in act if not (drop_zero_groups and not np.any(g.xs))
            for c in range(g.xs.shape[0])
        ]

    # grow_Z: divide the whole chord metric by the number of attached groups
    if mode == "grow_Z":
        ng = max(n_sens_groups, 1)
        fac_list = [f / ng for f in fac_list]

    # ---------------- state
    r = [float(x[0]) for x in xs_list]
    r_sens = [float(x[0]) for x in sens_xs]
    phi = 0
    B = 0.0

    if record_anchors:
        a_ki, a_gi, a_t, a_m, a_phi = [], [], [], [], []
        a_ip, a_im, a_dp = [], [], []
        pend_ip = pend_im = pend_dp = 0.0

    d_traj = np.empty(L, dtype=np.float64) if store_trajectory else None
    phi_traj = np.empty(L, dtype=np.float64) if store_trajectory else None
    m_traj = np.empty(L, dtype=np.int64) if store_trajectory else None

    # local bindings.  Python-float lists avoid boxing a numpy scalar on
    # every coordinate access; the values are the same float64 numbers.
    _sqrt = math.sqrt
    _floor = math.floor
    if L * nc <= 100_000_000:
        X = [x.tolist() for x in xs_list]
        if mode == "arc":
            sens_xs = [x.tolist() for x in sens_xs]
    else:
        X = xs_list
    FAC = fac_list
    R = r

    for k in range(L):
        m = 0
        if k == 0:
            d = 0.0
        else:
            if mode == "chord" or mode == "grow_Z":
                d2 = 0.0
                for c in range(nc):
                    diff = X[c][k] - R[c]
                    d2 += FAC[c] * diff * diff
                d = _sqrt(d2)
                if d >= delta:
                    m = int(_floor(d / delta))
                    phi += m
                    f = m * delta / d
                    for c in range(nc):
                        R[c] += f * (X[c][k] - R[c])
            elif mode == "arc":
                d2b = 0.0
                for j in range(dth.shape[0]):
                    d2b += float(dth[j, k]) ** 2
                B += basal_scale * _sqrt(d2b)
                ds2 = 0.0
                for c in range(len(sens_fac)):
                    diff = sens_xs[c][k] - r_sens[c]
                    ds2 += sens_fac[c] * diff * diff
                ds = _sqrt(ds2)
                if B + ds >= delta:
                    d = B + ds
                    m = int(_floor(d / delta))
                    phi += m
                    f = m * delta / d
                    B *= (1.0 - f)
                    for c in range(len(sens_fac)):
                        r_sens[c] += f * (sens_xs[c][k] - r_sens[c])
                else:
                    d = B + ds
            else:
                raise ValueError(f"unknown mode {mode!r}")

        if store_trajectory:
            d_traj[k] = d
            phi_traj[k] = phi
            m_traj[k] = m

        if record_anchors:
            ip_k = float(utility_ip[k]) if utility_ip is not None else 0.0
            im_k = float(utility_im[k]) if utility_im is not None else 0.0
            dp_k = float(dp_raw[k]) if dp_raw is not None else 0.0
            if m >= 1:
                a_ki.append(k)
                a_gi.append(int(idx[k]))
                a_t.append(float(idx[k]) * h)
                a_m.append(m)
                a_phi.append(phi)
                a_ip.append(pend_ip + ip_k)
                a_im.append(pend_im + im_k)
                a_dp.append(pend_dp + dp_k)
                pend_ip = pend_im = pend_dp = 0.0
            else:
                pend_ip += ip_k
                pend_im += im_k
                pend_dp += dp_k

    out = {
        "phi": float(phi),
        "n_looks": L,
        "mode": mode,
        "d_traj": d_traj,
        "phi_traj": phi_traj,
        "m_traj": m_traj,
    }
    if record_anchors:
        out["anchors"] = _anchor_dict(a_ki, a_t, a_m, a_phi, a_ip, a_im, a_dp,
                                      gi=a_gi)
        out["pending_I_plus"] = pend_ip
        out["pending_I_minus"] = pend_im
        out["pending_dP_raw"] = pend_dp
        out["total_I_plus"] = float(np.sum(utility_ip)) if utility_ip is not None else 0.0
        out["total_I_minus"] = float(np.sum(utility_im)) if utility_im is not None else 0.0
        out["total_dP_raw"] = float(np.sum(dp_raw)) if dp_raw is not None else 0.0
    return out




# ---------------------------------------------------------------------------
# RECEPTOR mode (revision 2)
# ---------------------------------------------------------------------------
#
# The CHORD rule runs inside the sense organ, in physical time, on the world
# grid h_w.  At each look the brain reads only the accumulated count m since
# the previous look (plus the coalesced integrals).  No look rate enters.

@dataclass
class GridGroup:
    """A sensory group whose receptor stream is available at every grid point.

    xs may be None for a group that is exactly zero (a silent source).
    """

    name: str
    xs: np.ndarray | None
    sigma_hat: np.ndarray
    sigma_min: np.ndarray
    omega: float = OMEGA_G
    copies_of: str | None = None

    def __post_init__(self):
        self.sigma_hat = np.atleast_1d(np.asarray(self.sigma_hat, dtype=np.float64))
        self.sigma_min = np.atleast_1d(np.asarray(self.sigma_min, dtype=np.float64))
        if self.xs is not None:
            self.xs = np.asarray(self.xs, dtype=np.float64)
            if self.xs.ndim != 2:
                raise ValueError("xs must be (n_coords, n_grid)")
            if len(self.sigma_hat) != self.xs.shape[0]:
                raise ValueError("sigma_hat length must match coords")
            if len(self.sigma_min) != self.xs.shape[0]:
                raise ValueError("sigma_min length must match coords")


def _basal_stream(pacemaker, n: int, h: float, z_p: float, omega_p: float):
    xs: list[np.ndarray] = []
    fac: list[float] = []
    f = (omega_p / z_p) / (SIGMA_HAT_P ** 2)
    for j in range(K_P):
        sin, cos = pacemaker.pair_stream(j, n, h)
        xs.append(sin)
        xs.append(cos)
        fac.append(f)
        fac.append(f)
    return xs, fac


def organ_stream(pacemaker,
                 groups: list[GridGroup],
                 n: int,
                 h: float = H_W,
                 delta: float = DELTA,
                 z_p: float = Z_P,
                 omega_p: float = OMEGA_P,
                 z_ref: float = Z_REF,
                 drop_zero_groups: bool = True,
                 list_threshold: int = 60_000_000):
    """Run the CHORD rule once per grid step and return the m counter.

    The returned ``m_grid[k]`` is the number of delta-crossings realized on
    the grid interval ending at grid point k (m_grid[0] = 0).  Cumulative sums
    of m_grid are exactly what a look reads.
    """
    if n <= 0:
        raise ValueError("empty grid")
    assert_non_freezing(delta, z_p, omega_p)
    act = active_groups(groups)
    xs, fac = _basal_stream(pacemaker, n, h, z_p, omega_p)
    for g in act:
        if g.xs is None:
            continue
        if drop_zero_groups and not np.any(g.xs):
            continue
        for c in range(g.xs.shape[0]):
            xs.append(g.xs[c])
            sig = max(float(g.sigma_hat[c]), float(g.sigma_min[c]))
            fac.append((g.omega / z_ref) / (sig * sig))
    nc = len(xs)
    m_grid = np.empty(n, dtype=np.int32)
    m_grid[0] = 0
    R = [float(x[0]) for x in xs]
    X = [x.tolist() for x in xs] if n * nc <= list_threshold else xs
    _sqrt = math.sqrt
    _floor = math.floor
    FAC = fac
    for k in range(1, n):
        d2 = 0.0
        for c in range(nc):
            diff = X[c][k] - R[c]
            d2 += FAC[c] * diff * diff
        d = _sqrt(d2)
        if d >= delta:
            m = int(_floor(d / delta))
            f = m * delta / d
            for c in range(nc):
                R[c] += f * (X[c][k] - R[c])
        else:
            m = 0
        m_grid[k] = m
    phi = int(m_grid.sum(dtype=np.int64))
    return {"m_grid": m_grid, "phi": phi, "nc": nc, "n": n, "h": h}


def read_organ(organ: dict, idx: np.ndarray, h: float | None = None,
               utility_ip: np.ndarray | None = None,
               utility_im: np.ndarray | None = None,
               dp_raw: np.ndarray | None = None,
               record_anchors: bool = False):
    """Read an organ stream at look indices: total Phi and per-look m.

    Per look k the brain reads the accumulated count since the previous look,
    ``m_look[k] = cum[idx[k]] - cum[idx[k-1]]``, and Phi at the look is
    ``cum[idx[k]]``.  Coalescing of I_plus/I_minus/dP_raw follows 1.6.
    """
    h = organ["h"] if h is None else h
    cum = np.cumsum(organ["m_grid"], dtype=np.int64)
    phi_look = cum[idx]
    m_look = np.zeros(len(idx), dtype=np.int64)
    if len(idx) > 1:
        m_look[1:] = np.diff(phi_look)
    out = {"phi_look": phi_look, "m_look": m_look,
           "phi": float(phi_look[-1]), "n_looks": len(idx),
           "total_I_plus": float(np.sum(utility_ip)) if utility_ip is not None else 0.0,
           "total_I_minus": float(np.sum(utility_im)) if utility_im is not None else 0.0,
           "total_dP_raw": float(np.sum(dp_raw)) if dp_raw is not None else 0.0}
    if record_anchors:
        a_ki, a_gi, a_t, a_m, a_phi = [], [], [], [], []
        a_ip, a_im, a_dp = [], [], []
        pend_ip = pend_im = pend_dp = 0.0
        for k in range(len(idx)):
            ip_k = float(utility_ip[k]) if utility_ip is not None else 0.0
            im_k = float(utility_im[k]) if utility_im is not None else 0.0
            dp_k = float(dp_raw[k]) if dp_raw is not None else 0.0
            if m_look[k] >= 1:
                a_ki.append(k)
                a_gi.append(int(idx[k]))
                a_t.append(float(idx[k]) * h)
                a_m.append(int(m_look[k]))
                a_phi.append(float(phi_look[k]))
                a_ip.append(pend_ip + ip_k)
                a_im.append(pend_im + im_k)
                a_dp.append(pend_dp + dp_k)
                pend_ip = pend_im = pend_dp = 0.0
            else:
                pend_ip += ip_k
                pend_im += im_k
                pend_dp += dp_k
        out["anchors"] = _anchor_dict(a_ki, a_t, a_m, a_phi, a_ip, a_im, a_dp,
                                      gi=a_gi)
        out["pending_I_plus"] = pend_ip
        out["pending_I_minus"] = pend_im
        out["pending_dP_raw"] = pend_dp
    return out

def window_advances(m_traj, times, window: float) -> np.ndarray:
    """Sum of Phi advancement (``m``) inside each consecutive window of
    ``window`` world seconds."""
    times = np.asarray(times)
    m = np.asarray(m_traj, dtype=np.float64)
    bucket = (times // window).astype(np.int64)
    nb = int(times[-1] // window) + 1
    return np.bincount(bucket, weights=m, minlength=nb)
