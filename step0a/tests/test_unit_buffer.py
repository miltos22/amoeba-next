"""Tests for the unit buffer (revision 4, item 8)."""

from __future__ import annotations

import numpy as np

from step0a.experience_time import organ_stream
from step0a.pacemaker import Pacemaker
from step0a.unit_buffer import (
    band_Y, band_prefix, build_look_records, build_records, build_streams,
    p_raw_stream, read_counts, records_hash, records_read_at,
)
from step0a.world_signals import (
    H_W, UtilitySource, irregular_look_indices, look_indices, n_grid,
)


def _case(life=60.0):
    n = n_grid(life)
    pm = Pacemaker(noisy=False, seed=0)
    organ = organ_stream(pm, [], n)
    u = UtilitySource([(20.0, 20.2, -5.0), (40.0, 40.5, 2.0)]).grid(n)
    cp, cm = UtilitySource.cumulative_integrals(u)
    streams, basal = build_streams(pm, [], n)
    p = p_raw_stream(pm, n)
    rec = build_records(organ, cp, cm, p_raw=p, h=H_W,
                        streams=streams, basal_count=basal, snapshots=True)
    return n, organ, u, cp, cm, rec


def _prefix_dict(rec, count):
    return {
        "index": rec["index"][:count],
        "I_plus": rec["I_plus"][:count],
        "I_minus": rec["I_minus"][:count],
        "P_raw": rec["P_raw"][:count],
        "n_sens": 0,
        "snap": None,
    }


def test_records_identical_across_looks():
    n, organ, u, cp, cm, rec = _case()
    schedules = [look_indices(s, n) for s in ("2", "15", "50")]
    schedules.append(irregular_look_indices(n, seed=5))
    counts = [int(records_read_at(rec, idx)[-1]) for idx in schedules]
    common = min(counts)
    hashes = set()
    for j, idx in enumerate(schedules):
        rd = read_counts(rec, idx)
        assert rd.sum() == counts[j]
        hashes.add(records_hash(_prefix_dict(rec, common)))
    assert len(hashes) == 1
    # the full memory is look-independent
    assert len({records_hash(rec) for _ in schedules}) == 1


def test_band_Y_equals_direct_integral():
    n, organ, u, cp, cm, rec = _case()
    prefix = band_prefix(rec)
    R = len(rec["index"])
    for k in range(1, 7):
        q = 2 ** k
        half = 2 ** (k - 1)
        Y = band_Y(prefix, R, k)
        p = np.arange(len(Y), dtype=np.int64)
        first_prev = rec["prev_grid"][p + 1 + half]
        last_grid = rec["grid"][p + q]
        direct = ((cp[last_grid] - cp[first_prev])
                  - (cm[last_grid] - cm[first_prev]))
        err = np.abs(Y - direct)
        scale = np.maximum(np.abs(direct), 1.0)
        assert np.max(err / scale) <= 1e-9


def test_broken_look_buffer_fails():
    n, organ, u, cp, cm, rec = _case()
    unit_hash = records_hash(rec)
    idx2 = look_indices("2", n)
    idx50 = look_indices("50", n)
    ip2, im2 = UtilitySource.interval_integrals(cp, cm, idx2)
    ip50, im50 = UtilitySource.interval_integrals(cp, cm, idx50)
    br2 = build_look_records(organ, idx2, ip2, im2)
    br50 = build_look_records(organ, idx50, ip50, im50)
    assert len(br2["index"]) != len(br50["index"])
    assert records_hash(br2) != records_hash(br50)
    assert records_hash(br2) != unit_hash
