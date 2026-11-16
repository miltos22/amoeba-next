"""late_need: the stage-0 late-life need statistic from a life record.

A life record is JSON lines: one ``header`` row, window rows, one ``life``
row.  Two writers exist and both are handled here, by header, with their
field names hard-coded from the writers (no field is guessed):

  forager / noise-floor   experiments/noise_floor/noise_floor.py
                          (forager lives use the same writer:
                          experiments/upper_reference/forager_life.py:106)
      header  args.life_s                       noise_floor.py:340 -> 102
      window  t_start, t_end                    noise_floor.py:174-175
              mean_abs_need                     noise_floor.py:85
              mean_need_pos, mean_need_neg      noise_floor.py:86-87
  Stream AC               baselines/life.py
      header  args.life_s                       baselines/life.py:631 -> 192
      window  t_start, t_end                    baselines/life.py:287-288
              mean_abs_need, mean_need_pos,
              mean_need_neg  from baselines.stats.aggregate, which is
              experiments.noise_floor._aggregate itself
              (baselines/stats.py:28), so the same three lines above.

``late_need`` is the time-weighted mean of the per-window mean |energy -
set point| over the window rows that lie wholly inside [0.7 L, L], L from
the header.  The hungry side (mean_need_pos) and the over-full side
(mean_need_neg) are computed over the same windows when the record has
them.

A record whose header is not one of these two raises; it is never parsed
on a guess.

    PYTHONPATH=. .venv/bin/python experiments/stage0_test/late_need.py FILE [...]
"""

from __future__ import annotations

import argparse
import json

LATE_START_FRACTION = 0.7
EPS = 1e-9

_FORAGER_ARGS = ("command", "preset", "n_patches", "behaviour", "seed", "life_s", "window_s")
_STREAM_AC_ARGS = ("command", "preset", "n_patches", "seed", "life_s", "window_s", "feeling", "reward")

_REQUIRED_WINDOW = ("t_start", "t_end", "mean_abs_need")


def read_record(path):
    """Return (header, window rows); raise if there is no header row."""
    header = None
    windows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kind = row.get("row")
            if kind == "header":
                if header is not None:
                    raise ValueError(f"{path}: more than one header row")
                header = row
            elif kind == "window":
                windows.append(row)
    if header is None:
        raise ValueError(f"{path}: no header row")
    if not windows:
        raise ValueError(f"{path}: no window rows")
    return header, windows


def record_kind(header):
    """Name the writer from the header, or raise.  Never guesses."""
    if header.get("row") != "header":
        raise ValueError("not a life-record header row")
    args = header.get("args")
    if not isinstance(args, dict):
        raise ValueError("life-record header has no args object")
    missing_forager = [name for name in _FORAGER_ARGS if name not in args]
    if not missing_forager:
        return "forager_or_noise_floor"
    missing_ac = [name for name in _STREAM_AC_ARGS if name not in args]
    if not missing_ac:
        return "stream_ac"
    raise ValueError(
        "unrecognised life-record header: "
        f"missing forager args {missing_forager}, missing Stream AC args {missing_ac}"
    )


def life_length(header, kind):
    """L in world seconds, from the header's args.life_s."""
    args = header["args"]
    if "life_s" not in args:
        raise ValueError(f"{kind} header has no args.life_s")
    length = float(args["life_s"])
    if not length > 0.0:
        raise ValueError(f"{kind} header has a non-positive life_s: {length!r}")
    return length


def _check_window(row, kind):
    for name in _REQUIRED_WINDOW:
        if name not in row:
            raise ValueError(f"{kind} window row lacks field {name!r}: keys={sorted(row)}")
    t_start = float(row["t_start"])
    t_end = float(row["t_end"])
    if not t_end > t_start:
        raise ValueError(f"{kind} window row has t_end <= t_start: {t_start!r} {t_end!r}")
    return t_start, t_end


def _time_weighted(selected):
    total = 0.0
    weight = 0.0
    for row, width in selected:
        total += width * float(row["mean_abs_need"])
        weight += width
    if weight <= 0.0:
        raise ValueError("no late window carries time")
    return total / weight


def _time_weighted_optional(selected, field):
    if any(field not in row for row, _ in selected):
        return None
    total = 0.0
    weight = 0.0
    for row, width in selected:
        total += width * float(row[field])
        weight += width
    return total / weight


def late_need(path):
    """Return the late-need statistic of one life record.

    Keys: late_need, late_need_pos, late_need_neg, n_windows, t_first,
    t_last, life_s, record_kind.
    """
    header, windows = read_record(path)
    kind = record_kind(header)
    length = life_length(header, kind)
    low = LATE_START_FRACTION * length
    selected = []
    for row in windows:
        t_start, t_end = _check_window(row, kind)
        if t_start >= low - EPS and t_end <= length + EPS:
            selected.append((row, t_end - t_start))
    if not selected:
        raise ValueError(
            f"{path}: no window lies wholly inside "
            f"[{LATE_START_FRACTION!r} * {length!r}, {length!r}]"
        )
    return {
        "late_need": _time_weighted(selected),
        "late_need_pos": _time_weighted_optional(selected, "mean_need_pos"),
        "late_need_neg": _time_weighted_optional(selected, "mean_need_neg"),
        "n_windows": len(selected),
        "t_first": float(selected[0][0]["t_start"]),
        "t_last": float(selected[-1][0]["t_end"]),
        "life_s": length,
        "record_kind": kind,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="life-record JSON-lines files")
    args = parser.parse_args(argv)
    for path in args.files:
        result = late_need(path)
        print(
            f"{path} record_kind={result['record_kind']} life_s={result['life_s']!r} "
            f"late_need={result['late_need']!r} "
            f"late_need_pos={result['late_need_pos']!r} "
            f"late_need_neg={result['late_need_neg']!r} "
            f"n_windows={result['n_windows']} "
            f"t_first={result['t_first']!r} t_last={result['t_last']!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
