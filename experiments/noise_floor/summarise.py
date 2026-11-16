"""Summarise noise-floor JSON lines: one line per grid cell, ACROSS lives.

usage: summarise.py <outdir>

Reads every *.jsonl in <outdir>, prints a table and writes summary.csv and
summary.txt.  Files without a header row or without a life row are listed
as incomplete and left out.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

GROUP_FIELDS = ("preset", "n_patches", "behaviour", "redraw_s")


def _sd(values):
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _load(path):
    header = None
    life = None
    windows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kind = row.get("row")
            if kind == "header":
                header = row
            elif kind == "life":
                life = row
            elif kind == "window":
                windows.append(row)
    return header, windows, life


def main(argv) -> int:
    if len(argv) != 2:
        print("usage: summarise.py <outdir>", file=sys.stderr)
        return 2
    outdir = argv[1]
    paths = sorted(glob.glob(os.path.join(outdir, "*.jsonl")))
    if not paths:
        print(f"no *.jsonl in {outdir}", file=sys.stderr)
        return 1

    groups: dict[tuple, list[dict]] = {}
    incomplete: list[str] = []
    for path in paths:
        try:
            header, windows, life = _load(path)
        except (OSError, json.JSONDecodeError) as exc:
            incomplete.append(f"{os.path.basename(path)} ({exc})")
            continue
        if header is None or life is None:
            incomplete.append(os.path.basename(path))
            continue
        args = header["args"]
        key = (
            args["preset"],
            int(args["n_patches"]),
            args["behaviour"],
            float(args["redraw_s"]),
        )
        groups.setdefault(key, []).append({"header": header, "windows": windows, "life": life})

    if incomplete:
        print("incomplete files (not used):")
        for name in incomplete:
            print(f"  {name}")

    rows = []
    for key in sorted(groups):
        lives = groups[key]
        n = len(lives)
        life_rows = [item["life"] for item in lives]
        abs_need = [row["mean_abs_need"] for row in life_rows]
        energy = [row["mean_energy"] for row in life_rows]
        food = [row["food_eaten"] for row in life_rows]
        path = [row["path_m"] for row in life_rows]

        w1 = []
        within_sd = []
        any_zero = 0
        for item in lives:
            windows = item["windows"]
            later = [w["mean_abs_need"] for w in windows if w["window"] >= 1]
            if later:
                w1.append(sum(later) / len(later))
            if len(windows) >= 2:
                within_sd.append(_sd([w["mean_abs_need"] for w in windows]))
            zeros = [w["frac_energy_zero"] for w in windows]
            if not zeros:
                zeros = [item["life"]["frac_energy_zero"]]
            if any(z > 0.0 for z in zeros):
                any_zero += 1

        rows.append(
            {
                "preset": key[0],
                "n_patches": key[1],
                "behaviour": key[2],
                "redraw_s": key[3],
                "n_lives": n,
                "mean_abs_need_mean": _mean(abs_need),
                "mean_abs_need_sd": _sd(abs_need),
                "mean_energy_mean": _mean(energy),
                "mean_energy_sd": _sd(energy),
                "food_eaten_mean": _mean(food),
                "food_eaten_sd": _sd(food),
                "path_m_mean": _mean(path),
                "path_m_sd": _sd(path),
                "mean_abs_need_w1_mean": _mean(w1),
                "mean_abs_need_w1_sd": _sd(w1),
                "within_life_window_sd_mean": _mean([s for s in within_sd if not math.isnan(s)]),
                "min_energy": min(row["min_energy"] for row in life_rows),
                "lives_with_any_zero": any_zero,
            }
        )

    columns = list(rows[0].keys())
    csv_path = os.path.join(outdir, "summary.csv")
    txt_path = os.path.join(outdir, "summary.txt")

    def fmt(value):
        return repr(value) if isinstance(value, float) else value

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([fmt(row[column]) for column in columns])

    def short(value):
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = []
    for row in rows:
        lines.append("  ".join(f"{column}={short(row[column])}" for column in columns))
    text = "\n".join(lines) + "\n"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text, end="")
    print(f"wrote {csv_path}")
    print(f"wrote {txt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
