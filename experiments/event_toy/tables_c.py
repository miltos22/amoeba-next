"""Third-group tables from out_c/cells.jsonl. Numbers only."""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_c")
CELLS = os.path.join(OUT, "cells.jsonl")


def load():
    rows = []
    for line in open(CELLS):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def stream_name(r):
    return "real" if r["contingent"] else "broken"


def sort_key(r):
    return (r["T"], r["variant"], r["combo"], r["m"], r["beta"], r["rho"],
            r["sign"], 0 if r["contingent"] else 1)


def med(r):
    m = r["median_N"]
    return ("%d" % m) if float(m) == int(m) else ("%g" % m)


def main():
    rows = sorted(load(), key=sort_key)
    lines = ["# third-group tables (%d cells)" % len(rows), ""]

    lines.append("## Table 1: per-cell summary")
    lines.append("")
    lines.append("| combo | m | beta | rho | T | stream | sign | reached/16 | "
                 "median N* | high | low | stuck | median final p | "
                 "cap_frac_head (7, 2dp) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cap = " ".join("%.2f" % v for v in r["cap_frac_head"])
        lines.append("| %s | %g | %g | %g | %g | %s | %+d | %d | %s | %d | %d | "
                     "%d | %.4f | %s |"
                     % (r["combo"], r["m"], r["beta"], r["rho"], r["T"],
                        stream_name(r), r["sign"], r["n_reached"], med(r),
                        r["n_high"], r["n_low"], r["n_stuck"],
                        r["median_final_p"], cap))

    lines.append("")
    lines.append("## Table 2: per-head delta_end, mean and SD across the 16 seeds")
    lines.append("")
    lines.append("| combo | m | beta | rho | T | stream | sign | "
                 + " | ".join("head %d" % (i + 1) for i in range(7)) + " |")
    lines.append("|---|---|---|---|---|---|---|" + "---|" * 7)
    for r in rows:
        de = np.array(r["delta_end"], dtype=float)  # (7, ns)
        cells = ["%.4f+-%.4f" % (de[i].mean(), de[i].std(ddof=0))
                 for i in range(7)]
        lines.append("| %s | %g | %g | %g | %g | %s | %+d | %s |"
                     % (r["combo"], r["m"], r["beta"], r["rho"], r["T"],
                        stream_name(r), r["sign"], " | ".join(cells)))

    # Table 3: item-1 real cells, combo max
    sel = {}
    for r in rows:
        if (r["variant"] == "g1" and r["combo"] == "max" and r["contingent"]
                and r["m"] in (10.0, 30.0)):
            sel[(r["m"], r["rho"], r["T"])] = float(r["median_N"])
    lines.append("")
    lines.append("## Table 3: median N*, real stream, item 1, combo max")
    lines.append("")
    lines.append("| m | rho | T=3 | T=30 | T=300 |")
    lines.append("|---|---|---|---|---|")
    for m in (10.0, 30.0):
        for rho in (4.0, 0.25):
            cells = []
            for T in (3.0, 30.0, 300.0):
                v = sel.get((m, rho, T))
                cells.append("-" if v is None else
                             (("%d" % v) if v == int(v) else ("%g" % v)))
            lines.append("| %g | %g | %s |" % (m, rho, " | ".join(cells)))
    lines.append("")
    lines.append("| ratio median(rho 0.25) / median(rho 4) | T=3 | T=30 | T=300 |")
    lines.append("|---|---|---|---|")
    for m in (10.0, 30.0):
        cells = []
        for T in (3.0, 30.0, 300.0):
            a = sel.get((m, 4.0, T))
            b = sel.get((m, 0.25, T))
            cells.append("-" if (a is None or b is None) else "%.3f" % (b / a))
        lines.append("| m=%g | %s |" % (m, " | ".join(cells)))

    txt = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "tables.md"), "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
