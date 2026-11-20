"""Second-group tables from out_b/cells.jsonl. Numbers only."""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_b")
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
    return (r["T"], r["variant"], r["rho"], r["sign"], 0 if r["contingent"] else 1)


def main():
    rows = sorted(load(), key=sort_key)
    lines = ["# second-group tables (%d cells)" % len(rows), ""]
    lines.append("## Table 1: per-cell summary")
    lines.append("")
    lines.append("| variant | features | rho | T | stream | sign | reached/16 | median N* | "
                 "high | low | stuck | median final p | bound/head (7, 2dp) | mean rbar |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        bfh = " ".join("%.2f" % v for v in r["bound_frac_head"])
        lines.append("| %s | %s | %g | %g | %s | %+d | %d | %s | %d | %d | %d | %.4f | %s | %.5f |"
                     % (r["variant"], r["features"], r["rho"], r["T"],
                        stream_name(r), r["sign"], r["n_reached"],
                        ("%g" % r["median_N"]), r["n_high"], r["n_low"],
                        r["n_stuck"], r["median_final_p"], bfh,
                        float(np.mean(r["rbar_end"]))))

    lines.append("")
    lines.append("## Table 2: per-head head_contrib, mean and SD across 16 seeds")
    lines.append("")
    hdr = "| variant | features | rho | T | stream | sign | " + \
          " | ".join("head %d" % (i + 1) for i in range(7)) + " |"
    lines.append(hdr)
    lines.append("|---|---|---|---|---|---|" + "---|" * 7)
    for r in rows:
        hc = np.array(r["head_contrib"], dtype=float)  # (7, ns)
        cells = []
        for i in range(7):
            cells.append("%.4f+-%.4f" % (hc[i].mean(), hc[i].std(ddof=0)))
        lines.append("| %s | %s | %g | %g | %s | %+d | %s |"
                     % (r["variant"], r["features"], r["rho"], r["T"],
                        stream_name(r), r["sign"], " | ".join(cells)))

    txt = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "tables.md"), "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
