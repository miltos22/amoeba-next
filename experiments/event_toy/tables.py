"""Build the W23 tables from out/cells.jsonl (never typed by hand)."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "cells.jsonl")
D = os.path.abspath(os.path.join(HERE, "..", "..", "design_notes"))

LEARNERS = [("L1", 1.0, "L1"), ("L2", 0.043, "L2 kappa=0.043"),
            ("L2", 1.0, "L2 kappa=1.0"), ("L3", 1.0, "L3")]
RHOS = [4.0, 1.0, 0.25]
TS = [3.0, 30.0, 300.0]


def load():
    rows = []
    for line in open(OUT):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def key(p):
    return (p["learner"], (p.get("kappa") if p["learner"] == "L2" else None),
            p["features"], p["rho"], p["T"], p["sign"], bool(p["background"]),
            bool(p["contingent"]), p["dt"], p["alpha_p"])


def cellstr(r):
    if r is None:
        return "-"
    m = r["median_N"]
    ms = ("%d" % m) if float(m) == int(m) else ("%g" % m)
    return "%s (%d/16)" % (ms, r["n_reached"])


def table(rows, pick, learner, kappa, feats, rhos, ts):
    idx = {}
    for r in rows:
        k = key(r)
        if (k[0], k[1], k[2]) != (learner, None if learner != "L2" else kappa, feats):
            continue
        if pick(r):
            idx[(k[3], k[4])] = r
    out = []
    out.append("| rho \\ T | " + " | ".join("%g s" % t for t in ts) + " |")
    out.append("|---" * (len(ts) + 1) + "|")
    for rho in rhos:
        out.append("| %g | " % rho + " | ".join(cellstr(idx.get((rho, t))) for t in ts) + " |")
    return out


def main():
    rows = load()
    lines = ["# W23 tables (from cells.jsonl: %d cells)" % len(rows), ""]

    def mainpick(r):
        return (r["sign"] == 1 and r["background"] and r["contingent"]
                and r["dt"] == 0.1 and r["alpha_p"] == 0.1)

    lines.append("## Main grid (sign +1, background True, contingent True, dt 0.1, alpha_p 0.1)")
    for learner, kappa, label in LEARNERS:
        for feats in ("F1", "F2"):
            lines.append("")
            lines.append("### %s, %s" % (label, feats))
            lines += table(rows, mainpick, learner, kappa, feats, RHOS, TS)

    def broken(r):
        return (not r["contingent"] and r["rho"] == 4.0 and r["sign"] == 1
                and r["background"] and r["dt"] == 0.1 and r["alpha_p"] == 0.1)

    lines.append("")
    lines.append("## Broken stream (contingent False, rho 4, sign +1, background True)")
    for learner, kappa, label in LEARNERS:
        for feats in ("F1", "F2"):
            lines.append("")
            lines.append("### %s, %s" % (label, feats))
            lines += table(rows, broken, learner, kappa, feats, [4.0], [3.0, 300.0])

    spots = [
        ("(a) sign -1", lambda r: r["sign"] == -1 and r["background"] and r["contingent"]
         and r["dt"] == 0.1 and r["alpha_p"] == 0.1),
        ("(b) background False", lambda r: r["sign"] == 1 and not r["background"]
         and r["contingent"] and r["dt"] == 0.1 and r["alpha_p"] == 0.1),
        ("(c) alpha_p 0.03", lambda r: r["alpha_p"] == 0.03 and r["sign"] == 1
         and r["background"] and r["contingent"] and r["dt"] == 0.1),
        ("(c) alpha_p 0.3", lambda r: r["alpha_p"] == 0.3 and r["sign"] == 1
         and r["background"] and r["contingent"] and r["dt"] == 0.1),
        ("(d) dt 0.05", lambda r: r["dt"] == 0.05 and r["sign"] == 1
         and r["background"] and r["contingent"] and r["alpha_p"] == 0.1),
    ]
    for name, pick in spots:
        lines.append("")
        lines.append("## Spot %s (rho 4)" % name)
        for learner, kappa in (("L2", 1.0), ("L3", 1.0)):
            for feats in ("F1", "F2"):
                lines.append("")
                lines.append("### %s, %s" % (("%s kappa=1.0" % learner)
                                             if learner == "L2" else learner, feats))
                lines += table(rows, pick, learner, kappa, feats, [4.0], TS)

    txt = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    open(os.path.join(HERE, "out", "tables.md"), "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
