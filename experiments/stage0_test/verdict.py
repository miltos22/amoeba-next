"""Stage-0 verdicts, exactly CRITERIA_DRAFT.md section 7 items 1 and 4-7.

Input: a CSV with columns ``cell,arm,seed,late_need`` and the five arms
BRAIN, BROKEN, STREAM-AC, BSTAR, FORAGER-SMELL.

Per cell:
    G     = mean(BSTAR) - mean(FORAGER-SMELL)
    D_A1(s) = BRAIN(s) - [BSTAR(s) - f_A * (BSTAR(s) - FORAGER-SMELL(s))]
    D_A2(s) = BRAIN(s) - BROKEN(s) + f_A * G
    D_K(s)  = BROKEN(s) - [BSTAR(s) - f_A * (BSTAR(s) - FORAGER-SMELL(s))]
    D_C(s)  = BRAIN(s) - STREAM-AC(s) - f_C * G
    For each D: m = mean, SD = sample SD over seeds, n, and the one-sided
    95 % Student-t bounds U = m + t(0.95, n-1) * SD / sqrt(n) and
    Lo = m - t * SD / sqrt(n).

Claim A (checks in this order: INVALID, FAIL, PASS, else INCONCLUSIVE):
    INVALID       U(D_K) < 0
    FAIL          Lo(D_A1) > 0 or Lo(D_A2) > 0
    PASS          U(D_A1) < 0 and U(D_A2) < 0
    INCONCLUSIVE  otherwise
Claim C:
    PASS          U(D_C) < 0
    FAIL          Lo(D_C) > 0
    INCONCLUSIVE  otherwise

Seeds are paired: every arm must carry exactly the same seed set in a
cell; a missing seed is an error, never dropped.

The t quantile is computed here, with numpy only (scipy is not in the
venv and nothing is installed): the Student-t CDF is evaluated from the
regularised incomplete beta function and inverted by bisection.  The
test asserts the three reference values of the brief to 4 decimals.

    PYTHONPATH=. .venv/bin/python experiments/stage0_test/verdict.py RUN.csv
"""

from __future__ import annotations

import argparse
import csv
import functools
import math

import numpy as np

ARMS = ("BRAIN", "BROKEN", "STREAM-AC", "BSTAR", "FORAGER-SMELL")
CLAIM_A_VERDICTS = ("PASS", "FAIL", "INVALID", "INCONCLUSIVE")
CLAIM_C_VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE")
CSV_COLUMNS = ("cell", "arm", "seed", "late_need")


# -- Student t quantile, numpy only ----------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc_regularised(a: float, b: float, x: float) -> float:
    """I_x(a, b): the regularised incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for a Student t with df degrees of freedom."""
    if df <= 0.0:
        raise ValueError(f"t_cdf needs df > 0, got {df!r}")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * _betainc_regularised(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0.0 else tail


@functools.lru_cache(maxsize=None)
def t_quantile_95(df: int) -> float:
    """The one-sided 95 % quantile t(0.95, df), by numerical inversion."""
    if df < 1:
        raise ValueError(f"need at least one degree of freedom, got {df!r}")
    lo = 0.0
    hi = 1.0
    while t_cdf(hi, df) < 0.95:
        hi *= 2.0
        if hi > 1.0e6:
            raise RuntimeError(f"t quantile did not bracket for df={df!r}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < 0.95:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1.0e-15:
            break
    return 0.5 * (lo + hi)


# -- data loading ----------------------------------------------------------
def load_csv(path: str) -> dict:
    """Read the verdict CSV into {cell: {arm: {seed: value}}}."""
    data: dict[str, dict[str, dict[int, float]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or ())
        missing = [name for name in CSV_COLUMNS if name not in fields]
        if missing:
            raise ValueError(f"{path}: CSV lacks columns {missing}; has {sorted(fields)}")
        for line_number, row in enumerate(reader, start=2):
            cell = row["cell"].strip()
            arm = row["arm"].strip()
            if not cell:
                raise ValueError(f"{path}:{line_number}: empty cell")
            if arm not in ARMS:
                raise ValueError(f"{path}:{line_number}: unknown arm {arm!r}; known: {ARMS}")
            seed = int(row["seed"])
            value = float(row["late_need"])
            arms = data.setdefault(cell, {name: {} for name in ARMS})
            if seed in arms[arm]:
                raise ValueError(f"{path}:{line_number}: duplicate {cell}/{arm}/seed {seed}")
            arms[arm][seed] = value
    if not data:
        raise ValueError(f"{path}: no data rows")
    return data


# -- one D statistic -------------------------------------------------------
def _stats(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    n = int(array.size)
    if n < 2:
        raise ValueError(f"need at least 2 paired seeds for an interval, got {n}")
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    half = t_quantile_95(n - 1) * sd / math.sqrt(n)
    return {"mean": mean, "sd": sd, "n": n, "U": mean + half, "Lo": mean - half}


def evaluate_cell(arms: dict, f_a: float = 1.0 / 3.0, f_c: float = 1.0 / 6.0) -> dict:
    """Apply section 7 to one cell's paired per-seed late_need values."""
    missing = [arm for arm in ARMS if arm not in arms or not arms[arm]]
    if missing:
        raise ValueError(f"cell is missing arm(s) {missing}")
    union = set()
    for arm in ARMS:
        union |= set(arms[arm])
    for arm in ARMS:
        seeds = set(arms[arm])
        if seeds != union:
            raise ValueError(
                f"seed sets differ: arm {arm} "
                f"missing {sorted(union - seeds)}, extra {sorted(seeds - union)}"
            )
    if len(union) < 2:
        raise ValueError(f"need at least 2 seeds, got {len(union)}")
    seeds = sorted(union)

    bstar = [arms["BSTAR"][s] for s in seeds]
    smell = [arms["FORAGER-SMELL"][s] for s in seeds]
    brain = [arms["BRAIN"][s] for s in seeds]
    broken = [arms["BROKEN"][s] for s in seeds]
    stream = [arms["STREAM-AC"][s] for s in seeds]

    g_value = float(np.mean(bstar) - np.mean(smell))
    line = [b - f_a * (b - f) for b, f in zip(bstar, smell)]
    d_a1 = [x - y for x, y in zip(brain, line)]
    d_a2 = [x - y + f_a * g_value for x, y in zip(brain, broken)]
    d_k = [x - y for x, y in zip(broken, line)]
    d_c = [x - y - f_c * g_value for x, y in zip(brain, stream)]

    stats = {
        "D_A1": _stats(d_a1),
        "D_A2": _stats(d_a2),
        "D_K": _stats(d_k),
        "D_C": _stats(d_c),
    }

    if stats["D_K"]["U"] < 0.0:
        claim_a = "INVALID"
    elif stats["D_A1"]["Lo"] > 0.0 or stats["D_A2"]["Lo"] > 0.0:
        claim_a = "FAIL"
    elif stats["D_A1"]["U"] < 0.0 and stats["D_A2"]["U"] < 0.0:
        claim_a = "PASS"
    else:
        claim_a = "INCONCLUSIVE"

    if stats["D_C"]["U"] < 0.0:
        claim_c = "PASS"
    elif stats["D_C"]["Lo"] > 0.0:
        claim_c = "FAIL"
    else:
        claim_c = "INCONCLUSIVE"

    return {
        "G": g_value,
        "f_a": float(f_a),
        "f_c": float(f_c),
        "n_seeds": len(seeds),
        **stats,
        "claim_A": claim_a,
        "claim_C": claim_c,
    }


def evaluate(data: dict, f_a: float = 1.0 / 3.0, f_c: float = 1.0 / 6.0) -> dict:
    """Apply evaluate_cell to every cell of a loaded CSV."""
    return {cell: evaluate_cell(data[cell], f_a=f_a, f_c=f_c) for cell in sorted(data)}


# -- printing --------------------------------------------------------------
def print_result(cell: str, result: dict) -> None:
    print(
        f"cell={cell} G={result['G']!r} f_a={result['f_a']!r} f_c={result['f_c']!r} "
        f"n_seeds={result['n_seeds']}"
    )
    for name in ("D_A1", "D_A2", "D_K", "D_C"):
        stats = result[name]
        print(
            f"  {name} mean={stats['mean']!r} sd={stats['sd']!r} n={stats['n']} "
            f"U={stats['U']!r} Lo={stats['Lo']!r}"
        )
    print(f"  claim_A={result['claim_A']}")
    print(f"  claim_C={result['claim_C']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV with columns cell,arm,seed,late_need")
    parser.add_argument("--f-a", type=float, default=1.0 / 3.0)
    parser.add_argument("--f-c", type=float, default=1.0 / 6.0)
    args = parser.parse_args(argv)
    results = evaluate(load_csv(args.csv), f_a=args.f_a, f_c=args.f_c)
    for cell in sorted(results):
        print_result(cell, results[cell])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
