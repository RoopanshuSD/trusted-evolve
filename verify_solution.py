"""Standalone, dependency-light verifier (numpy only). Independent of the search code on purpose.

    python verify_solution.py results/best_n26.json

Checks with ZERO tolerance: every circle inside the unit square, no two circles overlapping, radii >= 0.
"""
import json
import sys

import numpy as np


def verify(path: str) -> float:
    sol = json.load(open(path, encoding="utf-8"))
    c = sol["circles"]
    x, y, r = (np.array([k[a] for k in c], dtype=float) for a in ("x", "y", "r"))
    assert len(c) == sol["n"], "wrong number of circles"
    assert (r >= 0).all(), "negative radius"
    assert (x - r >= 0).all() and (x + r <= 1).all() and (y - r >= 0).all() and (y + r <= 1).all(), "outside square"
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            assert (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 >= (r[i] + r[j]) ** 2, f"circles {i} and {j} overlap"
    return float(r.sum())


if __name__ == "__main__":
    total = verify(sys.argv[1] if len(sys.argv) > 1 else "results/best_n26.json")
    print(f"VALID  n=26  sum of radii = {total:.12f}")
