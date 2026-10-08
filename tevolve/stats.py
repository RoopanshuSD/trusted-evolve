"""Statistics for comparing search strategies across seeds.

Autonomous research loops make the same mistake as human ones: they compare one run of A with one run of B and
call the winner an improvement. `decision_error` measures how often a k-seed comparison picks the wrong strategy
relative to the decision made on all seeds.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats


def summarize(xs) -> dict:
    xs = np.asarray(xs, dtype=float)
    lo, hi = bootstrap_ci(xs)
    return {"n": int(xs.size), "mean": float(xs.mean()), "std": float(xs.std(ddof=1)) if xs.size > 1 else 0.0,
            "min": float(xs.min()), "max": float(xs.max()), "ci95": [lo, hi]}


def bootstrap_ci(xs, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    xs = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, size=(n_boot, xs.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(a, b) -> dict:
    """Welch's t-test plus a rank test; positive diff means a > b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"diff_mean": float(a.mean() - b.mean()), "welch_p": float(t.pvalue), "mannwhitney_p": float(u.pvalue)}


def decision_error(a, b, k: int, max_combos: int = 20_000, seed: int = 0) -> float:
    """P(k-seed mean comparison disagrees with the all-seed comparison), over subsets of seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    truth = a.mean() > b.mean()
    idx_a, idx_b = range(a.size), range(b.size)
    total = math.comb(a.size, k) * math.comb(b.size, k)
    rng = np.random.default_rng(seed)
    if total <= max_combos:
        pairs = itertools.product(itertools.combinations(idx_a, k), itertools.combinations(idx_b, k))
    else:
        pairs = ((rng.choice(a.size, k, replace=False), rng.choice(b.size, k, replace=False)) for _ in range(max_combos))
    wrong = n = 0
    for ia, ib in pairs:
        n += 1
        wrong += (a[list(ia)].mean() > b[list(ib)].mean()) != truth
    return wrong / n
