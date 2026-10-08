"""Circle packing: place n disjoint circles inside the unit square to maximise the sum of their radii.

This is the n=26 problem from DeepMind's AlphaEvolve paper (2025), and the case Aemon uses as its showcase.

A candidate is a flat vector z = [x_1..x_n, y_1..y_n, r_1..r_n]. The *evaluator* is the trust boundary:
`verify()` uses zero tolerance (no epsilon slack for containment or overlap), and only verified candidates get a
score. The local optimiser may return slightly infeasible points, so every candidate goes through `repair()`,
which shrinks radii until the zero-tolerance check passes. The search therefore never earns credit for an
almost-valid packing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class Score:
    valid: bool
    value: float
    reason: str = ""


class CirclePacking:
    def __init__(self, n: int = 26):
        self.n = n
        self.iu = np.triu_indices(n, 1)
        self.dim = 3 * n

    # ------------------------------------------------------------------ representation
    def split(self, z: np.ndarray):
        n = self.n
        return z[:n], z[n:2 * n], z[2 * n:]

    def random_start(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        return np.concatenate([rng.uniform(.1, .9, n), rng.uniform(.1, .9, n), np.full(n, 0.02)])

    # ------------------------------------------------------------------ trust boundary
    def verify(self, z: np.ndarray) -> Score:
        """Zero-tolerance check. Returns the score only for a valid packing."""
        z = np.asarray(z, dtype=float)
        if z.shape != (self.dim,) or not np.all(np.isfinite(z)):
            return Score(False, 0.0, "malformed candidate")
        x, y, r = self.split(z)
        if np.any(r < 0):
            return Score(False, 0.0, "negative radius")
        if np.any(x - r < 0) or np.any(x + r > 1) or np.any(y - r < 0) or np.any(y + r > 1):
            return Score(False, 0.0, "circle leaves the unit square")
        a, b = self.iu
        d2 = (x[a] - x[b]) ** 2 + (y[a] - y[b]) ** 2
        if np.any(d2 < (r[a] + r[b]) ** 2):
            return Score(False, 0.0, "circles overlap")
        return Score(True, float(r.sum()))

    def repair(self, z: np.ndarray, max_rounds: int = 100) -> np.ndarray:
        """Shrink radii until the zero-tolerance verifier passes (centres unchanged)."""
        x, y, r = (v.copy() for v in self.split(np.asarray(z, dtype=float)))
        x, y = np.clip(x, 0, 1), np.clip(y, 0, 1)
        r = np.maximum(np.minimum.reduce([r, x, 1 - x, y, 1 - y]), 0.0)
        a, b = self.iu
        for _ in range(max_rounds):
            d = np.sqrt((x[a] - x[b]) ** 2 + (y[a] - y[b]) ** 2)
            over = r[a] + r[b] - d
            bad = over > 0
            if not bad.any():
                break
            for i, j, o in zip(a[bad], b[bad], over[bad]):
                tot = r[i] + r[j]
                if tot > 0:
                    r[i] -= o * r[i] / tot * (1 + 1e-9)
                    r[j] -= o * r[j] / tot * (1 + 1e-9)
            r = np.maximum(r, 0.0)
        out = np.concatenate([x, y, r])
        # floating-point safety: nudge down until the strict check passes
        for _ in range(60):
            if self.verify(out).valid:
                return out
            out[2 * self.n:] *= (1 - 1e-12)
        return out

    # ------------------------------------------------------------------ local optimiser
    def _constraints(self, z):
        x, y, r = self.split(z)
        a, b = self.iu
        pair = (x[a] - x[b]) ** 2 + (y[a] - y[b]) ** 2 - (r[a] + r[b]) ** 2
        return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r, pair, r])

    def _jacobian(self, z):
        n = self.n
        x, y, r = self.split(z)
        a, b = self.iu
        m = len(a)
        J = np.zeros((5 * n + m, 3 * n))
        I = np.arange(n)
        J[I, I], J[I, 2 * n + I] = 1, -1
        J[n + I, I], J[n + I, 2 * n + I] = -1, -1
        J[2 * n + I, n + I], J[2 * n + I, 2 * n + I] = 1, -1
        J[3 * n + I, n + I], J[3 * n + I, 2 * n + I] = -1, -1
        dx, dy, s, k = x[a] - x[b], y[a] - y[b], r[a] + r[b], np.arange(m) + 4 * n
        J[k, a], J[k, b] = 2 * dx, -2 * dx
        J[k, n + a], J[k, n + b] = 2 * dy, -2 * dy
        J[k, 2 * n + a], J[k, 2 * n + b] = -2 * s, -2 * s
        J[4 * n + m + I, 2 * n + I] = 1
        return J

    def local_optimize(self, z0: np.ndarray, maxiter: int = 1000) -> np.ndarray:
        """SLSQP on max sum(r) with analytic gradients; result is repaired to strict feasibility."""
        n = self.n
        grad = np.zeros(self.dim)
        grad[2 * n:] = -1.0
        res = minimize(lambda z: -z[2 * n:].sum(), z0, jac=lambda z: grad, method="SLSQP",
                       constraints=[{"type": "ineq", "fun": self._constraints, "jac": self._jacobian}],
                       options={"maxiter": maxiter, "ftol": 1e-12})
        return self.repair(res.x)

    # ------------------------------------------------------------------ mutation operators (used by evolution)
    def op_relocate_small(self, z, rng):
        x, y, r = (v.copy() for v in self.split(z))
        k = int(rng.integers(1, 4))
        idx = np.argsort(r)[:k]
        x[idx], y[idx], r[idx] = rng.uniform(.05, .95, k), rng.uniform(.05, .95, k), 0.01
        return np.concatenate([x, y, r])

    def op_jitter(self, z, rng):
        x, y, r = (v.copy() for v in self.split(z))
        s = float(rng.choice([0.01, 0.03, 0.06]))
        x = np.clip(x + rng.normal(0, s, self.n), .01, .99)
        y = np.clip(y + rng.normal(0, s, self.n), .01, .99)
        return np.concatenate([x, y, r * 0.9])

    def op_swap(self, z, rng):
        x, y, r = (v.copy() for v in self.split(z))
        i, j = rng.choice(self.n, 2, replace=False)
        x[[i, j]], y[[i, j]] = x[[j, i]], y[[j, i]]
        return np.concatenate([x, y, r])

    OPERATORS = ("relocate_small", "jitter", "swap")
