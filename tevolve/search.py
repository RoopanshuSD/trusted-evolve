"""Search strategies. Each gets the same budget, counted in local-optimiser calls, and is deterministic per seed.

  random_restart : independent random starts + local optimisation (the strong classical baseline)
  evolve         : elite population; parents are mutated by operators, re-optimised, and kept only if verified-better
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .problems.circle_packing import CirclePacking


@dataclass
class RunResult:
    strategy: str
    seed: int
    budget: int
    best: float
    best_z: list
    curve: list[float]          # best verified score after each local solve (anytime curve)
    seconds: float
    config: dict = field(default_factory=dict)
    operator_wins: dict = field(default_factory=dict)


def random_restart(prob: CirclePacking, seed: int, budget: int, **_) -> RunResult:
    rng = np.random.default_rng(seed)
    t0, best, best_z, curve = time.perf_counter(), 0.0, None, []
    for _ in range(budget):
        z = prob.local_optimize(prob.random_start(rng))
        s = prob.verify(z)
        if s.valid and s.value > best:
            best, best_z = s.value, z
        curve.append(best)
    return RunResult("random_restart", seed, budget, best, best_z.tolist(), curve, time.perf_counter() - t0)


def evolve(prob: CirclePacking, seed: int, budget: int, pop_size: int = 20, init: int = 20, elite: int = 5,
           operators: tuple[str, ...] = CirclePacking.OPERATORS, weights: tuple[float, ...] | None = None) -> RunResult:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    weights = np.asarray(weights or [1.0] * len(operators), dtype=float)
    weights = weights / weights.sum()
    pop, curve, wins = [], [], {op: 0 for op in operators}
    best = 0.0
    for _ in range(min(init, budget)):
        z = prob.local_optimize(prob.random_start(rng))
        s = prob.verify(z)
        pop.append((s.value if s.valid else 0.0, z))
        best = max(best, pop[-1][0])
        curve.append(best)
    for _ in range(budget - len(curve)):
        pop.sort(key=lambda p: -p[0])
        parent = pop[int(rng.integers(min(elite, len(pop))))][1]
        op = str(rng.choice(operators, p=weights))
        child = prob.local_optimize(getattr(prob, f"op_{op}")(parent, rng))
        s = prob.verify(child)
        val = s.value if s.valid else 0.0
        if val > best:
            wins[op] += 1
            best = val
        pop.append((val, child))
        pop = sorted(pop, key=lambda p: -p[0])[:pop_size]
        curve.append(best)
    pop.sort(key=lambda p: -p[0])
    return RunResult("evolve", seed, budget, pop[0][0], pop[0][1].tolist(), curve, time.perf_counter() - t0,
                     config={"pop_size": pop_size, "elite": elite, "operators": list(operators),
                             "weights": weights.round(3).tolist()}, operator_wins=wins)


STRATEGIES = {"random_restart": random_restart, "evolve": evolve}
