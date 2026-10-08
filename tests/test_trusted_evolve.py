import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from tevolve.loop import ClaudeProposer, RandomProposer, research_loop, validate
from tevolve.problems.circle_packing import CirclePacking
from tevolve.search import evolve, random_restart
from tevolve.stats import decision_error

P = CirclePacking(26)


def test_verifier_rejects_overlap_and_escape():
    z = P.random_start(np.random.default_rng(0))
    z[2 * 26:] = 0.0
    assert P.verify(z).valid and P.verify(z).value == 0.0
    bad = z.copy()
    bad[0], bad[26], bad[52] = 0.05, 0.5, 0.06          # crosses the left edge
    assert P.verify(bad).reason == "circle leaves the unit square"
    two = z.copy()
    two[[0, 1]], two[[26, 27]], two[[52, 53]] = 0.5, 0.5, 0.1   # identical centres
    assert P.verify(two).reason == "circles overlap"


def test_verifier_has_zero_tolerance():
    z = np.zeros(3 * 26)
    z[:26] = np.linspace(0.02, 0.98, 26)
    z[26:52] = 0.5
    z[52:] = 0.0
    z[52] = 0.02 + 1e-15                                 # exceeds the wall by 1e-15
    assert not P.verify(z).valid


def test_repair_always_yields_valid():
    rng = np.random.default_rng(1)
    for _ in range(20):
        z = np.concatenate([rng.uniform(0, 1, 26), rng.uniform(0, 1, 26), rng.uniform(0, 0.3, 26)])
        assert P.verify(P.repair(z)).valid


def test_analytic_jacobian_matches_finite_differences():
    z = P.random_start(np.random.default_rng(2))
    z[52:] = 0.05
    J = P._jacobian(z)
    for row in (0, 30, 60, 90, 150, J.shape[0] - 1):
        fd = approx_fprime(z, lambda v: P._constraints(v)[row], 1e-7)
        assert np.allclose(J[row], fd, atol=1e-5)


def test_search_is_deterministic_and_verified():
    a = random_restart(P, seed=3, budget=3)
    b = random_restart(P, seed=3, budget=3)
    assert a.best == b.best and P.verify(np.array(a.best_z)).valid
    e = evolve(CirclePacking(8), seed=0, budget=12, pop_size=6, init=4)
    assert e.curve == sorted(e.curve) and P.__class__(8).verify(np.array(e.best_z)).valid


def test_decision_error():
    assert decision_error([3, 3, 3], [1, 1, 1], k=1) == 0.0
    assert 0 < decision_error([1, 5, 2, 6], [3, 4, 3, 4], k=1) < 1


def test_validate_clips_out_of_range_proposals():
    v = validate({"hypothesis": "x", "operators": ["swap", "teleport"], "weights": [-1], "pop_size": 999, "elite": 0})
    assert v["operators"] == ["swap"] and v["weights"] == [1.0] and v["pop_size"] == 64 and v["elite"] == 1


def test_loop_gate_with_random_proposer_small():
    ledger = research_loop(RandomProposer(0), rounds=2, seeds=[0, 1, 2], budget=6, n=6, workers=1)
    assert ledger[0]["decision"] == "baseline" and len(ledger) == 3
    assert all(t["decision"] in ("accepted", "rejected") for t in ledger[1:])


def test_claude_proposer_parses_structured_output():
    payload = {"hypothesis": "More relocation helps escape local optima", "operators": ["relocate_small", "jitter"],
               "weights": [3, 1], "pop_size": 24, "elite": 4}
    calls = []

    def create(**kw):
        calls.append(kw)
        return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(payload))])
    prop = ClaudeProposer(client=NS(messages=NS(create=create))).propose([])
    assert prop == payload
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"
