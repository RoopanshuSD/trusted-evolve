"""A small autonomous research loop: propose -> run on k seeds -> statistical acceptance gate -> ledger.

The proposer (random baseline, or Claude) only *suggests* a search configuration together with a hypothesis.
Code runs it on fixed seeds and decides: a proposal replaces the incumbent only if its mean beats the incumbent's and
Welch's test gives p < alpha. Every trial, accepted or rejected, is written to the ledger the next proposal reads.

    python -m tevolve.loop --proposer random --rounds 4 --seeds 5 --budget 300
    python -m tevolve.loop --proposer claude --rounds 6            # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .problems.circle_packing import CirclePacking
from .search import evolve
from .stats import compare

DEFAULT = {"operators": ["relocate_small", "jitter", "swap"], "weights": [1, 1, 1], "pop_size": 20, "elite": 5}

PROPOSAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["hypothesis", "operators", "weights", "pop_size", "elite"],
    "properties": {
        "hypothesis": {"type": "string"},
        "operators": {"type": "array", "items": {"type": "string", "enum": list(CirclePacking.OPERATORS)}},
        "weights": {"type": "array", "items": {"type": "number"}},
        "pop_size": {"type": "integer"},
        "elite": {"type": "integer"},
    },
}


def validate(p: dict) -> dict:
    """Code-side bounds check: proposals outside the allowed space are clipped or rejected."""
    ops = [o for o in p["operators"] if o in CirclePacking.OPERATORS]
    if not ops:
        raise ValueError("no valid operators")
    w = [max(float(x), 0.0) for x in (p.get("weights") or [1] * len(ops))][: len(ops)]
    w += [1.0] * (len(ops) - len(w))
    if sum(w) == 0:
        w = [1.0] * len(ops)
    pop = int(min(max(p["pop_size"], 4), 64))
    return {"hypothesis": str(p.get("hypothesis", ""))[:500], "operators": ops, "weights": w, "pop_size": pop,
            "elite": int(min(max(p["elite"], 1), pop))}


def _one(args):
    cfg, seed, budget, n = args
    return evolve(CirclePacking(n), seed, budget, pop_size=cfg["pop_size"], elite=cfg["elite"],
                  operators=tuple(cfg["operators"]), weights=tuple(cfg["weights"])).best


def evaluate(cfg: dict, seeds: list[int], budget: int, n: int, workers: int) -> list[float]:
    jobs = [(cfg, s, budget, n) for s in seeds]
    if workers <= 1:
        return [_one(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_one, jobs))


class RandomProposer:
    """Baseline: samples configurations uniformly. Same gate, same budget."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def propose(self, ledger: list[dict]) -> dict:
        k = int(self.rng.integers(1, 4))
        ops = list(self.rng.choice(CirclePacking.OPERATORS, k, replace=False))
        return {"hypothesis": "random sample", "operators": ops, "weights": self.rng.uniform(0.2, 1, k).round(2).tolist(),
                "pop_size": int(self.rng.integers(8, 41)), "elite": int(self.rng.integers(2, 9))}


class ClaudeProposer:
    """Claude reads the ledger and proposes the next experiment as structured JSON (schema-constrained)."""

    MODEL = os.environ.get("TEVOLVE_MODEL", "claude-opus-5-5")

    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def propose(self, ledger: list[dict]) -> dict:
        history = json.dumps([{k: t[k] for k in ("hypothesis", "config", "mean", "decision", "p_value")}
                              for t in ledger], indent=1)
        prompt = (
            "You are running a research loop on circle packing (26 circles in the unit square, maximise the sum of "
            "radii). The search is an elite-population evolution: parents are mutated by operators "
            f"{list(CirclePacking.OPERATORS)} (relocate_small: move the 1-3 smallest circles to random spots; "
            "jitter: Gaussian noise on all centres; swap: exchange two circles), then locally optimised with SLSQP. "
            "A proposal is accepted only if it beats the incumbent's mean over fixed seeds with Welch p < 0.05.\n\n"
            f"Ledger so far:\n{history}\n\nPropose ONE new configuration that tests a specific hypothesis you can "
            "state in one sentence. Avoid repeating rejected configurations.")
        resp = self.client.messages.create(
            model=self.MODEL, max_tokens=4000, messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": PROPOSAL_SCHEMA}})
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused")
        return json.loads(next(b.text for b in resp.content if b.type == "text"))


def research_loop(proposer, rounds: int, seeds: list[int], budget: int, n: int = 26, alpha: float = 0.05,
                  workers: int = 8) -> list[dict]:
    inc_cfg = dict(DEFAULT)
    inc_scores = evaluate(inc_cfg, seeds, budget, n, workers)
    ledger = [{"round": 0, "hypothesis": "incumbent (default config)", "config": inc_cfg, "scores": inc_scores,
               "mean": float(np.mean(inc_scores)), "decision": "baseline", "p_value": None}]
    for rnd in range(1, rounds + 1):
        try:
            prop = validate(proposer.propose(ledger))
        except Exception as e:  # malformed proposal: recorded, not fatal
            ledger.append({"round": rnd, "hypothesis": f"invalid proposal: {e}", "config": None, "scores": [],
                           "mean": None, "decision": "invalid", "p_value": None})
            continue
        cfg = {k: prop[k] for k in ("operators", "weights", "pop_size", "elite")}
        scores = evaluate(cfg, seeds, budget, n, workers)
        c = compare(scores, inc_scores)
        accept = c["diff_mean"] > 0 and c["welch_p"] < alpha
        ledger.append({"round": rnd, "hypothesis": prop["hypothesis"], "config": cfg, "scores": scores,
                       "mean": float(np.mean(scores)), "decision": "accepted" if accept else "rejected",
                       "p_value": c["welch_p"], "diff_vs_incumbent": c["diff_mean"]})
        if accept:
            inc_cfg, inc_scores = cfg, scores
    return ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", choices=["random", "claude"], default="random")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/loop")
    a = ap.parse_args()
    proposer = RandomProposer() if a.proposer == "random" else ClaudeProposer()
    ledger = research_loop(proposer, a.rounds, list(range(100, 100 + a.seeds)), a.budget, workers=a.workers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ledger_{a.proposer}.json").write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    for t in ledger:
        m = "-" if t["mean"] is None else f"{t['mean']:.6f}"
        p = "" if t["p_value"] is None else f" p={t['p_value']:.3g}"
        print(f"round {t['round']}: {t['decision']:9} mean={m}{p}  {t['hypothesis'][:70]}")


if __name__ == "__main__":
    main()
