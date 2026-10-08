"""Multi-seed study: every strategy x every seed, same budget. Writes results/ (runs, summary, figures, best solution).

    python -m tevolve.study --seeds 10 --budget 600 --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .problems.circle_packing import CirclePacking
from .search import STRATEGIES
from .stats import compare, decision_error, summarize

ALPHAEVOLVE_2025 = 2.63586276      # Novikov et al. 2025 (AlphaEvolve), n=26
BEST_KNOWN = 2.635983084918        # Packomania csqv list, n=26 (matched by ShinkaEvolve, 2025)

CONFIGS = {
    "random_restart": ("random_restart", {}),
    "evolve": ("evolve", {}),
    "evolve_no_swap": ("evolve", {"operators": ("relocate_small", "jitter")}),
}


def _run(args):
    name, seed, budget, n = args
    strat, kw = CONFIGS[name]
    res = STRATEGIES[strat](CirclePacking(n), seed, budget, **kw)
    d = asdict(res)
    d["config_name"] = name
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--n", type=int, default=26)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(c, s, a.budget, a.n) for c in CONFIGS for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        runs = list(ex.map(_run, jobs))
    with (out / "runs.jsonl").open("w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps({k: v for k, v in r.items() if k != "best_z"}) + "\n")

    prob = CirclePacking(a.n)
    best_run = max(runs, key=lambda r: r["best"])
    check = prob.verify(np.array(best_run["best_z"]))
    assert check.valid, check.reason
    x, y, r = prob.split(np.array(best_run["best_z"]))
    (out / f"best_n{a.n}.json").write_text(json.dumps({
        "n": a.n, "sum_of_radii": check.value, "verifier": "zero tolerance",
        "found_by": {k: best_run[k] for k in ("config_name", "seed", "budget", "seconds")},
        "circles": [{"x": float(xi), "y": float(yi), "r": float(ri)} for xi, yi, ri in zip(x, y, r)]}, indent=1))

    finals = {c: [r["best"] for r in runs if r["config_name"] == c] for c in CONFIGS}
    summary = {c: summarize(v) for c, v in finals.items()}
    for c, v in finals.items():
        summary[c]["beat_alphaevolve"] = int(sum(x > ALPHAEVOLVE_2025 for x in v))
        summary[c]["within_1e-6_of_best_known"] = int(sum(x >= BEST_KNOWN - 1e-6 for x in v))
        summary[c]["median_seconds"] = float(np.median([r["seconds"] for r in runs if r["config_name"] == c]))
    pairs = {}
    for p, q in (("evolve", "random_restart"), ("evolve", "evolve_no_swap")):
        pairs[f"{p} vs {q}"] = {**compare(finals[p], finals[q]),
                                **{f"decision_error_k{k}": decision_error(finals[p], finals[q], k) for k in (1, 3, 5)}}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "comparisons": pairs,
                                                  "best": check.value, "seeds": a.seeds, "budget": a.budget}, indent=1))

    lines = [f"n={a.n}, {a.seeds} seeds per strategy, budget = {a.budget} local solves per run.", "",
             "| Strategy | Mean best | 95% CI (bootstrap) | Max | Runs > AlphaEvolve (2.63586) | Runs at best known (±1e-6) | Median time |",
             "|---|---|---|---|---|---|---|"]
    for c, s in summary.items():
        lines.append(f"| `{c}` | {s['mean']:.6f} | [{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}] | {s['max']:.9f} | "
                     f"{s['beat_alphaevolve']}/{s['n']} | {s['within_1e-6_of_best_known']}/{s['n']} | {s['median_seconds']:.0f} s |")
    lines += ["", "| Comparison | Δ mean | Welch p | Mann-Whitney p | Wrong call with 1 seed | 3 seeds | 5 seeds |",
              "|---|---|---|---|---|---|---|"]
    for k, v in pairs.items():
        lines.append(f"| {k} | {v['diff_mean']:+.6f} | {v['welch_p']:.3g} | {v['mannwhitney_p']:.3g} | "
                     f"{v['decision_error_k1']:.0%} | {v['decision_error_k3']:.0%} | {v['decision_error_k5']:.0%} |")
    lines += ["", f"Best verified packing: **{check.value:.12f}** (`{best_run['config_name']}`, seed {best_run['seed']}, "
              f"{best_run['seconds']:.0f} s) — saved to `best_n{a.n}.json`."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _figures(runs, prob, best_run, out)
    print("\n".join(lines))


def _figures(runs, prob, best_run, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for c, color in zip(CONFIGS, ("#888888", "#1f4e9a", "#d08c2c")):
        curves = np.array([r["curve"] for r in runs if r["config_name"] == c])
        m = curves.mean(0)
        ax.plot(m, color=color, label=c)
        ax.fill_between(range(curves.shape[1]), curves.min(0), curves.max(0), color=color, alpha=0.15)
    ax.axhline(ALPHAEVOLVE_2025, ls="--", color="k", lw=0.8, label="AlphaEvolve (2025)")
    ax.set_ylim(2.60, 2.6375)
    ax.set_xlabel("local solves")
    ax.set_ylabel("best verified sum of radii")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Anytime performance, n=26 (mean, min-max band over seeds)")
    fig.tight_layout()
    fig.savefig(out / "anytime_n26.png", dpi=150)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1.2))
    x, y, r = prob.split(np.array(best_run["best_z"]))
    for xi, yi, ri in zip(x, y, r):
        ax.add_patch(Circle((xi, yi), ri, fc="#1f4e9a", ec="white", alpha=0.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"n=26, sum of radii = {prob.verify(np.array(best_run['best_z'])).value:.9f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "best_n26.png", dpi=150)


if __name__ == "__main__":
    main()
