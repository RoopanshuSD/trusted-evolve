n=26, 10 seeds per strategy, budget = 600 local solves per run.

| Strategy | Mean best | 95% CI (bootstrap) | Max | Runs > AlphaEvolve (2.63586) | Runs at best known (±1e-6) | Median time |
|---|---|---|---|---|---|---|
| `random_restart` | 2.631002 | [2.629298, 2.632681] | 2.635983085 | 1/10 | 1/10 | 113 s |
| `evolve` | 2.632320 | [2.631157, 2.633656] | 2.635983085 | 2/10 | 2/10 | 63 s |
| `evolve_no_swap` | 2.635071 | [2.634196, 2.635814] | 2.635983085 | 6/10 | 6/10 | 71 s |

| Comparison | Δ mean | Welch p | Mann-Whitney p | Wrong call with 1 seed | 3 seeds | 5 seeds |
|---|---|---|---|---|---|---|
| evolve vs random_restart | +0.001317 | 0.263 | 0.14 | 30% | 24% | 13% |
| evolve vs evolve_no_swap | -0.002752 | 0.0036 | 0.0191 | 18% | 2% | 0% |

Best verified packing: **2.635983084918** (`evolve_no_swap`, seed 5, 73 s) — saved to `best_n26.json`.
