# Organ and Domain replay comparisons — launched 2026-09-11

Four fixed fresh comparisons are running. This report is startup evidence, not a completed experiment result.

| Setting | Comparison | GPU at startup | Budget |
|---|---|---:|---|
| Organ | Feature-DER | 3 | 4 tasks × 40 epochs |
| Organ | ER | 6 | 4 tasks × 40 epochs |
| Domain | Feature-DER | 5 | 6 domains × 80 epochs |
| Domain | ER | 3 | 6 domains × 80 epochs |

At the live check on 2026-09-11 23:27 CST, all four had completed finite optimizer updates: Organ DER iteration 214, Organ ER 146, Domain DER 218, Domain ER 93. No traceback or CUDA OOM was present. The coordinator and descendants used the neutral command ./main -u run.py; GPU processes displayed ./main.

Cross-task checks passed for both settings and both replay terms, including paired checkpoint reload, inactive-loss assertions, source accounting, and foreground/background-inclusive Dice arithmetic. Maximum measured CUDA reserved memory was 9356 MiB; admission requires 10856 MiB free. Other users' future allocations remain outside this scheduler's control.

The Organ train counts remain 381/166/711/982. Domain inputs were copied to NAS once and are shared across the two comparisons. All large outputs, logs, cache and temporary files stay on NAS. The existing long-running jobs were not stopped or restarted.

See ../replay_runtime/README.md for the complete protocol and ../results/replay_comparisons_20260911/startup.json for scalar startup evidence. Reusable source and the bounded checks are in ../replay_runtime/. Test data are not used for selection. The new controls output background-inclusive Dice, with foreground Dice retained separately. RMA is deferred to a verified independent-reference denominator; it is not set to zero or estimated from another metric.
