# Organ baseline tuning — completed 2026-09-11

Twelve validation-only screening jobs (three methods × four recipes, 10 epochs per task) and three fresh formal jobs (40 epochs per task) completed. The selected recipe is the maximum final four-task foreground validation mean for each method. Formal checkpoints are selected on current-task validation; test scores do not choose a checkpoint.

The live audit confirmed exit code zero, all 15 summaries, exact epoch coverage, finite training losses, four paired stage checkpoints per job, and screening test evaluation disabled. Training counts are 381/166/711/982 (T1/T3/T4 half, T2 full).

| Method | Selected recipe | Final foreground Dice |
|---|---|---:|
| zs-ewc | c1 | 0.245428 |
| zs-gpm | c3 | 0.310995 |
| zs-sequential | c1 | 0.207733 |

The runs completed technically but still show substantial old-task forgetting. These are **foreground** metrics; they must not be inserted as background-inclusive Dice in the thesis. No multi-seed variance or unmeasured inclusive metric is inferred. The formal schedules differ from the earlier inherited 60/60/10/10 reporting trajectory, so this is an additional baseline setting, not a budget-matched claim of superiority.

Public aggregate evidence: results/organ_baseline_tuning_20260911/completion.json. No patient data, per-patient scores, model files, or raw logs are included.
