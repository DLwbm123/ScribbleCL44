**Completed:** see [the verified completion report](../reports/class_replay_completion_20260915.md). The launch/recovery notes below are historical. For the completed DER variant, overlay `guarded/*.py` into its source directory after the common overlay.

# Class ER and feature-DER benchmark comparisons

This is a four-file overlay for the existing Class runtime. Copy the files from class_runtime/ into a campaign's source/ directory, then overlay these Python files. The training loop remains runner_core.main("class"). The prototype MiB component plan was cancelled before any training was launched.

Two fixed fresh runs: ZS-ER and ZS-DER, seed 42, tasks T1/T2/T3, 80 epochs per task, batch 4, workers 4, SGD LR .03, momentum .9, weight decay .0001. Current-image PCE/Global/Spatial = 1/.1/0. Reservoir capacity 64, replay minibatch 4. No extra sweep or automatic method expansion.

- ER: replay sparse PCE coefficient 1; feature and replay-global coefficients zero. Historical sparse labels are classified over all currently seen classes.
- Feature-DER: shared U-Net feature MSE coefficient .5; replay PCE/global coefficients zero. This follows the feature-based DER used by the benchmark implementation, rather than claiming a standard logit-DER reproduction.
- Shared-backbone BN statistics are frozen for the new controls' historical replay/feature capture. The existing Class augmented-image reservoir and task/source metadata are reused.
- Neither method uses MiB. Their objectives differ from the first task, so incompatible previously trained T1 models are not reused.
- These are scribble-supervised adaptations. The original benchmark buffer was 32; the present Class/Organ/Domain replay-control budget is 64.

Checkpoint selection remains current-task foreground validation Dice across all 80 epochs. Background-inclusive per-case Dice, full stage evaluations, whole-class Dice, fixed-model parameter counts, and original-source replay counts are retained for benchmark metrics. RMA uses the established common independent references when results are aggregated. Existing Class checkpoint behavior is preserved: selected model weights plus the stage-end reservoir are carried onward.

The H5 cache follows the existing Organ loading pattern: arrays are read into host RAM once per dataset instance and released on close. Synthetic-input checks assert exact cached/uncached image and label parity, including the global sparse labels and shifted validation labels of all three tasks. Both methods execute two updates per task across all three tasks, with finite-loss, active/disabled replay-branch, and historical-source assertions. These small checks are engineering tests, not scientific Dice results. A slow private-file fixture extraction attempt was stopped before training; its artifacts are not reused.

Provide private data/sparse paths in the campaign root's paths.json. All outputs, temporary files and caches stay on the designated remote-home mount. Start the existing Python interpreter with the neutral relative entry run.py. Runtime arguments are passed in environment; process titles are job/run. The detached coordinator waits for checks/passed.json and then admits both runs on GPU2 when measured memory plus headroom is available. No continuous Codex monitoring is configured.

## Recovery on 2026-09-13

The original ER finished T1 (80 epochs) and 9 epochs of T2 before the user pause.
`--resume-from` restores the completed task's selected model, stage-end reservoir,
source identities and metric prefix into a fresh output directory. Only completed
stages are reused; T2 is restarted for 80 epochs and followed by T3 for 80 epochs.
The historical checkpoint contains no RNG state, so this is a task-boundary
restart with seed 42, not an exact continuation of the interrupted trajectory.
`check_resume.py` exercises checkpoint loading, historical replay, next-task
updates, and preservation of the earlier metric row.

Original DER failed during T1 epoch 14 with non-finite loss, after 13 epochs whose
validation predictions were entirely background. It has no completed task state.
A paired 32-update real-input BatchNorm check did not reproduce the failure or
establish BN as its cause; the batch-statistics variant was not adopted.
The recovery keeps the original replay BN rule, LR .03 and feature coefficient .5,
starts from scratch, and enables `--safe-numerics --grad-clip-norm 5`.
This is a documented numerical-stability variant, not a confirmed diagnosis of
the old failure. It reuses the existing `numerical_safety.py` and the guarded
`replay_runtime/mixup.py` for saliency normalization and checked graph-cut costs.
Actual non-finite gradients remain fatal before the optimizer update.

`job.py` runs one detached job per JSON `RUN_SPEC` environment value, with distinct
GPU2/3 assignments and receipts. The ER runtime snapshot remains unchanged while
the guarded DER source is in a separate directory. Runtime commands are neutral
`job.py` / `run.py`, with titles `run` / `job`. Inputs and old outputs are retained.

At the user's later request, `--task-epochs 80 80 60` (ER) and
`--task-epochs 80 60 60` (DER) set explicit budgets. Completed stages are reused;
the next task restarts with polynomial LR decay over its new budget. The same
budgets appear in the manifest, summary, and completion checks. The updated
resume regression exercises unequal task lengths. These runs share GPU3 and
live in the separate `class_replay_60e_20260913` campaign.
