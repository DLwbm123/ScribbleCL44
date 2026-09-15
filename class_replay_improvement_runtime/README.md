# Class replay improvement campaign, 2026-09-15

This is a separate, validation-gated improvement campaign for the completed Class ER / guarded feature-DER controls. It is not an unmodified baseline reproduction or evidence of improved performance yet. The original runtime and all earlier checkpoints/results are preserved.

## Changes

The `--improved-replay` flag is limited to Class ER and feature-DER. After T1, all BatchNorm running statistics are frozen consistently across current and replay forwards. Foreground labels retain their global IDs; annotated background marginalizes classes outside the source task instead of forcing those unknown classes to zero probability. ER applies that partial-label objective to current and replay samples. Validation selection averages foreground Dice over all seen tasks instead of selecting only the newest task.

The DER improvement also freezes the background and previous classifier heads after each task, and recomputes cached feature targets using the selected checkpoint when restoring/saving task boundaries. This deliberately changes historical-target DER into a boundary-refreshed variant; report it separately. Current training can still alter the shared backbone. Both improved variants use the existing safe saliency/GraphCut helpers and gradient clipping at 5. The guarded DER control retains its previous protections; ER control retains its earlier implementation.

These are bundled engineering variants: a performance change does not establish the causal contribution of each component. No test result is used to choose variants or checkpoints.

## Fixed budget and gates

- Run only on my-gpu GPU3, one trainer at a time, requiring 16000 MiB free before each launch. Never stop another process to make room.
- Reuse each method's completed 80-epoch T1 weights, reservoir and historical metrics. This is a stage-boundary restart with seed 42, not an exact RNG continuation.
- First run the small `check.py` assertions, then a real-data smoke check of two T2 optimizer updates per improved method, with validation.
- Run four paired screens: ER control/improved and guarded DER control/improved, each T2 for 5 epochs. Use the same T1 starting state, seed, data and 5-epoch LR schedule within each pair. Screens are validation-only. Screening models are not reused in formal continuation.
- Advance only if improved T1 validation retains at least 50% of its T1 reference, T2 validation is at least `max(0.15, 0.8 * paired control T2)`, and the two-task validation mean improves by at least 0.02 over control. These are prospectively chosen engineering gates, not paper significance criteria.
- Passing variants restart from the same T1 state for a full 60-epoch T2. Start 60-epoch T3 only if that T2-selected checkpoint still retains 50% of initial T1 validation and achieves T2 validation >= 0.15. Maximum formal schedule is `[80 reused, 60, 60]` for each of at most two improved variants.
- Keep seed 42, batch 4, SGD LR 0.03/momentum 0.9/weight decay 0.0001, PCE/Global/Spatial=1/0.1/0, buffer 64 and replay batch 4. Use workers=0 for all new controls and candidates to avoid the observed NFS multiprocessing cleanup failures. This worker/RNG change is a difference from the historical workers=4 runs, so screening uses new paired controls.
- The finite driver has no automatic retries, new sweeps, migration to another GPU or open-ended monitoring. A failed screen or gate stops that method's continuation. `progress.json` records per-job success/failure and gate decisions; `finished` means the finite queue ended, not that every variant passed or improved.

## Runtime and checks

Overlay `runner_core.py`, `entry.py` and `check.py` onto a copy of the prior Class replay source. Use the original ER mixup helper for ER control; use the previously published guarded helpers for guarded DER and improved variants. Put `job.py` in a neutral campaign `control` directory. A private `plan.json` supplies external data and sparse-annotation paths. `resume/<method>` contains the prior manifest, the first completed stage row, and links to T1 state/weights and prior training records.

All trainer arguments are passed in `JOB_ARGS`; process commands are neutral `job.py` / `entry.py`, with titles `run` / `job`. `job.py` reads its fixed plan and runs detached. The CPU mechanism check covers partial-background gradients, ignored/invalid labels, BN buffer invariance, old-head freezing, and near-zero feature error immediately after target refresh. The two real-data smoke checks additionally exercise checkpoint restore and optimizer updates.

Publication contains code, this protocol and aggregate launch evidence only. Datasets, per-patient metrics, private paths/configuration, raw logs, predictions, reservoir contents and model checkpoints remain on the server.
