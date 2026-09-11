# ER and feature-DER controls for Organ and Domain

Four fixed fresh runs, seed 42. These adapt benchmark ER and its feature-based DER to the existing weak-supervision training implementation. This is not a reproduction of standard logit-DER. No new hyperparameter sweep or performance threshold is used.

| Setting | Tasks | Epochs per task | Data |
|---|---:|---:|---|
| Organ | T1–T4 | 40 | T1/T3/T4 half; T2 full; 381/166/711/982 train slices |
| Domain | A–F | 80 | Existing full splits and sparse annotations |

Both use SGD LR .03, momentum .9, weight decay .0001, batch 4, replay batch 4, reservoir capacity 64, current-image PCE/Global/Spatial = 1/1/0.
ER uses replay PCE coefficient 1 with no feature or replay-global term.
Feature-DER uses feature MSE coefficient .5 with no replay PCE/global term.
The 64-sample buffer follows the present ScribbleCL control budget; the original benchmark used 32.

Organ reuses the validation-selected Sequential c1 transition policy: T2–T4 backbone LR = .1 times the new-head LR; shared backbone BN frozen from T2; T2 head BN calibrated from training images; gradient clip 5 from T2. ER/DER coefficients remain fixed across tasks.
Domain has the existing shared two-class head and no Organ-specific LR, calibration or freeze strategy.

Training reuses runner_core.py and DarkExperienceReplayPlus storage, not a second training loop. ER stores raw images, sparse labels and head/source identifiers with empty feature targets. DER captures shared U-Net features with BN statistics frozen. Both preserve the selected model and corresponding reservoir together; paired checkpoint restoration uses CPU storage to avoid a redundant CUDA buffer copy. Old method implementations remain available in the snapshot.

Selection uses the current-task **foreground validation Dice** after every epoch. Held-out test data are evaluated only after selection. Each result also records per-case background-inclusive Dice and the corresponding complete stage matrix. Domain records same-seed untrained scores for forward-transfer metrics. Parameter counts and unique historical replay-source counts are retained. RMA requires the declared independent reference and is not fabricated from an unavailable denominator.

These are additional 40-epoch Organ and 80-epoch Domain controls. They do not erase earlier budgets, annotation settings or exploratory test-selected results. No claim of budget-matched superiority over the inherited 60/60/10/10 Organ model is intended.

## Execution and checks

Place this source snapshot in the campaign's source directory, with scenario inputs at inputs/organ and inputs/domain. The existing Python interpreter is linked as source/main. All runtime arguments travel in RUN_JOB; the visible command is ./main -u run.py. The scheduler admits GPUs 3–7 by available memory and allows sharing. Cache, temporary files, logs and checkpoints stay on the designated NAS.

check.py runs bounded real-slice cross-task checks (four Organ tasks and two Domain tasks for each loss) and asserts finite training, active/disabled loss branches, selected checkpoint restoration, historical source accounting, and background-inclusive arithmetic. It emits checks/passed.json with measured CUDA peaks and an admission threshold including 1500 MiB overhead/headroom. Check scores are not scientific results.

The production coordinator is detached from SSH. It writes plan.json, progress.json, per-job logs and exit codes, and comparison.json after all four jobs finish. No unattended experiment expansion or periodic Codex monitoring is configured.
