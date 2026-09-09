# Organ-CL matched formal continuation controls

Verified 2026-09-09 at approximately 09:12 China time: **all four processes have exited. One run completed; three failed numerically during T3.** These are within-method ablations / coefficient comparisons, not cross-method EWC/GPM benchmark results. Exit markers, 60-epoch T2 logs, stage records, checkpoint sizes, and completion/failure records were reconciled. GPU 4–7 were free at this check. No run was restarted.

## Results and failures

All rows used validation for checkpoint selection; the following are **test Dice**, not selection scores. Every run completed 60 T2 epochs and restored its paired checkpoint selected on T2 validation before entering T3.

| GPU / T2 alpha | Selected T2 epoch | T1 after T2 | T2 after T2 | T2 after first T3 epoch | Run outcome |
|---|---:|---:|---:|---:|---|
| 7 / 0.05 | 14 | 0.641884 | 0.653453 | 0.320431 | Completed T2/T3 60 epochs each |
| 4 / 0 | 25 | 0.607163 | 0.563126 | approximately 0 | Failed during T3 epoch 11, update 1816 |
| 5 / 0.1 | 27 | 0.643020 | 0.686023 | approximately 0 | Failed during T3 epoch 4, update 634 |
| 6 / 0.05, no Spatial | 14 | 0.641884 | 0.653453 | 0.320431 | Failed during T3 epoch 39, update 6777 |

The selected T2 validation Dice are respectively 0.699130, 0.685071, 0.665281, and 0.699130. Alpha 0.1 has a higher T2 test score than 0.05, but was not selected using that test score. This is a single-seed comparison; it does not establish a general coefficient ranking.

The completed GPU 7 run ended at **03:26:33 on September 9**, approximately 4 h 35 min after launch. Its released T3 model is the **epoch-43 checkpoint selected by current-task T3 validation**, after executing all 60 T3 epochs.

| Stage of completed alpha-0.05 run | T1 test Dice | T2 test Dice | T3 test Dice |
|---|---:|---:|---:|
| After T1 | 0.664838 | — | — |
| After T2 | 0.641884 | 0.653453 | — |
| After T3, selected epoch 43 | 0.650466 | 0.078540 | 0.864890 |

The completed run's final mean test Dice is 0.531299. T2 drops 0.333021 after just one T3 epoch (49.0% of its previous test Dice retained), then 0.574913 at the selected final T3 model (12.0% retained). Validation shows the same problem: T2 0.699130 before T3, 0.119932 after one T3 epoch, and 0.019372 at final selection. Thus T2 acquisition improved, while T2 retention during T3 remains severely deficient. T1 and T3 final test scores do not show the same collapse.

All three failures report `buffer_capture/feature_targets: non-finite value`. They completed 10, 3, and 38 full T3 epochs before failing in the following epoch. Their partial T3 checkpoints are preserved, but no completed final result is assigned to them. Failure exit times were 00:45:02, 00:21:24, and 02:08:57 respectively. This is a numerical failure, not a remaining-time estimate or evidence of an OOM.

## Interpretation boundaries

- T2 was trained with its special alpha, clip 5, and head calibration. **T3 restored alpha/beta 0.5/0.5, no clipping, and no head recalibration.** The early T2 forgetting and failures make the T3 transition the next diagnostic target; this run does not isolate the contribution of each reverted setting.
- The Spatial-off run and the reference selected T2 epoch 14, before Spatial starts at epoch 30; their T2 metrics and first-T3-epoch metrics are identical in the recorded results. Disabling Spatial did not prevent the early T2 drop. The two later T3 outcomes do not, by themselves, prove that Spatial prevents numerical failures.
- Replay is a shared reservoir, not a guaranteed per-task quota. In the completed run, the selected T2 buffer contained 76 T1 / 52 T2 samples; the selected T3 buffer contained 15 T1 / 14 T2 / 99 T3. This is a candidate limitation for retention, not a causal ablation result.
- Checkpoint selection scores only the current task's validation set. The final T3 checkpoint is not chosen to optimize T2 retention or the mean of all seen tasks. Severe T2 loss is already present at the first T3 epoch, so selection alone cannot explain its onset.
- Shared-weight drift, shared-BN drift, sparse replay coverage, and T3 coefficient/clipping changes are not yet separately quantified. No additional training or calibration experiment was started during this status check.

Scalar evidence, including all validation/test stage scores, first-epoch retention, failure locations and checkpoint sizes: `results/organ_formal_controls_20260908/completion_scalar_20260909.json`. Private model/data files and per-patient metrics were not exported.

## Original launch protocol

| GPU | Run | T2 feature alpha | T2/T3 Spatial | Purpose |
|---|---|---:|---|---|
| 4 | `alpha0` | 0 | 0.01 from epoch 30 | Full-run feature replay ablation |
| 5 | `alpha01` | 0.1 | 0.01 from epoch 30 | Stronger small-alpha alternative |
| 6 | `no_spatial` | 0.05 | Disabled | Spatial ablation after the shared T1 starting point |
| 7 | Existing `formal_small_alpha_20260908` | 0.05 | 0.01 from epoch 30 | Selected reference; left running |

All use the same completed T1 validation-selected paired model/replay state (`run60/s01_state.pt`), with T1's training and selected parameters untouched. T1 was trained for 60 epochs, selected at epoch 17. T2 begins afresh with declared transition seed 43; it does not resume a 10-epoch sweep checkpoint. Each continuation trains T2 and T3 for 60 epochs. Data counts remain T1 381 (half), T2 166 (full), T3 711 (half).

Shared controls: native Organ ZS-DER++ loop, seed 42, batch 4, workers 8, task-initial LR 0.03, SGD momentum 0.9 / weight decay 1e-4, original PCE/Global weights 1/1, replay buffer 128 and minibatch 4. T2 uses supervised replay beta 0.5, gradient clip 5, and training-image-only T2-head BN calibration before checkpoint-selection validation. Alpha-zero disables the weighted feature loss, while replay forwards, sampling, and buffer updates remain active; it is not a complete no-replay baseline.

T3 restores the original alpha/beta 0.5/0.5 and original clipping setting. Its first-epoch T2 validation/test retention is recorded, then training continues. Numerical debug guards remain enabled. The T3 feature-replay configuration has not yet been established as numerically stable in a full run. Checkpoint selection uses validation; test metrics are diagnostic only.

The Spatial ablation affects **T2 and T3**, not the already-trained T1. Until T2 epoch 30 its configured active losses match the GPU 7 reference. It is an ablation of Spatial in this continuation, not a claim that T1 was trained without Spatial. CUDA grid-sampling backward is not fully deterministic, so exact numerical equality between repeated runs is not assumed.

The launcher derives every command from the actual GPU 7 launch record, changing only output/annotation tag and the named alpha/Spatial controls. Training code remains `65ab7ef8783a687ac9787f5419563f5491422b4e`, using the same saved source directory as GPU 7. No training loop was duplicated or edited for these controls. Script: `launch_organ_formal_controls.py`; `--dry-run` checks and prints the exact derived commands without launching.

Remote root: `/data_nas/jiangsuiyang/ScribbleCL/organ_T13_half_cl_20260908/formal_controls_20260908/`.
- Outputs: `alpha0/`, `alpha01/`, `no_spatial/`.
- Logs: `logs/alpha0.log`, `logs/alpha01.log`, `logs/no_spatial.log`.
- Exact commands, GPUs, wrapper PIDs and source: `launch.json`.
- Each detached shell writes its own `<run>.exitcode` on exit; runs do not depend on the SSH or Codex session staying open.

Preflight found approximately 24 GiB free on each selected GPU and 30 TiB available on the NAS filesystem. The launcher required at least 16,000 MiB free per selected GPU, 60 GiB free storage overall, and a successful small write/read probe. It does not stop or change existing processes. Public artifacts contain source, configuration and startup evidence only; patient data, checkpoint files, private numerical snapshots and large raw logs remain on NAS. Completed scalar results and failure summaries are now published alongside the original launch configuration. Private artifacts remain on NAS.
