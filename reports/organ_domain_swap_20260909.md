# Organ T2 Prostate domain swap: UCL versus BIDMC

**Both bounded runs completed successfully and stopped after the requested single T3 epoch.** Verified on 2026-09-09 at approximately 12:43 China time. UCL finished at 09:55:37 (29 min 05 s); BIDMC finished at 10:09:51 (43 min 19 s). Both T2 and T3 processes exited with code 0. Epoch logs, stage summaries, finite aggregate metrics, paired checkpoint presence/sizes, and completion markers were reconciled. No experiment was restarted.

## Completed comparison

The table reports **test Dice at validation-selected checkpoints**. Each T2 ran 20 epochs, then its selected paired state started a standardized one-epoch T3 continuation. The initial shared T1 checkpoint has test Dice 0.664838.

| Metric | UCL | BIDMC |
|---|---:|---:|
| T2 selected epoch | 14 | 13 |
| T2 validation Dice after T2 | 0.699130 | 0.732039 |
| T2 test Dice after T2 | 0.653453 | 0.696615 |
| T1 test Dice after T2 | 0.641884 | 0.373053 |
| T2 test Dice after one T3 epoch | 0.336907 | approximately 0 |
| T2 absolute test Dice drop | 0.316545 | 0.696615 |
| T2 retained fraction of test Dice | 51.56% | approximately 0% |
| T1 test Dice after one T3 epoch | 0.369228 | 0.316691 |
| T3 test Dice after one T3 epoch | approximately 0 | approximately 0 |

T2 validation after one T3 epoch was 0.244964 for UCL and approximately zero for BIDMC. BIDMC's T2 foreground-prediction fraction was exactly zero on both validation and test; epsilon in Dice accounts for the tiny positive values in the raw scalar export. Both runs' T3 foreground-prediction fractions were also zero at this early endpoint. These finite all-background predictions are distinct from the numerical overflows seen in earlier long runs.

**Changing T2 to BIDMC did not fix forgetting in this comparison.** BIDMC acquired its own task somewhat better, but damaged T1 much more during T2 and lost its own foreground prediction after one T3 epoch. This does not establish that BIDMC is universally worse: cohort size, updates per epoch, actual annotation coverage, and evaluation patients differ, and only one seed was tested. It also does not establish final T3 performance from a single epoch.

The selected T2 buffers contained 76 T1 / 52 T2 samples for UCL and 71 T1 / 57 T2 for BIDMC. These counts alone do not explain the much larger BIDMC T1 loss. T3 shared weights, BN statistics, and replay/clipping settings still require separate causal checks. The next useful target is the T3 transition rather than assuming another domain will solve it; no additional run was launched in this status check.

The UCL first-T3-epoch test score here (0.336907) differs from the previous continuous formal run (0.320431). This check starts a separate T3 process with explicitly reseeded transition seed 44; the old continuous run carried its post-training RNG trajectory into T3. Do not claim bitwise or trajectory equivalence between them.

The inherited retention hook writes `training_continues=true` even in single-epoch mode. This is misleading metadata, **not an active process**: both phase exit codes, exactly one T3 epoch, and `complete.json` establish that the requested stop occurred. The original files are preserved; the scalar export records `actually_stopped_after_t3_epoch1=true` and explains this discrepancy.

Public scalar evidence: `results/organ_domain_swap_20260909/completion_scalar.json`. No per-patient metrics, images, labels, raw traces, or model tensors were exported.

## Data and annotations

| Quantity | UCL (Domain D) | BIDMC (Domain A) |
|---|---:|---:|
| Training slices / patients | 166 / 7 | 301 / 7 |
| Validation slices / patients | 52 / 2 | 94 / 2 |
| Test slices / patients | 100 / 4 | 126 / 3 |
| Foreground scribble pixels | 28,887 | 89,136 |
| Background scribble pixels | 1,115,520 | 2,022,720 |
| Integer dense training foreground pixels | 146,926 | 515,139 |
| Foreground coverage | 19.6609% | 17.3033% |
| Foreground fraction of known labels | 2.5242% | 4.2207% |
| T2 updates per epoch / total | 42 / 840 | 76 / 1,520 |

The same recovered `v2_S2_area_scaled`, WSL-style rule is applied to both datasets: seed 42, baseline width-3 skeleton pixel budgets, foreground area multiplier 3 and background multiplier 20. Foreground is an eroded/branch-cut/perturbed skeleton grown within its class; background is a background skeleton grown to its specified budget. The rule does not force equal foreground coverage between domains. Dense labels are accessed only on the training split to synthesize annotations; conversion to integers matches the training/evaluation loader.

The recipe was recovered from the project's thesis experiment snapshot (`experiments/medical_continual_segmentation/medcl/data/sparse.py`), with unused helpers omitted and the ignore-index import made standalone as `organ_area_scribbles.py`. As a targeted protocol-recovery check, regenerated UCL labels matched the currently used annotation array exactly. This comparison validates this recovered recipe, rather than adding a routine file checksum. Both outputs passed foreground/background containment checks. The exact pixel-generation functions remain unchanged.

The current UCL foreground denominator is 146,926, yielding 19.6609% coverage. This corrects the 18.204% figure previously quoted in the balance report; annotation pixels and training inputs have not changed.

UCL uses the existing Organ data; BIDMC uses the existing Domain-A HDF5. All original train/validation/test splits and full T2 training sets are retained. T1 and T3 keep the existing half-training views. An isolated `Task_incre/UCL.h5` compatibility symlink points to the selected cohort in each run; in the BIDMC run it points to BIDMC and does not overwrite UCL. The run root, annotation audit, and annotation ID record the actual dataset.

This is an equal-epoch, same-recipe cohort comparison. It is **not** an equal-update, equal-foreground-coverage, or matched-patient experiment. BIDMC has more updates and a different cohort; conclusions must be limited accordingly.

## Training and stopping

1. Restore the same original completed T1 paired model/replay state, selected at T1 epoch 17 from its 60-epoch run. T1 training is not repeated.
2. Train T2 for **20 epochs**, retaining the original **60-epoch polynomial LR horizon**. Use alpha 0.05, beta 0.5, clip 5, and clean training-image calibration of the T2 head BN before validation. Select the best paired model/replay state on the corresponding T2 validation split.
3. Start a separate T3-only process from that selected paired T2 state, using the declared T3 transition seed 44. Execute **exactly one T3 epoch (178 updates)** with the original 60-epoch LR horizon. Record T2 validation/test Dice before/after, absolute drop, and retention ratio; then stop normally.

Shared settings: seed 42 / T2 transition seed 43, batch 4, workers 8, initial SGD LR 0.03, momentum 0.9, weight decay 1e-4, PCE/Global 1/1, replay buffer 128 and minibatch 4. The Spatial setting is 0.01 with first active epoch 30 of each task, so it never activates within these bounded budgets. Test scores are for reporting, never checkpoint selection.

T3 deliberately retains the same original alpha/beta 0.5/0.5, no clipping and no T2-head calibration during T3. This isolates the cohort change within the agreed continuation protocol; it does not simultaneously test a T3 stability fix. If numerical guards fail, that branch stops and its failure is retained. There is no automatic retry or formal-training promotion.

The new `--organ-t2-epochs` flag limits executed T2 epochs while leaving the LR horizon explicit. A native synthetic GPU integration check passed for one executed T2 epoch with a two-epoch horizon, followed by the normal T3 budget; paired restoration and unchanged T1 records passed as well. Existing single-epoch T3 resume is reused. No second training loop is introduced.

## Operation and reproduction

Root: `/data_nas/jiangsuiyang/ScribbleCL/organ_domain_swap_20260909`.
- `ucl/` and `bidmc/`: isolated data/annotation views, `t2/` and `t3/` outputs.
- Per variant: `controller.log`, `t2.log`, `t3.log`, `execution.json`, phase exit codes, and `complete.json` on successful completion.
- `t3/t3_epoch1_t2_retention.json`: requested immediate-forgetting result.
- `annotation_audit.json` and `launch.json`: actual sources, aggregate budgets and launch commands.

`prepare_organ_domain_swap.py --base-root <original-run> --bidmc <BIDMC.h5> --output <new-root>` builds the views and verifies recipe recovery. Copy the shared runtime source (including this revision's `runner_core.py`) into `<new-root>/source`, then run `run_organ_domain_swap.py --root <new-root> --base-root <original-run> --domain ucl --gpu 4` and the corresponding BIDMC/GPU5 command in detached sessions. The driver derives unchanged parameters from the original selected-run launch record.

NAS capacity and write/read probes passed; the launcher checks GPU free memory before starting. Runs survive SSH/Codex closure. Estimated wall time is 25–35 minutes for UCL and 35–50 minutes for BIDMC, including T3 and checkpoint I/O; these are estimates, not measured completion times. No recurring monitor is configured. Source, protocol and aggregate annotation/startup records are public; images, labels, patient identifiers, checkpoints and raw logs stay on the server. The completed scalar outcomes are now published after the completion check.
