# Organ T2 Prostate domain swap: UCL versus BIDMC

Two bounded runs have been launched, UCL on GPU 4 and BIDMC on GPU 5. They test whether changing the Prostate cohort changes T2 acquisition and immediate forgetting during T3. Results are pending; these are not completed formal experiments.

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

NAS capacity and write/read probes passed; the launcher checks GPU free memory before starting. Runs survive SSH/Codex closure. Estimated wall time is 25–35 minutes for UCL and 35–50 minutes for BIDMC, including T3 and checkpoint I/O; these are estimates, not measured completion times. No recurring monitor is configured. Source, protocol and aggregate annotation/startup records are public; images, labels, patient identifiers, checkpoints and raw logs stay on the server. Result publication follows a subsequent completion check.
