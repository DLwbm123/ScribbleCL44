# Completed Organ-CL T2 retention diagnostic: 10-epoch T3 continuation

Both fixed runs completed on **2026-09-10 at 11:11:10 Asia/Shanghai**, after launch at 10:29:15 (about 42 minutes elapsed). Both job exit codes, the coordinator exit code and the pipeline exit code are zero. This completes the two-run diagnostic, not a new four-task training sequence. No further training was automatically scheduled.

## Protocol

Both runs restore the same completed T2 model and paired replay checkpoint. T3 uses the same fixed half-data subset (711 training slices), seed 42, batch 4, 10 epochs and 178 updates per epoch. Initial backbone/head learning rates are 0.006/0.06, twice the previous 0.003/0.03 setting; polynomial decay reaches zero after this 10-epoch budget. The paired conditions use Spatial weights 0 and 0.01; positive Spatial starts at T3 epoch 1.

Other controls are unchanged: PCE/Global=1/1, feature replay alpha=0.05, supervised replay beta=0.5, buffer=128, replay batch=4, SGD momentum=0.9, weight decay=1e-4, clip=5, fixed backbone BN running statistics, trainable BN affine parameters, and frozen historical task heads. Test metrics are diagnostic observations only. Stage checkpoint selection maximizes **T3 validation Dice**, not T2 test retention or a test maximum.

All scores below are **foreground-only per-case macro Dice**, consistent with this diagnostic's historical T2 baseline. They are not the background-inclusive values used in the separate thesis table work.

## Epoch endpoints

The restored T2 baseline is identical for both runs: validation **0.6991301630**, test **0.6534525181**.

| Spatial | T2 test after epoch 1 | T2 test after epoch 10 | T2 retention at epoch 10 | T3 test after epoch 10 |
|---|---:|---:|---:|---:|
| 0 | 0.633883 | 0.574495 | 87.92% | 0.818900 |
| 0.01 | 0.602048 | 0.583271 | 89.26% | 0.822756 |

These are saved epoch endpoint models. In particular, the epoch-10 values are not the validation-selected stage results.

## Validation-selected stage models

| Spatial | Selected T3 epoch (1-based) | T3 validation | T1 test | T2 test | T3 test | Three-task test mean |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 0.800281 | 0.649437 | 0.562377 | 0.805165 | 0.672326 |
| 0.01 | 9 | 0.808630 | 0.624103 | 0.556685 | 0.808296 | 0.663028 |

## Interpretation

At the fixed epoch-10 endpoint, Spatial=0.01 gives T2 a **0.008776** Dice advantage and T3 a **0.003856** advantage. T2 nevertheless remains below its pre-T3 baseline: absolute drops are **0.078958** without Spatial and **0.070182** with Spatial.

The first-epoch result favors no Spatial for T2, and the validation-selected model also has slightly better T2 test Dice without Spatial (**0.562377 vs 0.556685**). Thus this pair does **not** establish that Spatial consistently reduces forgetting. Selection by current-task validation need not maximize historical-task retention. The higher LR and shorter budget changed together relative to prior long runs; their individual effects cannot be separated from this pair. This is a single-seed diagnostic, not evidence of statistically significant superiority.

## Completion evidence and public scope

- Exactly 10 T3 training epochs and 1,780 updates per condition; all recorded losses finite.
- Ten finite T2/T3 validation/test observations and ten nonempty endpoint checkpoints per condition.
- Selected `s03.pt`, `s03_best.pt` and paired `s03_state.pt` present and nonempty.
- Both independently re-evaluated restored T2 baselines match the source values within the required 1e-6 tolerance.
- The prelaunch CPU evaluation-state/RNG isolation check passed. No repeated training or extra test inference was needed for this completion audit.

Code: [launcher](../launch_organ_t3_lr10_pair.py), [shared training implementation](../runner_core.py), [evaluation isolation check](../test_organ_formal_resume.py). Results: [aggregate summary and both full curves](../results/organ_T3_lr006_spatial_pair10_20260910/summary.json), [epoch comparison CSV](../results/organ_T3_lr006_spatial_pair10_20260910/comparison.csv).

The launcher takes a private NAS run root containing `inputs/`, the read-only `prefix_T2/`, the source snapshot and `checks/cpu_passed.json`. Runtime data, original scribbles, patient-level outputs, full logs, caches and all checkpoints remain on the authorized private NAS and are excluded from publication. Endpoint `.pt` snapshots contain model weights only and must not be described as exact optimizer/replay resume states.
