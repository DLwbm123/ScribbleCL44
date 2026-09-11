# Completed Organ-CL T4-only continuation

Both conditions completed on **2026-09-10 at 22:49:38 Asia/Shanghai**, about 58 minutes after launch. Both job, coordinator and pipeline exit codes are zero. Each executed exactly ten T4 epochs and 2,460 updates; all recorded losses were finite and selected model/paired replay checkpoints were present.

These are **foreground per-patient macro Dice**, not background-inclusive thesis metrics. T4 validation Dice selects the checkpoint, followed by testing on all four tasks. Test values do not select checkpoints.

| Spatial | Selected T3 epoch | Selected T4 epoch | T1 test | T2 test | T3 test | T4 test | Four-task mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 6 | 0.634262 | 0.572855 | 0.792189 | 0.819519 | 0.704706 |
| 0.01 | 9 | 4 | 0.576130 | 0.595040 | 0.766780 | 0.819121 | 0.689268 |

Restored T2 test scores before T4 were 0.562377 and 0.556685; independent re-evaluation matched the source records within 1e-6. After T4, T2 scored 0.572855 and 0.595040. Spatial=0.01 gives higher final T2 Dice here, while Spatial=0 gives the higher four-task mean. This single-seed pair does not establish consistent superiority.

T1/T2 retain their 60-epoch run budgets; T3 models were selected from ten-epoch runs. Only T4 is newly trained for ten epochs. T1/T3/T4 use their fixed half subsets (381/711/982 slices), and T2 uses all 166 training slices; validation/test splits are unchanged. Selected T1/T2 epochs are 17/14, distinct from executed budgets.

T4 starts a new polynomial schedule at backbone/head LR 0.006/0.06. PCE/Global=1/1; Spatial=0 or 0.01 from epoch 1; feature alpha=0.05; replay beta=0.5; buffer=128; replay batch=4; training batch=4; seed=42, transition seed=45. SGD momentum=0.9, weight decay=0.0001, gradient clip=5. Backbone BN running statistics and old heads remain fixed; BN affine parameters remain trainable.

The shared runner's t4-from restore path loads the completed T3 paired model/replay state. Direct restoration now constructs all historical Organ heads before strict loading. The CPU regression/evaluation-isolation check passed, as did restored baselines and full GPU training. The paired launcher uses epochs=10, max-task=4 and t4-only, with input links and paired T3 sources prepared under its run root.

[Aggregate results and matrices](../results/organ_T4_from_T3best_pair10_20260911/summary.json), [runner](../runner_core.py), [launcher](../launch_organ_t3_lr10_pair.py), [restore check](../test_organ_formal_resume.py).

This release includes the executed source snapshot and aggregate results. Private data, scribbles, patient-level outputs, logs, caches and checkpoints remain on approved NAS storage.
