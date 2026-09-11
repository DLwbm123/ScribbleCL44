# Class-CL comparisons: five completed methods

Snapshot: **2026-09-11 22:24 Asia/Shanghai**. Six of eight jobs have completed: five comparison sequences and independent T2. PCE, Dense, ZS sequential, ZS-EWC and ZS-GPM each completed all three tasks with 80 epochs and 30,000 updates per task. Their selected checkpoints and aggregate results are available. EWC completed at 20:13 and GPM at 19:32 on September 11.

Metrics are **background-inclusive per-patient Dice**. Selection remains foreground validation Dice over all 80 epochs. Independent T2 is complete; RMA awaits the common independent T3 reference job. Unavailable values are not zero.

| Method | A-Dice | BWTR | WCD | MPE | DRR | RMA |
|---|---:|---:|---:|---:|---:|---|
| PCE-Sequential | 0.375486 | -0.595698 | 0.197122 | 0.000000 | 0.000000 | Pending |
| Dense-Sequential | 0.484412 | -0.678513 | 0.330855 | 0.000000 | 0.000000 | Pending |
| ZS-Sequential | 0.460977 | -0.641994 | 0.306319 | 0.000000 | 0.000000 | Pending |
| ZS-EWC | 0.446074 | -0.638961 | 0.290850 | 0.000000 | 0.000000 | Pending |
| ZS-GPM | 0.448550 | -0.632749 | 0.266062 | 0.000000 | 0.010667 | Pending |

All use full Class data (1,500 training slices per task), batch=4, seed=42, SGD LR=0.03, momentum=0.9, weight decay=0.0001 and Spatial=0. Global is 0 for PCE/Dense and 0.1 for the ZS methods. Dense uses dense current-task masks; the others use fixed scribbles. MPE measures model parameter expansion; EWC and GPM auxiliary state is not counted as additional model parameters.

The main replay/MiB sequence and independent T3 remain running. Independent T2 completed all 80 epochs at 22:09, selected epoch 38, and achieved validation foreground Dice 0.586710, test foreground Dice 0.672962, and test background-inclusive Dice 0.778618. The completed reference passed the 30,000-update, finite-loss and nonempty-checkpoint checks.

For the newly completed EWC and GPM runs, all 240 logged training losses were finite, all three stages reached 30,000 updates, and the final, selected and state checkpoints were nonempty. Their old-task foreground Dice is effectively zero after T3; background-inclusive results must not be interpreted as preserved old foreground classes. RMA remains pending, and these are partial campaign results.

The executed source was published in the preceding source-and-results release. This update adds only aggregate results and the current queue snapshot. Neutral process names, a 20,000 MiB admission threshold on GPU2/3 and a short NAS temporary-socket path remain in use. Older interrupted artifacts are preserved.

[Aggregate metrics and matrices](../results/class_comparisons_partial_20260911/summary.json), [runner](../class_runtime/runner_core.py), [launcher](../class_runtime/run_formal_supplement.py). Private inputs, patient-level values, source IDs, raw logs and models are excluded.

DRR clarification: the deployed metric implementation counts GPM representation construction as source-data reuse (16/1500 = 0.010667 per old task). The prior manually assembled partial table showed zero by considering only raw-image replay; this entry is corrected here. GPM still has no raw-image replay buffer.
