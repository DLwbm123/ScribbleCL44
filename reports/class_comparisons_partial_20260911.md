# Class-CL comparisons: five completed methods

Snapshot: **2026-09-11 21:22 Asia/Shanghai**. Five of eight jobs have completed. PCE, Dense, ZS sequential, ZS-EWC and ZS-GPM each completed all three tasks with 80 epochs and 30,000 updates per task. Their selected checkpoints and aggregate results are available. EWC completed at 20:13 and GPM at 19:32 on September 11.

Metrics are **background-inclusive per-patient Dice**. Selection remains foreground validation Dice over all 80 epochs. RMA awaits the common independent T2/T3 reference jobs; unavailable values are not zero.

| Method | A-Dice | BWTR | WCD | MPE | DRR | RMA |
|---|---:|---:|---:|---:|---:|---|
| PCE-Sequential | 0.375486 | -0.595698 | 0.197122 | 0.000000 | 0.000000 | Pending |
| Dense-Sequential | 0.484412 | -0.678513 | 0.330855 | 0.000000 | 0.000000 | Pending |
| ZS-Sequential | 0.460977 | -0.641994 | 0.306319 | 0.000000 | 0.000000 | Pending |
| ZS-EWC | 0.446074 | -0.638961 | 0.290850 | 0.000000 | 0.000000 | Pending |
| ZS-GPM | 0.448550 | -0.632749 | 0.266062 | 0.000000 | 0.000000 | Pending |

All use full Class data (1,500 training slices per task), batch=4, seed=42, SGD LR=0.03, momentum=0.9, weight decay=0.0001 and Spatial=0. Global is 0 for PCE/Dense and 0.1 for the ZS methods. Dense uses dense current-task masks; the others use fixed scribbles. MPE measures model parameter expansion; EWC and GPM auxiliary state is not counted as additional model parameters.

At the snapshot, the main replay/MiB method was running on GPU3 with T1 27/80 epochs complete. Independent T2 was running on GPU2 with 48/80 epochs complete; independent T3 remained queued. Both active processes were alive with neutral command line job. Estimated remaining campaign time is 14-20 hours, conditional on shared-server throughput. Independent T2 needs approximately 50-60 minutes, followed by independent T3 for approximately two hours. The main three-task sequence determines the full completion time.

For the newly completed EWC and GPM runs, all 240 logged training losses were finite, all three stages reached 30,000 updates, and the final, selected and state checkpoints were nonempty. Their old-task foreground Dice is effectively zero after T3; background-inclusive results must not be interpreted as preserved old foreground classes. RMA remains pending, and these are partial campaign results.

The executed source was published in the preceding source-and-results release. This update adds only aggregate results and the current queue snapshot. Neutral process names, a 20,000 MiB admission threshold on GPU2/3 and a short NAS temporary-socket path remain in use. Older interrupted artifacts are preserved.

[Aggregate metrics and matrices](../results/class_comparisons_partial_20260911/summary.json), [runner](../class_runtime/runner_core.py), [launcher](../class_runtime/run_formal_supplement.py). Private inputs, patient-level values, source IDs, raw logs and models are excluded.
