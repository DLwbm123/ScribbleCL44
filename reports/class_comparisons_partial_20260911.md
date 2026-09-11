# Class-CL comparisons: three completed methods

Snapshot: **2026-09-11 17:14 Asia/Shanghai**. The eight-job campaign is still running. PCE, Dense and ZS sequential each completed all three tasks with 80 epochs and 30,000 updates per task, finite losses, exit code zero and nonempty selected checkpoints.

Metrics are **background-inclusive per-patient Dice**. Selection remains foreground validation Dice over all 80 epochs. RMA awaits the common independent T2/T3 reference jobs; unavailable values are not zero.

| Method | A-Dice | BWTR | WCD | MPE | DRR | RMA |
|---|---:|---:|---:|---:|---:|---|
| PCE-Sequential | 0.375486 | -0.595698 | 0.197122 | 0.000000 | 0.000000 | Pending |
| Dense-Sequential | 0.484412 | -0.678513 | 0.330855 | 0.000000 | 0.000000 | Pending |
| ZS-Sequential | 0.460977 | -0.641994 | 0.306319 | 0.000000 | 0.000000 | Pending |

All use full Class data (1,500 training slices per task), batch=4, seed=42, SGD LR=0.03, momentum=0.9, weight decay=0.0001 and Spatial=0. Global is 0 for PCE/Dense and 0.1 for ZS. Dense uses dense current-task masks; the others use fixed scribbles. Final sequential means include old-class forgetting and differ from acquisition Dice.

EWC and GPM were alive on T3, with 11/80 and 15/80 epochs completed. The main replay/MiB method and independent T2/T3 jobs remain queued. Estimated remaining time is 18-24 hours, conditional on shared-server throughput. Active methods need roughly 3-4 hours; the longest queued job is the 80-epoch main sequence. Its earlier 40-epoch-per-task counterpart took about eight hours. Independent references can run on the other GPU.

Neutral process names, a 20,000 MiB admission threshold on GPU2/3 and a short NAS temporary-socket path are used. Older interrupted artifacts remain preserved. The original NFS worker cleanup warnings did not prevent these three runs completing. This release uses the deployed source snapshot and leaves unrelated local work untouched.

[Aggregate metrics and matrices](../results/class_comparisons_partial_20260911/summary.json), [runner](../class_runtime/runner_core.py), [launcher](../class_runtime/run_formal_supplement.py). Private inputs, patient-level values, source IDs, raw logs and models are excluded.
