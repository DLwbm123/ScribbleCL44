# Class replay improvement launch — 2026-09-15

Latest active campaign: `class_replay_improve_20260915_r2`, started 2026-09-15T14:46:57.686599+08:00. This is a launch receipt; screening and performance improvements are not yet established.

| Item | Verified launch state |
|---|---|
| Host / GPU | my-gpu / GPU3 |
| Free memory before launch | 24504 MiB; existing unrelated workload preserved |
| Detached supervisor | PID 1871877, title `run` |
| First corrected ER smoke trainer | PID 1871879, title `job`, GPU3 |
| Runtime code | `3af1e70628470d276f10456fae7c41c2bc0157dc` |
| Source branch | `organ-t2-coverage-20260908` |

The exact budget, variants and gates are in [the improvement protocol](../class_replay_improvement_runtime/README.md). The queue executes two short real-data checks, four paired 5-epoch T2 screens, then at most two validation-gated `[80 reused, 60, 60]` continuations. It runs one trainer at a time, requires 16000 MiB free before each new job, and never interrupts unrelated GPU work. Screening and variant/checkpoint selection use validation only.

## Engineering checks and correction

The CPU mechanism check passed, including partial-background gradients, ignored/invalid labels, frozen BatchNorm buffers and old heads, and near-zero feature mismatch after refreshing targets. The final check explicitly includes the actual int16 replay-label dtype.

The first new campaign exposed an ER implementation issue: `gather` requires int64 indices, while cached replay labels are int16. The shared loss now casts only the gather indices to int64. The assertion check was strengthened accordingly. That first attempt and its logs remain preserved. Its DER smoke completed two finite optimizer updates (second-step loss 0.748064, feature penalty 0.000947); this is a smoke result, not a performance claim. Its subsequently started DER control was interrupted when the newly created queue was stopped for the correction. Only that queue's verified process group was stopped; no pre-existing workload was touched. The corrected campaign starts both methods from the same preserved T1 states.

The actual NFS mount and capacity were checked (about 353 TB free), and a small write/read probe passed. Training inputs and T1 weights/reservoirs are reused via links. New code was copied into isolated runtime directories; prior experiment outputs were not overwritten. The corrected supervisor/trainer command lines and GPU assignment were checked. All new controls/candidates use workers=0 to avoid the observed NFS multiprocessing cleanup failures; this is a disclosed difference from historical workers=4 runs.

Private run artifacts are `progress.json` (per-job states and validation gates), `logs/` (logs and exit codes), `jobs/` (outputs), `plan.json` (private input locations and source commit), and `launch.json` (launch receipt). The finite queue has no open-ended monitoring or automatic retries. A queue status of `finished` means the prescribed queue ended, not that every gate passed.

Public delivery includes source, protocol and aggregate launch evidence. Patient/subject-level results, data, predictions, raw logs, checkpoints, replay contents and credentials remain private. Long-run stability and actual retention/acquisition improvements must be assessed after screening; no successful performance result is claimed here.
