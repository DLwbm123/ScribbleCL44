# Class independent T3 — completion

The 80-epoch run on 1500 training slices completed. Validation selected epoch 48. Held-out foreground Dice is 0.6746877956913865; **background-inclusive Dice is 0.7820229984174101**. The common T3 independent reference is now available for the benchmark RMA calculation (T1 excluded). Its selected stage checkpoint is present; the earlier live log inventory confirmed all 80 epoch rows. These are single-seed results, not mean/standard deviation across runs.

Aggregate result: results/class_independent_T3_completion_20260912/summary.json. The original implementation is in class_runtime/. Patient data, per-patient metrics, checkpoints and raw logs remain private.
