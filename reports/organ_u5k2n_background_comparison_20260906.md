# Organ-CL background-inclusive Dice comparison (2026-09-06)

## Scope and status

- Scenario: Organ-CL, method `zs-derpp`, seed 42.
- Evaluated checkpoint: run `u5k2n`, `s04_best.pt`, activated through T4.
- This is **not a completed formal run**. The run stopped during T4 at epoch 21 and has no final `s04.pt`, `s04_state.pt`, or `summary.json`; only the latest available best checkpoint was evaluated.
- The same predictions were used once to calculate both metrics. Dice uses smoothing `1e-5`, first averaged per patient/class and then across tasks.

## Results

| Task | Foreground-only Dice | Background Dice | Background-inclusive Dice | Inclusive - foreground |
|---|---:|---:|---:|---:|
| T1 | 0.5352 | 0.9919 | 0.7636 | +0.2283 |
| T2 | 0.0007 | 0.9936 | 0.4971 | +0.4965 |
| T3 | 0.8534 | 0.9878 | 0.9206 | +0.0672 |
| T4 | 0.7918 | 0.9921 | 0.8920 | +0.1002 |
| **Mean** | **0.5453** | **0.9914** | **0.7683** | **+0.2230** |

For Organ-CL each task has one foreground class, so the background-inclusive score is the arithmetic mean of background Dice and foreground Dice. The inclusive score is higher for every task because background Dice is close to 0.99. T2 demonstrates the masking risk most clearly: foreground Dice is approximately zero while the inclusive score is 0.4971.

## Interpretation boundary

These numbers are useful for auditing metric definitions, but they must not be reported as a successful or completed Organ-CL experiment. The checkpoint exhibits severe T2 forgetting, and the underlying T4 run did not complete.
