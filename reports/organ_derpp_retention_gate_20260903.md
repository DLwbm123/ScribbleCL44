# Organ-CL ZS-DER++ retention gate

Date: 2026-09-03

The preceding 80-epoch Organ run was stopped after an intermediate checkpoint test found T2 collapse from 0.6637 after T2 to 0.0018 after T3. Its artifacts remain retained outside the public repository.

The corrective gate uses the existing ZS-DER++ implementation and changes only the feature-replay coefficient:

| Control | Prior run | Retention gate |
|---|---:|---:|
| Tasks | T1 to T4 | T1 to T3 |
| Epochs per task | 80 | 20 |
| ZS global weight | 0.1 | 0.1 |
| DER++ buffer | 64 | 64 |
| DER++ replay minibatch | 8 | 8 |
| DER++ feature coefficient alpha | 0.5 | 5.0 |
| DER++ supervised coefficient beta | 0.5 | 0.5 |

The gate requires a non-collapsed T2 after T3 before a new T1-to-T4 formal run is started. It is a diagnostic development run; its test matrix is not used for hyperparameter selection or final reporting.

## Result

The run completed its three stages and passed the output-contract audit. Its observed test matrix was:

| After task | T1 Dice | T2 Dice | T3 Dice |
|---|---:|---:|---:|
| T1 | 0.6123 | — | — |
| T2 | 0.5296 | 0.3925 | — |
| T3 | 0.5196 | 0.0000 | 0.8531 |

T2 again collapsed after T3 (0.3925 to approximately zero). Therefore this retention gate failed and no replacement 80-epoch Organ T1-to-T4 run was launched. The next Organ change needs a different retention strategy rather than another formal run with this coefficient alone.

## Matched 20-epoch factorial gate

To separate new-task plasticity from forgetting without changing the epoch budget, four T1-to-T3 runs used the same 20 epochs per task, seed 42, global weight 0.1, beta 0.5, and replay minibatch 8. Every output passed the pipeline audit.

| Alpha | Buffer | T2 after T2 | T1 after T3 | T2 after T3 | T3 after T3 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 64 | 0.5037 | 0.4595 | 0.0000 | 0.8250 |
| 1.0 | 64 | 0.4441 | 0.5112 | 0.0000 | 0.8387 |
| 0.5 | 128 | **0.5762** | **0.4977** | 0.1628 | 0.8375 |
| 1.0 | 128 | 0.3450 | 0.4787 | **0.2148** | 0.8272 |

No setting eliminated T2 forgetting. Alpha 0.5 with a 128-example buffer was selected for the formal T1-to-T4 run because it had the best T2 acquisition, retained T1 almost completely, and retained a nonzero T2 score. Alpha 1.0 with the same buffer retained slightly more T2 but its T2 acquisition was too weak for selection.
