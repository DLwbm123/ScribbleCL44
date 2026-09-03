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
