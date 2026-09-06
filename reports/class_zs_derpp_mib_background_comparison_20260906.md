# Class-CL ZS-DER++ + MiB background-inclusive Dice comparison

The completed seed-42 formal run `v6c43` used 80 epochs per task and completed
all three Class-CL stages. Its final `s03.pt` checkpoint was evaluated once;
the same predictions produced both Dice definitions.

| Test task | Foreground only | Background Dice | Background included | Difference |
|---|---:|---:|---:|---:|
| T1: MYO/LV/LA | 0.738224 | 0.970777 | 0.796362 | +0.058138 |
| T2: RA/RV | 0.695712 | 0.963495 | 0.784973 | +0.089261 |
| T3: AO/PA | 0.658655 | 0.952823 | 0.756711 | +0.098056 |
| **A-Dice** | **0.697531** | — | **0.779349** | **+0.081818** |

The foreground-only score exactly reproduces `summary.json`. The run was
selected using foreground-only validation Dice (`0.742233`); the inclusive
score is a paired retrospective re-evaluation, not retraining or reselection.
