# Class EWC T2 diagnostics — completed

All three T2-only diagnostics completed ten epochs / 3750 updates with finite losses and exit code zero. The 80-epoch LR horizon was preserved. The source T1 model and Fisher information were restored; only T1/T2 validation scores selected the diagnostic checkpoint. No new test-set evaluation was performed.

| EWC coefficient | Selected diagnostic epoch | T1 foreground validation Dice | T2 foreground validation Dice |
|---:|---:|---:|---:|
| 1 | 6 | approximately 0 | .500419 |
| 1000 | 6 | approximately 0 | .575434 |
| 10000 | 6 | approximately 0 | .576332 |

Increasing the coefficient improved T2 validation learning slightly but did not preserve T1. These are foreground validation diagnostics and must not be inserted as background-inclusive test results in a thesis table. No extended EWC run is scheduled from these results. Aggregate evidence is in results/class_ewc_t2_probe_20260911/completion.json; the implementation was already delivered with the diagnostic launch.
