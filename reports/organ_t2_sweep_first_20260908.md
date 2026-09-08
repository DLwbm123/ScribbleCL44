# Organ-CL: screen the recipe before the paired coverage experiment

The user requested parameter screening before committing to long continual runs. The two initial 80-epoch-per-task CL jobs were stopped during early T1, preserving all checkpoints and logs. The original independent pair continues to completion. Its validation Dice near 0.07 despite sparse training loss near 0.02 is a failed performance signal, not evidence that the recipe is ready for formal training.

A read-only probe of a saved baseline T2 checkpoint found approximately 42.4% predicted foreground versus 1.775% dense foreground on validation. Replacing head and/or backbone running BN statistics with batch statistics in diagnostic model copies did not improve Dice (approximately 0.061–0.070). This does not identify a unique training cause, but rules out claiming a simple inference BN switch solves this checkpoint's poor performance. Test labels were not used in this probe.

## Four predeclared configurations

| Run | Initial LR | Global weight | Assigned GPU |
| --- | ---: | ---: | ---: |
| sw_lr01_g0 | 0.01 | 0 | 6 |
| sw_lr03_g0 | 0.03 | 0 | 7 |
| sw_lr01_g01 | 0.01 | 0.1 | 4 |
| sw_lr03_g01 | 0.03 | 0.1 | 5 |

Each uses T1 to T3, 20 epochs per task, seed 42, the fixed 19.6609% T2 foreground annotation, batch 4, SGD momentum 0.9, weight decay 0.0001, polynomial LR exponent 0.9, finite gradient clipping at 1, PCE weight 1, Spatial/GD disabled, DER++ alpha/beta 0.5/0.5, buffer 128 and replay minibatch 4. Existing GPU jobs are retained. A worker waits for 18,000 MiB free on its assigned device; the two independent jobs are not interrupted to start the sweep.

For Global=0, both the current and replay global branches are skipped. A targeted regression check verifies that replay PCE still receives gradients and that the disabled augmentation/global function is never called. Simply multiplying a computed global loss by zero would incur unnecessary work and BN side effects, so the new source snapshot fixes that path. Already-running independent processes use their original source snapshot.

Only validation is read during this sweep. Each task selects its best current-task validation checkpoint; model and replay state are restored together. All seen tasks are evaluated after each stage. A candidate is eligible only if:

- T2 validation Dice after T2 is at least 0.30;
- T3 validation Dice after T3 is at least 0.50;
- T2 after T3 retains at least 80% of its post-T2 Dice;
- the absolute T2 drop is at most 0.10.

Eligible candidates are ranked by final seen-task mean validation Dice. These gates are prospective development criteria, not results. If none passes, no new 80-epoch CL pair is launched. If one passes, the same selected recipe starts fresh 80-epoch-per-task runs for both 19.6609% and 40.4353% T2 annotations on GPU 6/7. The formal pair may evaluate test splits after stage selection, but test metrics do not influence promotion or checkpoint selection. Short-run ranking need not persist at 80 epochs, so this process reduces rerun risk without eliminating it.

`run_t2_sweep.py` records job argv/start/exit events and writes `sweep_selection.json`. The isolated `sweep_source` runtime, all run artifacts, original stopped checkpoints and failure records remain on the authorized NAS. No medical arrays or private checkpoints are included in the source repository. This report describes a launched protocol, not completed sweep or formal results.
