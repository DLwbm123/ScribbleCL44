# GPT Pro review prompt: Organ-CL ZS-DER++

Please act as an independent, skeptical code auditor. Review the **attached source-only archive** from the server runtime, not merely a similarly named public repository. The archive contains the active Python/Markdown source from `/home/jiangsuiyang/q4n6x`, excluding data, checkpoints, logs, cache directories, and archived checks. This runtime directory has no Git metadata, so first check whether it matches the public reference below before relying on the public branch.

Public experiment report: <https://github.com/DLwbm123/ScribbleCL44/blob/66cb12bfdb8e7fb596994fceff73d5dc33ede6f5/reports/organ_derpp_retention_gate_20260903.md>

## Scope and expected execution route

The Organ-CL runner must use `main.py --setting-run`, which dispatches through `runner.py` to `runner_core.py` with scenario `organ`. The expected model is the ZScribbleSeg U-Net and the method is `zs-derpp`:

- Loss for the current batch: sparse PCE plus `0.1 * L_global`.
- Replay: DER++ feature consistency (`alpha * feature-MSE`) plus `beta * (replay sparse PCE + 0.1 * replay L_global)`.
- Seed 42, batch size 4, learning rate 0.03, workers 8, validation interval 200.
- Organ tasks: T1=UtahI, T2=UCL, T3=Lits, T4=brain. All are binary foreground tasks, but each task may use a task-specific prediction head.
- Sparse annotations use ignore index `-100`; PCE must only supervise labelled pixels. The buffer is allowed for this DER++ experiment and stores 64 or 128 examples.

## Observed evidence

All results below are test Dice from the performance matrix, not validation Dice. Four T1-to-T3 selection gates used 20 epochs per task, global weight 0.1, beta 0.5, replay minibatch 8, and passed the output audit:

| Alpha | Buffer | T2 after T2 | T1 after T3 | T2 after T3 | T3 after T3 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 64 | 0.5037 | 0.4595 | 0.0000 | 0.8250 |
| 1.0 | 64 | 0.4441 | 0.5112 | 0.0000 | 0.8387 |
| 0.5 | 128 | 0.5762 | 0.4977 | 0.1628 | 0.8375 |
| 1.0 | 128 | 0.3450 | 0.4787 | 0.2148 | 0.8272 |

An earlier 20-epoch gate with alpha 5.0, buffer 64 produced T2=0.3925 after T2 and approximately zero after T3. A prior 80-epoch run under a different configuration reached T2=0.6637 immediately after T2 but then also collapsed after T3. Because T2 retention remained only 0.1628 in the best-plasticity candidate, the subsequent formal T1-to-T4 run was stopped before producing a usable stage result. Do not treat it as evidence.

One potentially confusing diagnostic: training logs contain a nonzero `zs_gd_loss` value even though `--zs-gd-loss` was not passed. For observed steps, the printed total loss numerically matches the configured PCE/global/DER++ terms without this value. Verify whether it is log-only or accidentally contributes to gradients.

## Required audit

Trace the actual data flow and give file-and-line evidence for every finding. In particular, inspect:

1. Dispatch and task topology: `main.py`, `runner.py`, `runner_core.py`; ensure Organ tasks, head creation, task IDs, optimizer reset/continuation, checkpoints, and stage transitions are correct.
2. Sparse labels: archive loading, slice ordering, augmentation alignment, ignore-index handling, background semantics, PCE mask/reduction, and label-to-head mapping for every task.
3. Evaluation: verify each matrix entry evaluates the correct task dataset with the correct head and Dice convention; distinguish validation selection from held-out test evaluation and rule out accidental evaluation with the latest head for all tasks.
4. DER++ buffer: confirm sampled replay examples retain their source task ID, sparse labels, and feature target; verify no stale/mismatched target, tensor aliasing, train/eval-mode inconsistency, or label/head mismatch is possible. Check reservoir replacement and whether the buffer is balanced across T1/T2 after T3.
5. Loss implementation: derive the exact scalar objective from code and compare it with the intended formula above. Check coefficient placement, loss normalization, replay/current batch weighting, and whether `L_global` or any GD/adversarial term is unintentionally applied or omitted.
6. State restoration: verify training uses the intended previous model/head state across tasks, while evaluation loads the appropriate in-memory or checkpoint state without unintended reinitialization.

## Deliverable

Return a concise audit with:

1. A verdict: **implementation bug found**, **implementation appears correct but the method/recipe fails**, or **inconclusive**.
2. Ranked findings with severity, exact file:line evidence, and an explanation of how each could produce the observed T2 collapse.
3. The smallest decisive test for each high-severity suspicion. Prefer a short deterministic smoke test or one-stage assertion; do not propose a large new method or a long sweep before checking code invariants.
4. A minimal patch only when an actual root-cause bug is demonstrated. Do not add replay variants, data caches, MiB, or unrelated architectural changes.
5. A statement about whether the existing matrices are trustworthy enough to report as a failed method result.
