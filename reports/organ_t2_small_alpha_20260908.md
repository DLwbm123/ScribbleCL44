# Organ T2 small feature-replay coefficients — running protocol

Status at publication: three bounded checks are running; no coefficient has been selected and formal training has not yet started. A detached controller will apply the declared gate and start one formal run only if a candidate passes. This document does not claim completed experiment results.

## Short checks

Three candidates use feature MSE alpha 0.01, 0.05, 0.1 on GPUs 4, 5, 6. Each restores the same T1 validation-selected paired model/replay state, then runs 420 updates / 10 T2 epochs with transition seed 43, LR 0.03 on the original 60-epoch decay horizon, gradient clip 5, supervised replay beta 0.5, PCE/Global 1/1, original sparse supervision, batch 4, replay minibatch 4, buffer 128. Spatial remains off during these ten epochs; first active epoch in the formal protocol remains 30. Replay forwards and buffer updates are retained for every coefficient.

Each final model is evaluated on validation data before and after a fixed train-image-only BN probe. The promotion criterion uses only the **T2-head-only** calibration variant. Backbone/both calibration variants are diagnostic and excluded from selection. Test data is not used in the sweep or gate.

Gate declared before results: all 420 updates and scalar losses/gradient norms must be finite; after T2-head calibration, T1 validation Dice must be at least 0.65 and T2 at least 0.33. Rank passing candidates by T2 Dice, then T1. These exploratory thresholds allow at most about 0.0222 lower T2 Dice than the prior alpha-zero calibrated reference (0.3522), while requiring better T1 retention than its 0.6270. Passing is evidence for trying a full run, not proof that forgetting is solved.

## Conditional formal run

- Reuse the previously completed T1 run's selected model **and paired replay state**, without retraining T1 or overwriting its files. T1 was trained for 60 epochs and selected at epoch 17.
- Fresh T2 training from that T1 state for 60 epochs; do not continue a diagnostic weights-only checkpoint.
- T2: selected alpha, beta 0.5, clip 5, T2-head BN calibration before checkpoint-selection validation.
- T3: 60 epochs with original alpha/beta 0.5/0.5 and original clipping setting; emit the first-epoch T2 retention evaluation and continue. The original T3 feature-replay setting remains an unproven numerical risk.
- Data: T1 train 381 (half), T2 166 (full), T3 711 (half). Batch 4, workers 8, LR 0.03 reset per task, SGD momentum 0.9 / decay 1e-4, Spatial 0.01 starting at epoch 30.
- Select checkpoints on validation. Test metrics are written only for diagnostic reporting after selection / the authorized T3 first-epoch check.
- Repeated head calibration during formal training can change its trajectory relative to the post-hoc sweep probe; do not equate the two results.

The shared runner exposes `--organ-t2-feature-alpha` under `--organ-t2-supervision-strategy`, and `--t2-from` for explicit previous-task paired restoration. Task-specific alpha does not change T1/T3 coefficients. A declared transition seed is used because historical checkpoint RNG state is unavailable.

## Checks and operational records

The native Organ-model CPU check passed for head-only running-buffer changes and policy isolation. Synthetic-data native GPU integration passed T1/T2/T3 training and paired selection, T3-only resume, and T2 resume with alpha 0.05; the T1 evaluation record was preserved. The controller self-check passed threshold/ranking and failed/incomplete/nonfinite rejection cases.

Code: `runner_core.py`, `test_organ_task_strategy.py`, `run_t2_small_alpha_formal.py`. Prior diagnostic tools are reused unchanged. The controller is an experiment dependency process, not a recurring automation; it does not require the Codex or SSH session to remain open.

Remote root: `/data_nas/jiangsuiyang/ScribbleCL/organ_T13_half_cl_20260908`.
- Sweep/controller: `small_alpha_checks_20260908/`.
- Status: `controller_status.json`; selection: `selection.json` (written after checks finish).
- Candidate logs: `alpha_0.01.log`, `alpha_0.05.log`, `alpha_0.1.log`.
- Conditional formal log: `formal.log`; exact command/PID/GPU: `formal_launch.json`.
- Conditional formal outputs: `formal_small_alpha_20260908/` under the remote root.

Mount/free space and write/read probes passed before launch. The controller checks free space and GPU memory again before formal launch, allows sharing when at least 16,000 MiB is free, and does not modify unrelated processes. Data, model files, raw numerical snapshots, and large logs stay on NAS. Public delivery currently contains source and protocol only; completed metrics will be exported after a subsequent completion check.
