# Organ T2 supervision/replay balance and task-specific candidate

All four bounded T2 checks and both BN probes completed. These are validation diagnostics, not a completed 60-epoch continual experiment. Formal training remains stopped.

## Controlled loss ablations

Each candidate restores the same validation-selected T1 paired model/replay state (T1 validation Dice 0.759391, test Dice 0.664838), then runs 420 updates / 10 T2 epochs. Transition seed 43 is declared because the historical checkpoint omitted RNG state. LR 0.03 retains the original 60-epoch decay horizon; SGD momentum 0.9, weight decay 1e-4, batch 4, buffer 128, replay minibatch 4, gradient clipping 5. Original PCE/Global weights are 1/1. Spatial is inactive throughout these first 10 epochs (original first active epoch 30).

| T2 current supervision | Feature alpha | Supervised replay beta | T1 validation Dice | T2 validation Dice |
|---|---:|---:|---:|---:|
| Original PCE, prior 420-step reference | 0.5 | 0.5 | 0.702652 | approximately 0 |
| Original PCE | 0 | 0 | 0.174437 | 0.074959 |
| Original PCE | 0 | 0.5 | 0.627003 | 0.170430 |
| Original PCE | 0.5 | 0 | approximately 0 | approximately 0 |
| Equal present-class mean PCE | 0.5 | 0.5 | 0.015252 | approximately 0 |

Zero coefficients remove loss contributions only: replay forwards, sampling, and reservoir updates are retained. This is not independent training or a complete replay-removal ablation. As training progresses the replay buffer may also contain T2 samples. The balanced variant changes only the primary current-task PCE; replay PCE and saliency supervision remain unchanged. Near-zero Dice includes the metric's epsilon, rather than exact mathematical zero.

Keeping supervised replay improves both tasks over removing both losses. Removing feature MSE improves short-term T2 acquisition relative to full replay, but reduces T1 retention. Equalizing foreground/background PCE gradient mass is not supported by these results.

T2 has 28,887 foreground scribble pixels out of 1,144,407 known pixels (2.524%); 92/166 training slices contain foreground scribbles. Read-only training-label overlap checks found all foreground/background scribbles inside their corresponding dense class. A 2026-09-09 re-audit of the active integer-converted training labels gives 146,926 dense foreground pixels and **19.6609%** foreground coverage. This corrects the previously reported 18.204% denominator; the 28,887 scribble pixels and experiment inputs are unchanged.

## Gradient evidence

At identical update 1, the original current PCE+Global backbone gradient norm is 11.68985, weighted feature replay 0.22965, weighted supervised replay 0.27112. Their cosines with the current gradient are +0.1363 and +0.00475. Equal-class PCE raises the current norm to 93.42985; the history norms are unchanged. At update 126 of feature-only replay, current/feature norms are 0.43848/3.35216, but cosine is approximately +0.00021.

The balance evolves; these probes do not establish systematic opposing gradients from history. Feature regression can dominate gradient magnitude later without useful segmentation retention. Probes use `autograd.grad` on the shared backbone at updates 1, 5, 42, 126, 420 and leave optimizer gradients unchanged. A runnable check covers this property and sparse-PCE empty/single-class cases.

The first launch attempt failed before training because an in-memory source insertion anchor was ambiguous. The anchor was scoped to the guarded backward call; all four corrected runs completed 420 finite updates. This startup error is not an algorithm failure.

## Train-image-only BN probes

Model parameters remain fixed. Reset and cumulatively estimate selected BN running buffers using all 166 clean T2 training images, batch 4, no augmentation, no label use, no optimizer. Validation/test images are not used for calibration. Ordinary eval-mode validation follows calibration.

| Model | Calibrated BN scope | T1 validation Dice | T2 validation Dice |
|---|---|---:|---:|
| Supervised replay only | None | 0.627003 | 0.170430 |
| Supervised replay only | T2 head (1 BN) | 0.627003 | 0.352248 |
| Supervised replay only | Backbone (22 BN) | 0.494230 | 0.333811 |
| Supervised replay only | Both | 0.494230 | 0.370906 |
| Balanced PCE + full replay | None / T2 head | 0.015252 | approximately 0 |
| Balanced PCE + full replay | Backbone | 0.213667 | 0.099499 |
| Balanced PCE + full replay | Both | 0.213667 | 0.112339 |

T2-head calibration reveals a partial running-statistics mismatch and preserves T1 predictions at fixed shared weights. Backbone recalibration changes T1. Calibration does not rescue the balanced-PCE candidate.

## Opt-in task-specific implementation

The shared `runner_core.py` accepts `--organ-t2-supervision-strategy` for continual Organ ZS-DER++. It is disabled by default and rejected for other methods/scenarios or independent Organ training.

| Task | Feature alpha | Supervised beta | Gradient clip | Head BN calibration |
|---|---|---|---|---|
| T1 | Original CLI value | Original CLI value | Original CLI value | None |
| T2 | 0 | 0.5 | 5 | T2 head, before checkpoint-selection validation |
| T3 | Original CLI value | Original CLI value | Original CLI value | None |

With the original launch parameters, T1/T3 alpha/beta are 0.5/0.5 and clipping is absent. LR, epochs, spatial schedule, annotations, and original PCE remain unchanged. There is one shared training loop. The manifest and stage checkpoints record effective policies. Each stage explicitly resets replay coefficients, so T2 does not leak into T3. T3-only resume validates the saved T2 policy before strict replay restoration, then resets T3 coefficients.

Calibration uses a separate clean training-image loader and changes only T2 head running buffers. It restores module train/eval modes and BN momentum afterward. The selected checkpoint contains the exact running statistics used in its validation. Duplicate validation at the same update does not repeat calibration.

**This is a candidate implementation, not a proven full-training improvement.** Repeated calibration during training can change the trajectory compared with the post-hoc probe above. Also, T2 updates the shared backbone: keeping T1's training strategy unchanged does not guarantee unchanged T1 performance. T3's original feature replay may still need its own stability check. No formal run was started by this change.

## Verification

CPU checks passed for head-buffer isolation, mode/momentum restoration, and strict replay loading. The native shared GPU loop passed a synthetic-data smoke run through T1/T2/T3 (one batch/epoch each) and a separate T3-only restore. Effective alpha values were 0.5/0/0.5. These checks verify implementation, not segmentation performance. See `results/organ_t2_balance_20260908/task_strategy_checks.json`.

## Reproduction and artifacts

- Scalar ablations: `results/organ_t2_balance_20260908/summary.json`.
- BN tables: `supervision_replay_bn_calibration.json` and `balanced_pce_bn_calibration.json` in that directory.
- Diagnostic tools: `diagnose_t2_transition.py`, `summarize_t2_short_checks.py`, `diagnose_t2_bn_calibration.py`.
- Checks: `test_t2_balanced_pce.py`; `test_organ_task_strategy.py` (CPU policy/BN checks; optional `--smoke-root /nas/path` native GPU loop on synthetic data).
- Actual execution root on the authorized server: `/data_nas/jiangsuiyang/ScribbleCL/organ_T13_half_cl_20260908`; corrected runs under `balance_checks_20260908_v2`.
- Diagnostic weights are weights-only, not exact resumable training states. They, raw checkpoints, private numerical snapshots, and all patient images/labels remain on NAS and are excluded from publication.
