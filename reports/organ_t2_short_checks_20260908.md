# Organ T2 bounded numerical-stability checks

The prior baseline failed on T2 update 5 with exploding DER++ feature MSE and gradients. Keeping LR 0.03 and the original BN behavior while adding gradient clipping at norm 5 completed 420 finite updates (10 T2 epochs). However, T2 validation predictions remained entirely background. The numerical failure has a bounded mitigation; the T2 segmentation-quality problem remains unresolved. No formal training was restarted, no test-set scores were used, and no training data or annotations were changed.

The checks began after the 21:39 status probe and were all completed and verified by approximately 21:56 on 2026-09-08 (Asia/Shanghai), within the requested approximately 20-minute window. All bounded processes exited; none remains running.

## Protocol

All checks restore the same selected T1 paired model/replay state (T1 validation Dice 0.7593914365), then enter T2. Batch size is 4; T2 has 166 training slices and 42 updates per epoch. The LR schedule retains its original 60-epoch horizon. PCE/Global/Spatial weights remain 1/1/0.01, DER++ alpha/beta 0.5/0.5, buffer 128, replay minibatch 4, SGD momentum 0.9 and weight decay 0.0001. Spatial is inactive throughout these checks because activation is at epoch 30.

Transition seed 43 is used unless indicated otherwise. Seed 44 changes the declared transition RNG (including head initialization and replay/mixing), while loader and worker seeds remain fixed by the existing runner. These are controlled transition checks, not exact replay of the original failure's missing RNG state.

GPUs 4–7 were used concurrently. The sequence was four 42-update screens, four 126-update checks, then three additional 126-update mechanism/seed checks and one 420-update confirmation. Repeated settings restart from the same T1 state to preserve their common trajectory. This amounts to 12 bounded executions, not 12 formal runs. The final batch has an 11-minute process timeout.

Validation occurs only at each bounded endpoint. Debug mode captures finite guards and scalar losses/gradient norms. Epoch checkpoint selection is disabled in these short diagnostics; no truncated run is presented as a formally selected model.

## Completed comparisons before the 10-epoch confirmation

Dice values below are validation scores; `~0` represents values below 1e-8 from the metric's smoothing constant.

| LR | Gradient clip norm | BN change | Updates | Outcome | T1 Dice | T2 Dice |
|---|---:|---|---:|---|---:|---:|
| 0.03 | none | none, prior baseline | 5 | numerical failure | — | — |
| 0.003 | none | none | 126 | finite | 0.5374 | ~0 |
| 0.03 | 5 | none | 126 | finite | 0.6497 | ~0 |
| 0.003 | 5 | none | 126 | finite | 0.5201 | ~0 |
| 0.03 | none | only primary current forward writes backbone BN statistics | 13 | numerical failure | — | — |
| 0.03 | 5 | only primary current forward writes backbone BN statistics | 126 | finite | 0.5077 | ~0 |
| 0.03 | none | freeze backbone BN running statistics | 4 | numerical failure | — | — |
| 0.03 | 5 | freeze backbone BN running statistics | 126 | finite, both tasks predict background | ~0 | ~0 |
| 0.03 | 5 | none; transition seed 44 | 126 | finite | 0.4717 | ~0 |

The unmodified-BN, LR-0.03, clip-5 check clipped 8 of its initial 42 updates. The remaining updates were not rescaled by clipping. Without clipping, reducing BN writers merely delayed failure; freezing all backbone BN statistics still failed. Thus the post-step BN-mode sensitivity found earlier does not establish that changing BN alone repairs training.

Freezing backbone BN means keeping its running buffers fixed, not freezing convolution or BN affine parameters. The clipped frozen-BN check verifies those buffers remained unchanged, but both validation tasks predicted background. That intervention is not recommended from these results.

## Annotation check

T2 has 28,887 marked foreground pixels. All marked foreground pixels lie inside the dense training foreground, and all marked background pixels lie in the dense training background. Marked foreground coverage is 18.20% of dense foreground; foreground is 2.524% of all known scribble pixels. This excludes an all-background or geometrically misplaced annotation archive. The class imbalance is an observation, not proof of why a model predicts only background.

## Ten-epoch result and decision

LR 0.03, clip norm 5, original BN behavior, transition seed 43:

- Completed 420/420 finite updates, corresponding to 10 T2 epochs.
- Validation Dice: T1 **0.7026519037**, T2 **3.33202e-10**. T2 predicted foreground fraction: **0.0**.
- The maximum pre-clipping gradient norm was **27.1869**, compared with approximately 4.76e11 in the failing baseline. Clipping activated on **11/420 updates (2.62%)**.
- Maximum weighted replay-feature MSE: **2.30315**. Spatial remained zero throughout.
- Current-task PCE fell from 0.83113 on the first minibatch to 0.03407 on the last. These are different minibatches; this does not establish useful foreground learning and must not override the all-background validation result.

The existing `--grad-clip-norm 5` is the smallest supported mitigation for the reproduced early numerical crash; no duplicate training loop or loss redesign is needed. This is not proof of stability through Spatial activation at epoch 30 or through T3. It is also not a validated configuration for the full continual-learning experiment.

Do not adopt frozen backbone BN from this test, and do not start the expensive formal run on the strength of numerical finiteness alone. The next bounded investigation should separate new-task foreground supervision from historical replay pressure using the same T2 data and annotation protocol. No such additional investigation or formal restart was launched in this round.

Full scalar evidence, including all 12 bounded outcomes and the initial 42-update scores, is in [summary.json](../results/organ_t2_short_checks_20260908/summary.json). A `passed` status there refers to numerical completion with finite evaluation metrics, not acceptable Dice. Ten checks met that numerical criterion; the two unclipped BN interventions failed as reported above.

## Reproduction

Use the existing environment and original run root:

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  timeout 660 python diagnose_t2_transition.py --root "$RUN_ROOT" \
    --output "$RUN_ROOT/clip5_confirmation_new" --steps 420 --lr .03 --clip 5 --evaluate
```

Other controls are `--lr .003`, `--clean-bn-writer`, `--freeze-backbone-bn`, and `--transition-seed 44`. The BN-freezing override is explicitly diagnostic and is confined to the wrapper; the formal runner remains unchanged. The existing formal runner already exposes `--grad-clip-norm 5`.

`summarize_t2_short_checks.py` exports scalar-only outcomes from the private check directory and checks finite successful-step metrics, endpoint step counts and validation scores. Failure snapshots, images, labels, model tensors and replay buffers remain on the authorized NAS. Published source and scalar tables do not contain them.
