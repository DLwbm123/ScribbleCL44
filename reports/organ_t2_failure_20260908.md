# Organ-CL T2 transition numerical failure

The 60/60/60 Organ run stopped during its first T2 epoch while capturing DER++ backbone feature targets. A bounded diagnostic from the saved T1 paired model/replay checkpoint reproduced the same failure branch on T2 update 5. The measured mechanism is a rapidly increasing replay-feature MSE and gradient, followed by FP32 overflow during a backbone forward using stored BatchNorm statistics. This is a numerical failure, not an out-of-memory exception or a completed low-Dice T2 run.

## Original run and evidence boundary

- T1/T3 training subsets: one half; T2: full; batch size 4; 60 epochs per task.
- T1 completed all 60 epochs. Validation selected epoch 17: validation Dice 0.7593914365; test Dice 0.6648380769.
- Both the model and DER++ buffer were restored from the same selected T1 checkpoint before T2. This run does not use the terminal-buffer/selected-model mismatch.
- SGD resets to LR 0.03 at the beginning of each task; momentum 0.9; weight decay 0.0001; gradient clipping disabled.
- PCE/Global/Spatial weights: 1/1/0.01; Spatial first activates at one-based epoch 30 of each task. It was inactive during the failing T2 epoch, and the selected T1 epoch precedes activation.
- Original `numerical_debug=false`: no failure snapshot or exact failed update was preserved. The diagnostic uses explicit transition seed 43; it is not a claim to reproduce the original random stream or original failed update number.
- The original finite checks still cover input, losses, gradients, optimizer/model tensors and captured features. However, finite values can already be dangerously large. The zero gradient norms in ordinary non-debug logs are placeholders returned by the checker, not evidence of zero gradients.

## Bounded diagnostic

The diagnostic reused the existing training loop, T1 paired state, annotations, batch size and 60-epoch learning-rate horizon. It skipped T1 and stopped on failure, with a maximum budget of 42 T2 updates. No hyperparameter sweep or formal restart was performed.

| T2 update | Weighted replay feature MSE | Total gradient norm | Spatial loss |
|---|---:|---:|---:|
| 1 | 0.008311 | 12.7543 | 0 |
| 2 | 0.137345 | 6.1058 | 0 |
| 3 | 0.240207 | 6.8224 | 0 |
| 4 | 29.098345 | 4,819.8369 | 0 |
| 5 | 8,147,532,288 | 476,192,881,402.7 | 0 |

On update 5, the current-task PCE was 0.6437 and Global loss 0.2594. Replay PCE also increased to 630,542.125, but replay feature MSE dominated the total loss of 8,147,847,680. The feature loss is `alpha * mse_loss(features, stored_targets)`, with alpha 0.5 and no feature normalization.

The snapshot's model tensors remained finite. Their maximum floating-point absolute value grew from 132.54 before the update to 1.60736e9 afterward. Raw and augmented inputs were finite, with maximum absolute value 6.05.

Using stored BatchNorm running statistics, the first non-finite operation was the convolution `Conv2.conv.0`. Re-evaluating that operation in FP64 produced finite output with maximum absolute value 1.13822e40, above FP32's approximately 3.40282e38 limit. All final feature elements were non-finite.

For the same post-update weights and input, a diagnostic forward using current-batch BatchNorm statistics without writing running buffers remained finite (feature absolute maximum 2726.39). This establishes normalization-mode sensitivity of the failed model. It does not establish a validated training repair, nor isolate BN behavior as the sole initiator of the gradient explosion.

## Interpretation and next step

The supported failure chain is: task transition with a new head and LR reset; rapidly increasing replay-feature mismatch and gradient; a very large SGD update; forward activation overflow under stored BN statistics; finite guard stops the run before contaminated features can be accepted in ordinary training. The task transition, restored historical feature targets and BN behavior are consistent with the observed instability, but their separate causal contributions have not been isolated.

Spatial timing, low scribble coverage, data-size reduction, and GPU contention are not demonstrated causes. In particular, Spatial contributes exactly zero in this diagnostic. An independent T2 run lacks the historical replay-feature constraint, so good independent Dice does not imply that the transition is numerically stable.

The smallest next investigation is a short fixed-transition check of learning-rate warmup/lower transition LR and gradient clipping, followed by an isolated check of consistent BN statistics between current and replay paths. Do not treat switching replay to batch statistics, clipping, or reducing alpha as a proven repair without checking learning and retention. Formal Organ training remains stopped.

## Reproduction and artifacts

Use the original server environment and the original run root containing `source`, `subset`, and `run60/s01_state.pt`:

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python diagnose_t2_transition.py --root "$RUN_ROOT" --output "$RUN_ROOT/diagnostic_new" --steps 42

PYTHONPATH="$RUN_ROOT/source" CUDA_VISIBLE_DEVICES=7 \
  python organ_poststep_diagnose.py \
    --snapshot "$RUN_ROOT/diagnostic_new/run/FIRST_NONFINITE.pt" \
    --output "$RUN_ROOT/diagnostic_new/poststep_scalar.json" --device cuda:0
```

The first command is expected to exit nonzero on the reproduced numerical failure; the second completed successfully. The older post-step helper probes model modes explicitly; its generic reconstruction disclaimer must not be read as denying the new snapshot's pre-step capture.

Two initial diagnostic wrapper attempts failed at argument/output-directory setup, before training; their artifacts were preserved. The third attempt ran and generated the failure evidence reported here.

Published: diagnostic wrapper, existing scalar layer-trace helper, this report, and [compact scalar results](../results/organ_t2_failure_20260908/summary.json). Checkpoints, replay images, labels, raw data and the private failure snapshot remain on the authorized NAS. No tensors or patient identifiers are included in the published summary.
