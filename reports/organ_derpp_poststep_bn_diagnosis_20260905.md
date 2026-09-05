# Organ-CL ZS-DER++ post-step diagnosis

## Gate status

`T1_NUMERICAL_GATE_FAIL__RETENTION_UNTESTED` remains in force.  No T2/T3,
held-out test, sweep, or formal run was started.  The only execution after the
failure is a bounded T1 numerical-stability check described below; it is not a
model-selection experiment.

## Read-only failure-snapshot evidence

The snapshot's legacy metadata identifies `buffer_capture/feature_targets` as
the failure, but does not retain an exact historical stage, mode map, pre-step
state, or replay draw.  The stage is therefore inferred as 0 only from the
saved Organ head keys.  All 161 saved model tensors, all 92 optimizer tensors,
and all five DER++ buffer tensor groups were finite.  The maximum absolute
values were respectively 239.08, 9681.37, and 22.43.

On the saved raw replay batch, the shared-backbone running-statistic probe
reproduced the failure: the final 64-channel feature map had 15,239,808
non-finite values.  The augmented-current-image probe also failed (15,240,551
non-finite values).  In contrast, batch statistics with `track_running_stats`
disabled were finite, did not mutate any BN buffers, and had feature-map
absolute maximum 143.77.

The first non-finite operation under the running-stat probe is
`backbone.Up_conv2.conv.3` (a `Conv2d`): its output has 37,330 non-finite
values.  Its input is still finite but has absolute maximum `1.447e37`; the
preceding BatchNorm output is likewise finite at that scale.  Its saved running
variance is finite and non-negative, while the same operation in FP64 produces
a finite output (absolute maximum `2.252e39`).  This localizes the immediate
failure to FP32 convolution overflow after a severe running-stat mismatch; it
does not show non-finite parameters or an illegal BN variance.

This is evidence that shared-backbone BN running-stat use is causal for the
captured numerical failure.  It is **not** evidence that validation caused the
failure, nor does it establish an explanation for Organ T2 retention.

## Implemented, bounded changes

- `--zs-clean-bn-writer` is opt-in.  With it, ZS saliency and cutout forwards
  freeze shared-backbone BN statistics; the primary current-image forward is
  the only ZS shared-backbone BN writer.  DER++ replay remains no-write.
- Numerical guards reject non-finite gradients and optimizer/model state even
  without debug mode.  Debug gradient norms use scaled FP64 accumulation.
- Debug runs retain one non-aliased pre-step model/optimizer candidate, actual
  drawn replay minibatch, transform trace, RNG, modes, trainable-parameter
  state, scalar losses, gradient norm, and learning rates.  A failure snapshot
  contains that bounded capture rather than another full historical buffer.
- Every debug step now flushes post-buffer checks and writes a scalar-only
  finite trace.  Validation restores every module's original training flag in
  `finally`, including exception paths.

The prior sparse-label rotation, raw replay storage, paired best-buffer restore,
test-selection prohibition, and buffer coverage logic were retained.

## Verification and next allowed command

The server environment passed `ruff check` and `30 passed` numerical tests.

The fixed-seed local T1 check used the opt-in clean BN writer, 32 training
batches, numerical debug, and no held-out test.  All 32 scalar trace rows are
`post_step_finite`; the largest recorded gradient norm is 13.96 and no new
failure snapshot was written.  The output contains no test matrix.  Its single
end-of-check validation mean was `3.48e-10`, which is expectedly not a usable
performance measurement after this deliberately truncated run.  It is not used
for configuration selection and does not change the gate status.

The following is intentionally printed but not automatically run as a 20-epoch
experiment:

```bash
CUDA_VISIBLE_DEVICES=4 /home/jiangsuiyang/anaconda3/envs/py38/bin/python main.py --setting-run \\
  --data-root <CL_Benchmark_data> --sparse-root <sparse_annotations> --output <new_private_output> \\
  --device cuda:0 --seed 42 --method zs-derpp --zs-global-weight 0.1 --zs-clean-bn-writer \\
  --epochs-per-task 20 --max-task 1 --der-buffer-size 128 --der-minibatch-size 8 \\
  --der-alpha 0.5 --der-beta 0.5 --numerical-debug
```
