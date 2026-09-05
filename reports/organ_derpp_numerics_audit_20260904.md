# Organ-CL ZS-DER++ numerical audit

## Scope

This repair targets numerical first-error localization only.  It does not use
test metrics for configuration selection, rerun the prior four-way sweep, or
start a T2/T3/official Organ experiment.

## Minimal runtime repairs

| Risk | Runtime change |
| --- | --- |
| Extreme or zero saliency | `rms_saliency` rescales multi-channel gradients safely; `normalized_unary` uses a uniform prior only for an exact zero map. |
| Invalid GraphCut input | Finite/range checks run before every `int32` cast and at the shared GCO adapter boundary. |
| Saturated sparse PCE | Training PCE consumes stable resized log-probabilities while preserving the original probability-resize convention. |
| Replay feature/BN mismatch | Buffer target capture, feature matching, and replay PCE/global forwards freeze shared-backbone BatchNorm running statistics without disabling gradients. |
| Unlocalized failure | `--numerical-debug` guards current/replay saliency and global paths, feature replay, backward/gradient, optimizer state, and buffer capture; its first failure writes a private diagnostic under the run output only. |

Existing fixes were retained: arbitrary-angle sparse-label borders use `-100`,
Organ dispatch enters `run("organ")`, replay stores raw pre-augmentation
samples, DER++ best model/buffer state is restored together, disabled GD is
recorded as zero, and DER++ state contains task/label/replay-coverage audit
fields.

## Executed server checks

The existing server environment ran all checks below:

| Check | Result |
| --- | --- |
| Compilation and Ruff on touched runtime/test files | PASS |
| `test_numerical_safety.py` | PASS, 26 tests, including deferred first-NaN, gradient, optimizer-state, and JSON-safe failure-summary guards |
| `gco_numerical_smoke.py` against the installed GCO backend | PASS; 2x2 solver energy matched exhaustive optimum and non-finite costs were rejected |
| `zs_derpp_smoke.py` | PASS; exact feature target, finite replay PCE, frozen backbone BN, buffer coverage, and serialization restore |
| `organ_derpp_runtime_audit.py` | PASS; Organ entry route, raw replay source, `-100` rotation border, and opt-in test evaluation |
| `organ_derpp_e2e_smoke.py --numerical-debug` | PASS; validation-only two-stage run, paired best state/buffer, coverage, and no first-nonfinite artifact |

## Fixed numerical gate

The sole post-repair diagnostic was T1 only: seed 42, 20 epochs, batch 4,
workers 8, LR 0.03, validation every 200 iterations, ZS-DER++ with PCE + 0.1
global, alpha 0.5, beta 0.5, buffer 128, replay minibatch 8, and no test
evaluation. Its run identifier is anonymous.

**Status: `T1_NUMERICAL_GATE_FAIL__RETENTION_UNTESTED`.** It passed epoch 0,
epoch 1, and epoch 2 and reached the epoch-3 validation at iteration 600. The
first detected non-finite value occurred after that validation and before the
epoch-3 row completed, at `buffer_capture/feature_targets`: a four-sample
64x244x244 backbone feature-target batch had 15,239,808 non-finite entries.
This is direct evidence of exploding/invalid backbone output at buffer target
capture, not evidence that GCO casting itself was the first non-finite
operation. It does not establish which preceding update caused that output;
the gate has not tested retention. No T2/T3, sweep, or formal run was started.

## Privacy and publication boundary

Only source, small tests, and scalar/boolean public evidence belong in this
repository.  Patient data, sparse labels, buffer tensors, failing batches,
checkpoints, and raw training logs remain on the server data volume.
