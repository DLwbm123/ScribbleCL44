# Organ-CL ZS-DER++ runtime repair audit

## Scope and source provenance

The authoritative runtime is `/home/jiangsuiyang/q4n6x` on the training server,
not a same-named local checkout.  That directory has no Git metadata.  This
release synchronizes the complete source-only runtime file set used by the
entry path (`main.py`, `runner.py`, `runner_core.py`, models, losses, and
helpers) into this public repository.  Runtime data, patient data, checkpoints,
and raw logs remain excluded.

The release check compares every runtime source file that affects this path
against the server source (the repaired files exactly; a few upstream files
differ only by a terminal newline).  It intentionally does not use a checksum
or publish any external-data path.

## Findings and repairs

| Item | Audit result | Repair / guard |
| --- | --- | --- |
| Sparse-label rotation border | Confirmed: an arbitrary rotation filled labels with `0`, a valid annotated background value. | Label rotation now fills with `IGNORE_INDEX` (`-100`); image rotation is unchanged. |
| Organ dispatcher | Not reproduced: server `runner.py` already calls `run("organ")`. | A runnable audit locks this route down, instead of adding a second dispatcher. |
| DER++ exemplars | Confirmed: replay received post-augmentation tensors. | `H5Slices` returns an unaugmented replay copy only for DER++; the buffer receives that copy and its sparse label. |
| Feature target / replay BN | Confirmed as a causal risk. | A shared `freeze_batchnorm_stats` guard wraps target extraction and feature replay, while keeping gradients through the replay feature loss. |
| Replay global forward BN | Confirmed as a causal risk. | The whole DER++ replay PCE/global forward runs under the same backbone BN-stat guard. |
| Best checkpoint / buffer | Confirmed: only the model had been restored. | Each best checkpoint now includes model plus DER++ state; restoration is paired. |
| Disabled GD logging | Confirmed misleading bookkeeping. | `zs_gd_loss` is a scalar zero unless `--zs-gd-loss` is present; no GD term enters the objective otherwise. |
| Test leakage in gates | Confirmed as a development-process risk. | Runs select only on validation. Test evaluation is opt-in (`--test-evaluation`), recorded in the manifest, and absent by default. |
| Buffer observability | Missing. | DER++ state now records stored samples, sparse known/background/foreground/ignore pixels per task, and replay draw coverage per task. |

The BN fixes prove that DER++ replay no longer mutates backbone running
statistics.  They do **not** prove that BN drift was the sole cause of the old
T2 collapse: that requires an after-repair controlled Organ T1→T3 validation
gate.  The gate below is therefore the causal performance test; historical
test matrices are not used to choose a configuration.

## Executed checks

All ran against `/home/jiangsuiyang/q4n6x` with the existing Python
environment.

| Check | Result |
| --- | --- |
| Python compilation of the repaired runtime files | PASS |
| `zs_derpp_smoke.py` | PASS: exact feature target, replay PCE, buffer coverage, serialized restore, and unchanged BN statistics |
| `organ_derpp_runtime_audit.py` | PASS: Organ dispatcher, raw replay source, rotated-border `-100`, and opt-in test evaluation |
| `organ_derpp_e2e_smoke.py` through `main.py --setting-run` | PASS: validation-only summary, validation matrix, DER++ coverage, paired best model/buffer restore |
| `zs_audit_pipeline_output.py` inside that end-to-end smoke | PASS: two stages, replay objectives, coverage, and selection provenance |

## Validation-only retention gate

Four short T1→T3 Organ DER++ runs use seed 42, 20 epochs/task, U-Net, PCE +
0.1 global, and no `--test-evaluation`.  They differ only in DER++ controls:

| Run ID | alpha | beta | buffer |
| --- | ---: | ---: | ---: |
| `x4a1` | 0.5 | 0.5 | 128 |
| `x5a2` | 1.0 | 0.5 | 128 |
| `x6a3` | 0.5 | 1.0 | 128 |
| `x7a4` | 0.5 | 1.0 | 256 |

Promotion criterion: after T3, T2 validation Dice must fall by at most 0.10
from its post-T2 value, retain at least 80% of that value, and T3 must show
non-trivial current-task learning.  If no run meets the criterion, no formal
training is started.

## Gate incident: numerical failure before retention evaluation

The four runs stopped during T1, before any T2/T3 matrix could be produced.
No test evaluation was enabled or read, no configuration was selected, and no
formal run was started.

| Run ID | alpha | beta | buffer | Last completed T1 epoch row | Terminal condition |
| --- | ---: | ---: | ---: | ---: | --- |
| `x4a1` | 0.5 | 0.5 | 128 | 3 | `FloatingPointError: non-finite training loss` |
| `x5a2` | 1.0 | 0.5 | 128 | 3 | `FloatingPointError: non-finite training loss` |
| `x6a3` | 0.5 | 1.0 | 128 | 4 | `FloatingPointError: non-finite training loss` |
| `x7a4` | 0.5 | 1.0 | 256 | 6 | `FloatingPointError: non-finite training loss` |

Each terminal log first reports `mixup.py` producing an invalid value while
constructing the graph-cut unary cost.  The fault therefore occurs before a
retention claim is possible and is independent of the tested DER++ alpha,
beta, and buffer settings.  This is a numerical-stability incident to diagnose
separately; it is not evidence that BatchNorm drift is, or is not, the source
of historical T2 forgetting.
