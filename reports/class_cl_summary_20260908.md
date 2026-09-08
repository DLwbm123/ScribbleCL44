# Class-CL completed results, verified 2026-09-08

The persisted Class-CL ZS-DER++ + MiB formal run `v6c43` completed T1/T2/T3, each for 80 epochs and 30,000 updates (90,000 total). All 240 recorded epoch losses were finite; each stage records epochs 0–79. Final summary, matrix and nonempty final/best/paired state files were present, and the final runtime audit log ends in PASS. This verification reads existing artifacts; no new training or test inference was run.

Task order on MMWHS: T1 = MYO/LV/LA; T2 = RA/RV; T3 = AO/PA. Metrics below are held-out test foreground Dice, selected through validation checkpoints. [Full precision aggregate evidence](../results/class_cl_summary_20260908/summary.json).

| Training completed through | T1 test Dice | T2 test Dice | T3 test Dice | Seen-task mean |
|---|---:|---:|---:|---:|
| T1 | .7114 | — | — | .7114 |
| T2 | .7598 | .7264 | — | .7431 |
| T3 | .7382 | .6957 | .6587 | **.6975** |

During T3, T1 decreased by .021616 and T2 by .030664 relative to their post-T2 scores. Their average drop over this transition is .026140. Relative to its original post-T1 score, final T1 is still higher by .026792. These differences describe this single-seed trajectory and are not a causal comparison with Organ-CL or a no-replay baseline.

Separate whole-heart seven-class test evaluation gives **.691844**. This is a separate evaluation on `whole_heart_test.h5`, not the unweighted mean of the three task-subset scores. The .697531 value is the mean of T1/T2/T3 evaluations; the two metrics should not be interchanged.

| Whole-heart class | Dice |
|---|---:|
| MYO | .6588 |
| LV | .7647 |
| LA | .7631 |
| RA | .6476 |
| RV | .7255 |
| AO | .7294 |
| PA | **.5539** |

Final seen-task validation mean is .742233. The task-specific best-validation checkpoints were selected at one-based epochs 79 (T1), 7 (T2) and 27 (T3); all stages nevertheless executed 80 epochs. Validation was checked every 200 updates in this historical run, rather than the per-epoch selection of the current Organ experiment.

## Hyperparameters and sweep

Manifest-verified formal settings: ZS-DER++ + MiB, seed 42, 80 epochs/task, PCE 1, Global .1, **Spatial 0**, MiB KD 1, replay capacity 64, replay batch 4, DER++ alpha/beta .5/.5. The launch protocol records SGD LR .03 and training batch 4; these two fields are absent from the older manifest and are therefore launch-record provenance rather than manifest verification. This historical Class configuration must not be confused with the current Organ run's Global 1, Spatial .01 and buffer 128.

Two 20-epoch-per-task T1/T2 sweep candidates completed:

| Run | MiB KD weight | Post-T2 seen validation mean | Post-T2 seen test mean |
|---|---:|---:|---:|
| c6m3 | **1** | **.5981** | .6234 |
| c7m4 | 10 | .5658 | **.6594** |

KD 1 was selected by validation. KD 10 had higher test Dice in the short sweep, but those test results were not used for hyperparameter selection. Neither short run establishes the better coefficient for a full T1/T2/T3 sequence. Earlier candidates c6m1/c7m2 failed at T2 from memory pressure with replay batch 8; they are not completed performance results. Two-batch smoke runs are also excluded from performance conclusions.

The September 3 intermediate-checkpoint report recorded a separate test diagnostic (T1 .7173, T2 .6771, mean .6972). That historical diagnostic differs from the completed run's persisted post-T2 matrix above; exact checkpoint/evaluator parity has not been re-established here, so they are not merged into one trajectory. This summary uses the final persisted matrix consistently.

Source of truth: authorized NAS `runs/v6c43`, `runs/c6m3`, `runs/c7m4`, and `logs/v6c43.log`. The historical Class runner resides in the separate Class checkout `q2m8v`; the current Organ branch's runner is not asserted to be an exact reproduction of that Class implementation. Public scope here is aggregate results and this read-only verification; patient-level outputs, raw data and checkpoints remain private. No baseline superiority or clinical claim is established by these single-seed results.
