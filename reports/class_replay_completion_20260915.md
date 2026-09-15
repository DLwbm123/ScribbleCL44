# Class ER / guarded feature-DER: completed mixed-budget runs

Verified on 2026-09-15 at 10:36 China Standard Time. Both supervisors report `complete`, both training exit codes are 0, both summaries contain all three stages, and both final training records reach T3 epoch 60. The recorded processes have exited.

| Method | T1/T2/T3 epochs | Finished (CST, 2026-09-13) | Final mean foreground Dice | A-Dice (includes background) | BWTR | RMA | WCD (includes background) | MPE | DRR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| zs-er | 80/80/60 | 19:08:47 | 0.090774 | 0.361603 | -0.594816 | 0.752010 | 0.172613 | 0.000000 | 0.033333 |
| zs-der | 80/60/60 | 21:38:22 | 0.170163 | 0.414094 | -0.587539 | 0.878904 | 0.233408 | 0.000000 | 0.031667 |

**Both methods show severe forgetting:** final held-out T1 and T2 foreground Dice are approximately zero. T3 foreground Dice is 0.272323 for ER and 0.510489 for guarded feature-DER. Completion is not a claim of satisfactory retention. ER selected T3 epoch 2; guarded DER selected T3 epoch 41. These are single-seed observations.

## Protocol and provenance

- Common settings: seed 42, batch 4, workers 4, SGD LR 0.03, momentum 0.9, weight decay 0.0001, current-image PCE/Global/Spatial weights 1/0.1/0, reservoir 64, replay batch 4. Selection uses current-task foreground validation; selected weights and the stage-end reservoir carry forward.
- ER uses replay sparse PCE weight 1. Feature-DER uses feature MSE weight 0.5 and disables replay PCE/global losses. It is a scribble-supervised feature-DER adaptation.
- Original DER failed at T1 epoch 14 with non-finite loss. The completed DER run uses safe saliency/GraphCut and gradient norm clipping at 5; it must be reported separately from the failed original version. Its underlying BN replay rule, LR and feature coefficient were retained.
- The user shortened remaining tasks on September 13. ER reused completed T1/T2; DER reused completed guarded T1. Remaining stages restarted with their actual 60-epoch learning-rate schedules. ER also retains a discarded 12-epoch T3 attempt privately. Stage-boundary RNG resets to seed 42 because older checkpoints lack full RNG state; this is not an exact trajectory continuation.
- Mixed budgets differ between methods and from the 80-epoch independent references. Do not label this as a matched 60- or 80-epoch-per-task comparison. Historical failed/restarted attempts are excluded from the completed performance table but remain part of the compute cost.
- Benchmark metrics reuse `class_runtime/run_formal_supplement.py:metrics`. A-Dice, BWTR and RMA use background-inclusive task Dice. RMA uses fixed common independent T2/T3 references 0.7786178640330347 and 0.7820229984174101, both trained for 80 epochs; T1 is excluded. These existing references were read from published aggregate evidence, not rerun. WCD includes background; MPE is model parameter expansion and DRR counts reused source slices.

## Source and validation

The updated `class_replay_runtime` overlay was copied from the actual ER runtime. Copy the existing `class_runtime` into a neutral campaign `source` directory, then overlay `class_replay_runtime/*.py`. For guarded DER, additionally overlay `class_replay_runtime/guarded/*.py` into its separate source directory. The guarded helpers were copied from the actual completed DER runtime. Keep external datasets, sparse annotations, resume checkpoints and the campaign `paths.json` private. Use the existing `job.py` supervisor and environment-supplied `RUN_SPEC`; run arguments are recorded in the sanitized manifests, with `--task-epochs 80 80 60` for ER and `--task-epochs 80 60 60 --safe-numerics --grad-clip-norm 5` for DER. A fresh run does not reproduce the historical restart trajectory.

The completion supervisor checks all task epoch sequences, finite losses, nonempty task recovery states, active historical replay, and disabled loss branches. The prior unequal-task-budget resume regression passed. Current closeout verified receipts, exit codes and summary budgets; published Python files were syntax-checked. No training was launched or changed during closeout. NFS multiprocessing temporary-directory cleanup emitted EBUSY warnings in the historical logs; both jobs still exited successfully.

Public payload: source overlay, sanitized manifests, aggregate matrices and benchmark metrics in `results/class_replay_completion_20260915/aggregate.json`. Patient/subject-level metrics, input data, predictions, checkpoints, replay contents, authentication and raw logs are excluded.
