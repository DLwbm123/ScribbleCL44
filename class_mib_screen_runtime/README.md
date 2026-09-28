# Class ER / feature-DER with sparse MiB: bounded screen

Authorized scope (2026-09-28): short validation of ER and DER only. No automatic
formal continuation, extra seeds, tuning sweep, or other-method training.

The shared `class_replay_runtime/runner_core.py` now accepts `--with-mib` for
sparse Class baselines. It reuses the existing unbiased sparse classification
and teacher distillation implementation used by `zs-derpp-mib`. Existing methods
remain unchanged without the switch. This is the project's sparse MiB adaptation,
not a new claim of a full original-paper MiB reproduction.

Each method reuses its own completed 80-epoch T1 selected model and stage-end
reservoir. MiB is inactive at T1, making that prefix compatible. RNG resets to
seed 42 at the boundary, as in the existing restore path. Switching MiB after
later completed tasks is rejected. Historical T1 results remain provenance;
new screening uses validation only.

Run one two-update T2 MiB check per method, then paired T2 controls and MiB
candidates for five full epochs each. Every job restarts from the same T1
prefix; smoke/screen weights are never reused. Fixed settings: seed 42,
batch 4, workers 0, SGD LR .03, PCE/Global/Spatial 1/.1/0, reservoir 64,
replay minibatch 4, MiB KD coefficient 10. ER uses replay PCE coefficient 1;
feature-DER uses feature MSE .5 and its existing numerical guards/clipping 5.
No BN/head-freezing or feature-refresh improvement bundle is enabled.

Keep original current-task validation checkpoint selection, and report both
T1/T2 foreground validation Dice and their mean. Test data is not evaluated for
the new stages. Paired controls use the same workers and schedule as candidates.
These short, single-seed observations cannot replace full three-task results.

Deployment reuses the actual `class_replay_60e_20260913` ER/guarded-DER sources,
overlays the shared runner plus `entry.py`, and links the interpreter as `main`.
`job.py` runs detached, using one A100 per method (GPU2 ER, GPU3 DER), with
22000 MiB free-memory admission. Its executable assertions check completed
epochs, finite losses, active MiB and the correct replay branches. Failure stops
that method's queue without retry. Private `plan.json` supplies input paths;
`resume/<method>` supplies existing T1 weights, reservoir, manifest and metrics.
Arguments travel through the environment; visible process names are neutral.

Campaign: `class_mib_screen_20260928`. Read `receipts/<method>.json`,
`logs/*.exitcode`, and `jobs/*/summary.json` for actual completion. Publish the
code, aggregate paired validation results and report to the corresponding
GitHub repository after completion; exclude raw data, patient metrics,
checkpoints, replay buffers, private paths and raw logs.
