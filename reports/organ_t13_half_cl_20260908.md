# Organ CL: half T1/T3 training sets, full T2

Status: **paused at the user's request after T1 epoch 1 of the initial 80-epoch run**. The exact matching training and loader processes were terminated; previous outputs are preserved. The launcher is now configured for **60 epochs per task**, writing to new `run60`, `logs/train60.log` and `training60.exitcode` paths. **The 60-epoch run has not been started.** The original launcher is retained remotely as `run_organ_half_cl_80_launched.sh`.

Historical startup was 2026-09-08 at 18:50:32 Asia/Shanghai on GPU7. The completed first T1 epoch had 96 updates and took 116.89 seconds. The interruption is intentional, not a numerical training failure.

## Final user-authorized protocol

| Task | Original training slices | Selected training slices | Retained training patients | Epochs | Updates/epoch |
|---|---:|---:|---:|---:|---:|
| T1 | 762 | 381 | 25 / 25 | 60 | 96 |
| T2 | 166 | 166 | 7 / 7 | 60 | 42 |
| T3 | 1,421 | 711 | 17 / 17 | 60 | 178 |

The user corrected the reduced tasks from T1/T2 to **T1/T3**, then requested completing all three tasks while evaluating T2 after the first T3 epoch. No T1/T2-reduced dataset or run was created. Each halved task uses seed-42 random sampling within each training patient; odd counts are allocated reproducibly to reach the global half rounded up. All training patients are retained. Images, dense training labels and original scribbles use the same selected indices. The source datasets are preserved: training images use HDF5 virtual views, and validation/test datasets link to the originals. T2 remains full data. This is slice subsampling, not a reduced-patient cohort or scribble-density change.

Full validation/test sizes remain T1 153/298, T2 52/100, T3 343/496. Synthetic self-checks passed for deterministic sampling, patient retention, image/label/scribble alignment and unchanged evaluation links. Actual views were checked once for expected training and annotation shapes, patient boundaries, a readable finite image slice and full evaluation sizes. [Aggregate subset metadata](../results/organ_t13_half_cl_20260908/subset.json).

Training uses the same guarded continual runner as the throughput probe, with fresh initialization and the existing paired model/replay checkpoint restore between tasks. This is not the independent Domain-reference adapter. Fixed controls: seed 42; SGD LR .03, momentum .9, weight decay 1e-4, polynomial exponent .9; batch 4; workers 8; CPU threads 4; PCE/Global/Spatial 1/1/.01; Spatial first active at one-based epoch 30 **of each task** (warmup argument 28); no gradient clipping; ZS-DER++ alpha/beta .5/.5, buffer capacity 128 and replay minibatch 4. cuDNN deterministic is enabled, benchmarking disabled, CuBLAS workspace `:4096:8`, deterministic algorithms warn-only. The known grid-sample backward warning remains.

Validation selects each task's best paired checkpoint. Test metrics are reported at task boundaries, without selecting checkpoints. After exactly the first full T3 epoch (178 updates), the new `--t3-first-epoch-evaluation` hook evaluates the current model on full T2 validation and test splits. It records before/after Dice, absolute drop and retention ratio in `run60/t3_epoch1_t2_retention.json`, then continues to T3 epoch 2 without changing optimizer, model, replay state or LR horizon. The configured next run finishes all 60 T3 epochs; there is no early-stop controller and no T4 training. Normal persistent checkpoints are enabled; the timing harness's RAM-only checkpoint behavior is not used.

## Operation and estimate

Existing run root on the authorized NAS: `ScribbleCL/organ_T13_half_cl_20260908`.

- Background session: `organ-t13-half-cl-20260908`.
- Next-run main log: `logs/train60.log`; old log: `logs/train.log`.
- Epoch records: `run60/train.jsonl`; configuration after restart: `run60/manifest.json`.
- First-T3-epoch T2 forgetting: `run60/t3_epoch1_t2_retention.json`.
- Completion marker: `training60.exitcode`; final metrics: `run60/summary.json`.

Approximate planning estimate: **5.5–7 hours total for 60 epochs/task**, with the first-T3-epoch retention result around **2.5–3.5 hours after restart**. The earlier 5.5–6.5 hour full-data estimate applied to T3 running only one epoch and is superseded. These are throughput extrapolations, not guarantees; Spatial-on T3 throughput and future storage contention were not directly measured. GPU7 had 24,121 MiB free before launch; the actual NAS mount had 30 TiB free and passed a write/read probe. No other process was stopped or modified.

The experiment is currently stopped. Once restarted, the background launcher runs independently of SSH/Codex. No recurring monitoring is scheduled. Its eventual completion must be checked before reporting final scores.

## Reproduction

With the existing environment and data, prepare the subset with `prepare_organ_half_train.py --data-root "$DATA_ROOT" --sparse-root "$ORIGINAL_SPARSE_ROOT" --output "$RUN_ROOT/subset" --halve T1 T3`. Put the guarded source tree and launcher under `$RUN_ROOT/source`, then run `bash "$RUN_ROOT/source/run_organ_half_cl.sh" "$RUN_ROOT" "$PYTHON" 7` in tmux. The data-preparation self-check is `python prepare_organ_half_train.py --self-test`.

Published scope: source changes, subset-generation/launch scripts, aggregate subset counts and this prospective protocol. Images, annotation arrays, patient identifiers/selected indices, checkpoints and patient-level runtime results remain private. Final training results are pending; the prepared 60-epoch run remains paused.
