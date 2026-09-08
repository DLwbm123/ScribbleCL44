# Full-data Organ CL throughput estimate — 2026-09-08

Completed two bounded timing probes on physical GPU7 (RTX 3090), using the existing guarded Organ CL `runner_core.py`. Spatial-off ran 12 updates each for T1/T2/T3; Spatial-on ran 12 each for T1/T2. Both exited successfully. The first four updates per task were excluded from step-time averages, leaving eight measured updates. These are timing probes, not acquisition or forgetting results. No formal training or reduced dataset was launched.

## Measurements and full-data projection

| Task | Full-data updates/epoch, batch 4 | Spatial off, seconds/update | Spatial on, seconds/update | Full epoch including validation | Planned task duration |
|---|---:|---:|---:|---:|---:|
| T1 | 191 | 0.9082 | 0.9562 | 180–189 seconds | 80 epochs: 4 h 07 min |
| T2 | 42 | 0.9690 | 1.0144 | 42–44 seconds | 80 epochs: 57 min |
| T3 | 356 | 0.9647 | not measured | 364 seconds | 1 epoch: 6 min |

For Spatial first active at one-based epoch 30, the calculation is `191*(29*T1_off + 51*T1_on) + 42*(29*T2_off + 51*T2_on) + 356*T3_off`, plus measured full validation costs. Per-epoch validation and final/seen-task stage validations together yield **18,711 seconds, approximately 5 h 12 min**. Checkpoint filesystem writes, test evaluation, initialization, loader startup and future contention are excluded. A practical reservation is **5.5–6.5 hours**, a planning allowance rather than a statistical confidence interval or a guaranteed bound under storage stalls.

Spatial increased measured update time by 5.28% on T1 and 4.69% on T2. Per-second GPU process telemetry recorded no other compute PID on GPU7 during either completed probe. Thus these particular timings are not explained by sharing GPU7. Other system-wide CPU/storage contention was not excluded. Full training uses 18,996 updates versus 3,360 for T2-only 80 epochs (5.65 times as many), and the CL loop additionally performs feature replay, replay PCE/Global, and feature-target capture. These differences explain why a 20–30 minute independent T2 run cannot be extrapolated directly to the whole sequence; no isolated timing attribution to each replay operation was measured.

## Protocol and limits

- Full data: T1 762 training slices, T2 166, T3 1,421; no slice reduction. Validation sizes are 153, 52 and 343.
- Current guarded CL implementation: SGD LR .03, momentum .9, optimizer weight decay 1e-4, polynomial LR exponent .9; batch 4, workers 8, CPU threads 4; PCE/Global/Spatial weights 1/1/.01, no gradient clipping; ZS-DER++ alpha/beta .5/.5, reservoir capacity 128, replay minibatch 4; seed 42, deterministic cuDNN, benchmark disabled, CuBLAS workspace `:4096:8`, deterministic algorithms warn-only. The existing grid-sample backward warning remains.
- This measures the existing continual runner, **not** the Domain-reference independent adapter. It does not establish numerical or optimizer parity with the corrected independent implementation. It is not authorization to launch formal CL before that implementation choice is settled.
- For timing only, Spatial was forced off or on throughout a short one-epoch task. The short LR schedule reaches zero after 12 updates; this is not a formal training schedule. Spatial-on computation and nonzero Spatial losses were confirmed in both tasks.
- The replay reservoir was still filling during T1/T2. Mean sampled replay task-group counts were 1, 1.75 and 2.375 in off-mode T1/T2/T3; a mature full-run buffer can have a different mixture. Eight measured updates per condition provide an approximate estimate, not precise long-run characterization.
- Step timers start before batch retrieval and end after synchronized feature-target capture and buffer insertion. They exclude evaluation, checkpoint exports, iterator startup and a small post-step logging tail. Full validation was separately timed.
- Selected paired model/replay state was copied to CPU RAM and restored through the existing runner; final tensor exports were suppressed in the timing harness. This is diagnostic-only and must not be used as a formal checkpoint policy.
- An earlier disk-writing probe was terminated after extended file/page waits near stage transitions. Its timings are excluded from the projection and partial outputs are preserved. It did not provide a controlled checkpoint-write cost. Only its confirmed benchmark process and orphaned telemetry were stopped; other jobs were preserved.

## Reproduction and public scope

[Scalar timing measurements and calculation](../results/organ_cl_throughput_20260908/summary.json). The public harness parameterizes the private data paths used by the executed copy and clarifies that the recorded buffer size is before the last insertion; timing logic is unchanged. Use the guarded source tree in this branch and an existing compatible environment and annotations:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python benchmark_organ_cl.py --source "$SOURCE" --data-root "$DATA_ROOT" \
  --sparse-root "$ANNOTATIONS/fg20" --output "$NEW_OUTPUT_OFF" \
  --physical-gpu 7 --spatial off
```

Repeat with a different output directory and `--spatial on`. Each run asserts the expected update counts. Do not use its temporary model as a selected scientific or display checkpoint.

Published: harness, aggregate timings and this report. Private: raw GPU process telemetry, medical images, scribble arrays, patient-level validation logs and partial checkpoint tensors. The paused data-halving script is outside this delivery.
