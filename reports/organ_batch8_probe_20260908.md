# Batch-8 feasibility probe — 2026-09-08

Outcome: **batch 8 with Spatial enabled failed with CUDA out-of-memory on physical GPU4 (RTX 3090, 23.56 GiB)**. One update completed; the second failed. The planned full-buffer measurement was not reached. No valid steady-state throughput, speedup or full-run ETA can be inferred from this failed attempt. The existing batch-4, 60-epoch formal run on GPU7 was preserved and remained active at the post-probe check.

The measured allocation peak was **21.5974 GiB**; PyTorch reservation peak was **21.8848 GiB**. At the failed allocation, the CUDA error reported **22.23 GiB** total process memory use, **1.32 GiB** free and an additional **1.69 GiB** requested. Only about 238 MiB was reserved but unallocated, so the error does not provide evidence that allocator fragmentation alone explains the failure. GPU4 was idle with 24,124 MiB free before launch, and telemetry saw no competing compute process during the probe. The error's logical `GPU 0` refers to physical GPU4 through CUDA_VISIBLE_DEVICES. These are observed memory values at failure, not an estimate of the unknown memory needed to finish all three tasks.

## Scope and controls

The probe used the exact guarded runner source deployed for the active 60-epoch run and its existing T1/T3 half-training views, with full T2. Current-image batch size changed from 4 to 8; **replay minibatch remained 4**, capacity 128. Seed 42, LR .03, SGD momentum .9/weight decay 1e-4, PCE/Global/Spatial weights 1/1/.01, workers 8, CPU threads 4, deterministic cuDNN, disabled cuDNN benchmarking, CuBLAS workspace `:4096:8`, and warn-only deterministic algorithms matched the prior throughput controls. Spatial was forced on immediately for the memory stress test. No mixed precision, gradient accumulation, replay reduction or other memory optimization was introduced.

The intended bound was one short epoch of 20 updates for each of T1/T2/T3, with timing restricted to updates where the replay buffer already contained 128 examples. In reality only the first T1 update completed, when the buffer was initially empty. The second update would use the newly populated replay buffer, and failed before its final insertion. No full-buffer condition and no T2/T3 stage were reached. Like the existing diagnostic harness, selected paired states would be copied to CPU RAM and final checkpoint exports skipped; this does not modify the formal run's persistent checkpoint policy. The recorded 6.23-second internal duration excludes interpreter/import startup.

## Reproduction and evidence

With the existing runtime, point `SOURCE` to the formal runner source and `SUBSET` to its previously generated views:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python benchmark_organ_cl.py --source "$SOURCE" \
  --data-root "$SUBSET/data" --sparse-root "$SUBSET/sparse" \
  --output "$NEW_OUTPUT" --physical-gpu 4 --spatial on \
  --batch-size 8 --train-batches 20 --max-task 3 --full-buffer-only
```

The process exited with code 1 and recorded the OOM in [timing.json](../results/organ_batch8_probe_20260908/timing.json). Harness syntax/help checks and the result's batch/replay/update-count assertions passed. The probe telemetry subprocess exited, and GPU4 memory was released. No further probe or formal-run reconfiguration was performed after this feasibility failure.

Published: benchmark controls, aggregate memory/failure metrics and this report. Private: medical images, scribbles, patient-level logs, checkpoints and raw GPU process telemetry. The failure is a memory-capacity observation for this implementation and precision setting, not a model-quality result or proof that batch 8 is impossible with a different implementation.
