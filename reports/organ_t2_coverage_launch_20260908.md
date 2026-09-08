# Organ-CL T2 foreground coverage contrast

Four prospective runs compare the persisted T2 scribbles with a nested foreground expansion. This tests acquisition separately from retention. It does not replace T2 or tune against held-out test results.

| Run | Training | T2 foreground coverage | Budget |
| --- | --- | ---: | --- |
| ind_fg20 | T2 from scratch, ZS without replay | 19.6609% | 80 epochs |
| ind_fg40 | T2 from scratch, ZS without replay | 40.4353% | 80 epochs |
| cl_fg20 | T1 to T3, ZS-DER++ | 19.6609% | 80 epochs per task |
| cl_fg40 | T1 to T3, ZS-DER++ | 40.4353% | 80 epochs per task |

All four use the same shared `runner_core.py` training loop: seed 42, batch 4, SGD LR 0.003, momentum 0.9, optimizer weight decay 0.0001, polynomial LR exponent 0.9, finite gradient norm clipping at 1, PCE/Global/Spatial weights 1/0.1/0. The CL pair uses alpha/beta 0.5/0.5, reservoir buffer 128, replay minibatch 4. No spatial or GD branch is enabled. These are a new conservative recipe, not directly matched to historical LR 0.03 unclipped results.

Validation runs every epoch and selects the current task's best checkpoint. Model and replay state are restored together. Seen-task validation and test matrices are evaluated after each stage; test results are not used to choose checkpoints or configurations. Independent T2 uses the first binary head in a fresh OrganModel, without first training T1. It does not use the separate legacy independent Domain loop.

The foreground expansion reads only integer-converted training labels, grows toward the closest existing foreground scribble within the foreground mask, and preserves every original labeled pixel. Distance ties use seed 42. The target is at least 40% per foreground-bearing slice, so the aggregate is 40.4353%. This is generated supervision, not measured human annotation time. Neither validation nor test labels are used to construct scribbles.

| Quantity | Baseline | Expanded |
| --- | ---: | ---: |
| T2 foreground annotated pixels | 28,887 | 59,410 |
| Background annotated pixels | 1,115,520 | 1,115,520 |
| Integer dense foreground pixels | 146,926 | 146,926 |
| Total-pixel labeled fraction | 10.5194% | 10.8000% |
| Training slices | 166 | 166 |

T1/T3/T4 annotation files are reused unchanged. Baseline T2 is copied into the new experiment directory; source NPZ/H5 files are not edited. Protocol identifiers and coverage counts are retained in JSON sidecars and run manifests.

## Numerical evidence and launch gate

The existing post-step failure diagnosis found non-finite running-stat backbone output, finite batch-stat output, and a reconstructed first failing convolution whose FP64 output exceeded the FP32 range. This establishes an overflow risk in the diagnosed snapshot, not the exact first historical training cause and not the cause of T2 forgetting.

The new recipe preserves the existing replay/BN semantics and probability-space interpolation. Lower LR and clipping constrain finite updates. Non-finite losses, gradients, optimizer state, and feature targets fail rather than being replaced with zeros. CPU numerical and contrast tests passed (32 tests); DER++ state/BN smoke, real GCO exhaustive-energy checks, and the synthetic two-task end-to-end smoke passed.

The background controller starts the independent pair on GPU 4/5 and a 128-step real-data T1/T2 DER++ gate on GPU 6. Only a successful two-stage gate with nonzero feature replay promotes the two full CL runs on GPU 6/7. The gate is a short integration check, not proof of 80-epoch stability or retention. Existing GPU processes remain in place. All writes use the authorized NAS experiment directory; NFS mount, space, and write/read probe passed before launch.

## Reading the eventual results

Compare expanded minus baseline T2 validation Dice separately for independent acquisition, CL acquisition immediately after T2, and retention after T3. Report the absolute drop and retention ratio for each CL run. A single seed and two annotation levels do not establish a general dose-response curve. No completed formal result is claimed in this launch report.

Reproduction scripts: `prepare_t2_coverage.py`, `run_t2_coverage.py`; the latter records exact argv, GPU, PID, start/exit events in private runtime logs. Medical images, generated annotation arrays, checkpoints, and private diagnostic captures stay on the server.

## Shared-GPU memory correction

The first real-data gate with replay minibatch 8 completed two T1 epoch rows but hit CUDA OOM in T2 while another process remained on GPU 6. It produced no full CL result. The failed gate is preserved. Both CL runs now use replay minibatch 4; the independent runs keep their original settings and continue uninterrupted. A new `stability_gate_mb4` must pass before CL promotion. This is a shared-memory configuration adjustment, not a selection based on Dice. The controller enforces 18,000 MiB free before a minibatch-4 CL launch.
