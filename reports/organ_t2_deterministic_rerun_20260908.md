# T2 full-training repeatability check after execution-setting correction

Status: two fresh 80-epoch runs started on 2026-09-08 at approximately 16:26 Asia/Shanghai. Startup checks confirmed both manifests, training forward/backward execution, approximately 7.3 GiB GPU memory per run, and no immediate failure. Completion and full-trajectory agreement are not yet established.

Following the user's instruction to continue, the DomainModel and OrganModel baseline pair is rerun with the execution settings validated in the [eight-update diagnostic](organ_t2_container_diagnostic_20260908.md). That diagnostic found divergence even in Domain-vs-Domain repetitions with determinism disabled, while both Domain repetitions and Domain-vs-Organ matched for all eight updates with the enabled settings. Historical run outputs and their runtime sources are preserved.

The corrected adapter sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing the reference/CUDA code, enables cuDNN determinism, disables cuDNN benchmarking, and enables deterministic-algorithm checking with `warn_only=True`. The settings are preserved in the runtime source and recorded in the final adapter metadata; both startup logs show the expected enabled-mode grid-sample warning. This is not strict determinism: the installed CUDA grid-sample backward has no deterministic implementation. The purpose of this pair is to measure whether the short-window agreement extends through full training on GPUs 4/5.

Exactly two runs are scheduled: `domain_control` on GPU4 and `organ_reference` on GPU5, each with 80 epochs / 3,360 updates. Both initialize from scratch with seed 42, use the same UCL split and Domain-D pattern_f5_b10 scribbles, LR .03, SGD momentum .9, manual gradient decay 1e-4, batch 4, eight workers, PCE/Global/Spatial weights 1/1/.01, and no gradient clipping. Spatial first activates at one-based epoch 30 (runner warmup 28). Validation foreground Dice selects checkpoints every epoch; test is evaluated once after checkpoint selection. Finite-gradient and updated-parameter guards remain enabled. No annotation sweep or full CL run follows automatically.

The only intended training-setting change from the [previous epoch-30 pair](organ_t2_spatial_start30_20260908.md) is the execution determinism configuration. Both use the same clean reference source archive, ScribbleCL commit `c25b9b7ce4e379a0d82cea50c4d632534c00ee57`. At completion, compare epoch curves and selected scores across containers, record any first observed divergence, and distinguish full agreement from agreement within a tolerance. Do not interpret the earlier single-run score differences as an architectural effect.

Runtime root: `/data_nas/jiangsuiyang/ScribbleCL/organ_T2_spatial30_deterministic_20260908`.

- tmux: `organ-t2-deterministic-20260908`.
- Runtime entrypoint: `source/run_t2_reference.py`.
- Launcher: `bash source/run_t2_spatial30.sh <runtime-root>`.
- Logs: `logs/domain_control.log`, `logs/organ_reference.log`, and `runs/<run>/train.jsonl`.
- Outputs: `runs/<run>/best.pt`, `last.pt`, `summary.json`, `manifest.json`, and `adapter.json`.
- Completion: `<run>.exitcode` and `coordinator.exitcode` in the runtime root.

Before launch, the actual NFS mount, 30 TiB free storage, a small write/read probe, and GPU4/5 free memory (24,121 MiB each) were checked. Existing processes are preserved. The background jobs do not require SSH or Codex to remain open. No automatic Codex monitoring was created. Only source and this prospective protocol are published at launch; medical data, annotations, checkpoints and patient-level outputs remain private.
