# T2 full-training repeatability check after execution-setting correction

Status: **both runs completed successfully on 2026-09-08 by 16:48:30 Asia/Shanghai**, after approximately 22 minutes. Each completed 80 epochs / 3,360 updates. Both worker exit codes and the coordinator exit code are 0. All recorded training losses were finite; best and last checkpoints were present and nonempty (approximately 103 MB each). No additional experiments were launched at closeout.

## Completed results and agreement

| Model container | GPU | Best validation foreground Dice | Corresponding test foreground Dice | Selected epoch |
|---|---:|---:|---:|---:|
| DomainModel | 4 | .6251718906134406 | .4905944282927692 | 69 |
| OrganModel | 5 | .6251718906134406 | .4905944282927692 | 69 |

The two logs match exactly for all 80 recorded epoch-level total, PCE, Global and Spatial losses. The complete recorded validation Dice curves also match exactly. Maximum absolute differences are 0 for each of these quantities, with no first divergent logged epoch. Spatial first activated at epoch 30 in both runs and was active for 51 epochs. Both validation-selected checkpoints have the same reported validation and test foreground Dice and foreground prediction fractions.

This extends the earlier eight-update diagnostic to agreement across the full set of recorded 80-epoch training and validation metrics on GPUs 4/5. It does not claim every parameter tensor matched at every update: those tensors were not captured and compared throughout the full runs. The warn-only deterministic setting also does not guarantee identical future runs on arbitrary hardware or environments.

The result supports the execution-setting mismatch as the explanation for the prior apparent container effect in this single-task experiment. The Domain and Organ wrappers no longer differ in the measured outcomes after the settings were aligned. This is not a claim that multi-task Organ-CL forgetting has been solved; these are independent UCL/T2 runs using the reference Domain-D annotations and training implementation.

For context, the earlier failed Organ recipe gave validation/test .0710013605/.0495519224; the present reference recipe gives .6251718906/.4905944283. Several settings and the annotation protocol differ between those recipes, so this is not an isolated estimate of the effect of determinism, Spatial, LR or annotation coverage on the original .07 result. Likewise, the previous epoch-11 vs epoch-30 comparison was not run with matched deterministic controls and does not establish an optimal Spatial activation time.

Public aggregate metrics, completion checks, full validation curves and curve-difference summaries: [completion.json](../results/organ_t2_deterministic_rerun_20260908/completion.json). Data, annotation arrays, checkpoint tensors and patient-level outputs remain private on NAS.

Following the user's instruction to continue, the DomainModel and OrganModel baseline pair is rerun with the execution settings validated in the [eight-update diagnostic](organ_t2_container_diagnostic_20260908.md). That diagnostic found divergence even in Domain-vs-Domain repetitions with determinism disabled, while both Domain repetitions and Domain-vs-Organ matched for all eight updates with the enabled settings. Historical run outputs and their runtime sources are preserved.

The corrected adapter sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing the reference/CUDA code, enables cuDNN determinism, disables cuDNN benchmarking, and enables deterministic-algorithm checking with `warn_only=True`. The settings are preserved in the runtime source and recorded in the final adapter metadata; both startup logs show the expected enabled-mode grid-sample warning. This is not strict determinism: the installed CUDA grid-sample backward has no deterministic implementation. The purpose of this pair is to measure whether the short-window agreement extends through full training on GPUs 4/5.

Exactly two runs are scheduled: `domain_control` on GPU4 and `organ_reference` on GPU5, each with 80 epochs / 3,360 updates. Both initialize from scratch with seed 42, use the same UCL split and Domain-D pattern_f5_b10 scribbles, LR .03, SGD momentum .9, manual gradient decay 1e-4, batch 4, eight workers, PCE/Global/Spatial weights 1/1/.01, and no gradient clipping. Spatial first activates at one-based epoch 30 (runner warmup 28). Validation foreground Dice selects checkpoints every epoch; test is evaluated once after checkpoint selection. Finite-gradient and updated-parameter guards remain enabled. No annotation sweep or full CL run follows automatically.

The only intended training-setting change from the [previous epoch-30 pair](organ_t2_spatial_start30_20260908.md) is the execution determinism configuration. Both use the same clean reference source archive, ScribbleCL commit `c25b9b7ce4e379a0d82cea50c4d632534c00ee57`. The completed comparison above reports exact agreement of recorded metrics rather than agreement within a tolerance. The earlier single-run score differences should not be interpreted as an architectural effect.

Runtime root: `/data_nas/jiangsuiyang/ScribbleCL/organ_T2_spatial30_deterministic_20260908`.

- tmux: `organ-t2-deterministic-20260908`.
- Runtime entrypoint: `source/run_t2_reference.py`.
- Launcher: `bash source/run_t2_spatial30.sh <runtime-root>`.
- Logs: `logs/domain_control.log`, `logs/organ_reference.log`, and `runs/<run>/train.jsonl`.
- Outputs: `runs/<run>/best.pt`, `last.pt`, `summary.json`, `manifest.json`, and `adapter.json`.
- Completion: `<run>.exitcode` and `coordinator.exitcode` in the runtime root.

Before launch, the actual NFS mount, 30 TiB free storage, a small write/read probe, and GPU4/5 free memory (24,121 MiB each) were checked. Existing processes are preserved. The background jobs do not require SSH or Codex to remain open. No automatic Codex monitoring was created. Only source and this prospective protocol are published at launch; medical data, annotations, checkpoints and patient-level outputs remain private.
