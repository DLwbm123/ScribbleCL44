# Organ-CL matched formal continuation controls

Three additional formal continuations were launched on GPUs 4–6 alongside the existing GPU 7 run. These are within-method ablations / coefficient comparisons, not cross-method EWC/GPM benchmark results. Startup verification found all three wrappers alive with 24/23/24 successful updates respectively and no exit markers. All are running; no final performance is claimed.

| GPU | Run | T2 feature alpha | T2/T3 Spatial | Purpose |
|---|---|---:|---|---|
| 4 | `alpha0` | 0 | 0.01 from epoch 30 | Full-run feature replay ablation |
| 5 | `alpha01` | 0.1 | 0.01 from epoch 30 | Stronger small-alpha alternative |
| 6 | `no_spatial` | 0.05 | Disabled | Spatial ablation after the shared T1 starting point |
| 7 | Existing `formal_small_alpha_20260908` | 0.05 | 0.01 from epoch 30 | Selected reference; left running |

All use the same completed T1 validation-selected paired model/replay state (`run60/s01_state.pt`), with T1's training and selected parameters untouched. T1 was trained for 60 epochs, selected at epoch 17. T2 begins afresh with declared transition seed 43; it does not resume a 10-epoch sweep checkpoint. Each continuation trains T2 and T3 for 60 epochs. Data counts remain T1 381 (half), T2 166 (full), T3 711 (half).

Shared controls: native Organ ZS-DER++ loop, seed 42, batch 4, workers 8, task-initial LR 0.03, SGD momentum 0.9 / weight decay 1e-4, original PCE/Global weights 1/1, replay buffer 128 and minibatch 4. T2 uses supervised replay beta 0.5, gradient clip 5, and training-image-only T2-head BN calibration before checkpoint-selection validation. Alpha-zero disables the weighted feature loss, while replay forwards, sampling, and buffer updates remain active; it is not a complete no-replay baseline.

T3 restores the original alpha/beta 0.5/0.5 and original clipping setting. Its first-epoch T2 validation/test retention is recorded, then training continues. Numerical debug guards remain enabled. The T3 feature-replay configuration has not yet been established as numerically stable in a full run. Checkpoint selection uses validation; test metrics are diagnostic only.

The Spatial ablation affects **T2 and T3**, not the already-trained T1. Until T2 epoch 30 its configured active losses match the GPU 7 reference. It is an ablation of Spatial in this continuation, not a claim that T1 was trained without Spatial. CUDA grid-sampling backward is not fully deterministic, so exact numerical equality between repeated runs is not assumed.

The launcher derives every command from the actual GPU 7 launch record, changing only output/annotation tag and the named alpha/Spatial controls. Training code remains `65ab7ef8783a687ac9787f5419563f5491422b4e`, using the same saved source directory as GPU 7. No training loop was duplicated or edited for these controls. Script: `launch_organ_formal_controls.py`; `--dry-run` checks and prints the exact derived commands without launching.

Remote root: `/data_nas/jiangsuiyang/ScribbleCL/organ_T13_half_cl_20260908/formal_controls_20260908/`.
- Outputs: `alpha0/`, `alpha01/`, `no_spatial/`.
- Logs: `logs/alpha0.log`, `logs/alpha01.log`, `logs/no_spatial.log`.
- Exact commands, GPUs, wrapper PIDs and source: `launch.json`.
- Each detached shell writes its own `<run>.exitcode` on exit; runs do not depend on the SSH or Codex session staying open.

Preflight found approximately 24 GiB free on each selected GPU and 30 TiB available on the NAS filesystem. The launcher required at least 16,000 MiB free per selected GPU, 60 GiB free storage overall, and a successful small write/read probe. It does not stop or change existing processes. Public artifacts contain source, configuration and startup evidence only; patient data, checkpoint files, private numerical snapshots and large raw logs remain on NAS. Completed results will be verified and published when completion is next checked.
