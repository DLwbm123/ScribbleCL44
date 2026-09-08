# Organ T2 original-scribble contrast

Status: **completed on 2026-09-08 at 17:26:05 Asia/Shanghai**, with exit code 0. The run completed 80 epochs / 3,360 updates with finite recorded training losses. Spatial first activated at epoch 30 and was active for 51 epochs. Best and last checkpoints were present and nonempty (103,133,057 bytes each). No additional experiment was launched at closeout.

## Completed annotation contrast

The following checkpoints were selected by validation foreground Dice; test was evaluated after selection.

| Annotation protocol | Best validation Dice | Corresponding test Dice | Selected epoch |
|---|---:|---:|---:|
| Domain-D pattern_f5_b10 (existing baseline) | .6251718906 | .4905944283 | 69 |
| Original Organ-T2 (new run) | .5592428351 | .5312447371 | 31 |

Relative to the Domain-D scribbles, validation changed by -.0659290555 and test by +.0406503088. Thus the two splits give different rankings; this single-seed contrast does not establish that either annotation protocol is generally superior or that foreground coverage alone predicts performance. The protocols differ in foreground/background coverage and annotation placement. Changing annotations also changes model-dependent augmentation and subsequent training trajectories.

With the same original Organ-T2 scribbles, the earlier conservative/numerically repaired training recipe produced validation/test .0710013605/.0495519224, whereas the present reference recipe produces .5592428351/.5312447371. The original annotation file can therefore support substantially better learning; its low foreground coverage alone is not a sufficient explanation for the earlier .07 result. The older and current recipes differ in LR, clipping, losses, execution settings and implementation, so their individual contributions remain unidentified. This result is independent T2 acquisition, not a completed continual-learning retention test.

Public completion checks, metrics and validation curve: [completion.json](../results/organ_t2_original_scribble_20260908/completion.json). Execution metadata confirms the same enabled cuDNN determinism, disabled benchmarking, CuBLAS workspace and warn-only policy as the reference baseline. Raw data, annotation arrays, checkpoint tensors and patient-level outputs remain private on NAS.

The experiment changes only the training scribble file relative to the [completed deterministic Organ baseline](organ_t2_deterministic_rerun_20260908.md). Both use the same UCL images, dense validation/test labels, split, model container, runtime adapter, reference training source and GPU5. The baseline's validation-selected checkpoint achieved validation Dice .6251718906 and test Dice .4905944283 at epoch 69 with Domain-D pattern_f5_b10 scribbles.

This new run uses the persisted original Organ-T2 annotations (foreground coverage 19.6609%, annotation ID `organ_T2_nested_20260908_fg20`). It does not use the expanded 40% annotations. The original file was preserved in the prior coverage experiment and is reused without editing. Domain-D's scribbles had about 34.22% foreground coverage under the integer dense-label convention; the two protocols also differ in background coverage and spatial layout. This is a contrast of whole annotation protocols, not an isolated dose-response experiment for foreground coverage.

Fixed configuration: fresh initialization, seed 42, OrganModel with its first binary head, 80 epochs / 3,360 updates, batch 4, eight workers, LR .03 with polynomial exponent .9, SGD momentum .9, optimizer decay 0 plus manual gradient decay 1e-4, PCE/Global weights 1/1, Spatial weight .01 first enabled at one-based epoch 30 (warmup argument 28), and no gradient clipping. Finite-gradient/parameter guards, deterministic cuDNN, disabled cuDNN benchmarking, CuBLAS workspace `:4096:8`, and warn-only deterministic-algorithm checks remain enabled. The warn-only CUDA grid-sample limitation remains applicable.

Validation foreground Dice selects the checkpoint every epoch. Test is evaluated only after selection. The existing baseline is reused; no redundant Domain control, annotation expansion run, sweep or continual-learning sequence is scheduled. No checkpoint is loaded as initialization.

Runtime root: `/data_nas/jiangsuiyang/ScribbleCL/organ_T2_original_scribble_20260908`.

- tmux session: `organ-t2-original-scribble-20260908`.
- Launcher: `run_t2_original_scribble.sh`.
- Adapter: the unchanged runtime file under `organ_T2_spatial30_deterministic_20260908/source/`.
- Reference: the preserved source archive at ScribbleCL commit `c25b9b7ce4e379a0d82cea50c4d632534c00ee57`.
- Annotation: `organ_T2_coverage_20260908/annotations/fg20/organ/T2_v2_s2_seed42.npz` under the same NAS project root.
- Log: `logs/organ_fg20.log`; epoch metrics: `runs/organ_fg20/train.jsonl`.
- Outputs: `runs/organ_fg20/best.pt`, `last.pt`, `summary.json`, `manifest.json`, and `adapter.json`.
- Exit status: `organ_fg20.exitcode` in the runtime root.

Before launch, the actual NFS mount, 30 TiB free storage, GPU5 free memory (24,121 MiB), annotation readability and a small write/read probe were checked. Existing processes and results are preserved. Training runs independently of the SSH/Codex session; no Codex monitoring automation is created. Code and this prospective protocol are shareable; datasets, annotation arrays, checkpoints and patient-level outputs remain private.
