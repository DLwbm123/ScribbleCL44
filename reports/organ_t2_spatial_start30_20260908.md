# Organ T2: Spatial starts at epoch 30

Status: two fresh 80-epoch runs are running on GPUs 4/5. Startup checks confirmed both manifests have warmup 28 (first active epoch 30), LR .03, Spatial .01 and validation selection; both processes reached training and allocated 7,360 MiB each with no immediate error. This is a prospective timing contrast, not a completed result.

The requested "30" is interpreted as the first active epoch: Spatial is off for epochs 1-29 and has weight .01 for epochs 30-80 (51 active epochs). The reference condition `zero_based_epoch > warmup` therefore receives warmup 28. The adapter now exposes `--spatial-start-epoch`; its default remains 11, preserving prior launch behavior. An argument-boundary check verified starts 11 and 30 and their active-epoch counts 70 and 51.

Exactly two runs are launched in parallel on GPUs 4/5: `domain_control` with DomainModel and `organ_reference` with the reference OrganModel's first binary head. Both reuse the clean Domain shared-loop source at commit `c25b9b7ce4e379a0d82cea50c4d632534c00ee57` and the historical Domain-D UCL pattern_f5_b10 annotation. No annotation contrast or subsequent CL experiment is scheduled by this launcher.

All other settings match the [completed epoch-11 baseline](organ_t2_reference_recovery_20260908.md): fresh seed 42 initialization, 80 epochs / 3,360 updates, batch 4, LR .03, polynomial exponent .9, SGD momentum .9, optimizer decay 0 plus manual gradient decay 1e-4, PCE/Global weights 1/1, no gradient clipping, finite-gradient/parameter checks, and 8 workers. Validation foreground Dice selects the best checkpoint every epoch; test is evaluated only after selection. Best and last checkpoints are retained. CUDA training is not fully deterministic, so a difference from the earlier single run cannot be assigned entirely to activation timing.

Prior epoch-11 baseline:

| Model | Validation Dice | Test Dice at validation-selected checkpoint |
|---|---:|---:|
| DomainModel | .6242106280 | .4967082234 |
| OrganModel | .5532766640 | .4711945900 |

Runtime root: `/data_nas/jiangsuiyang/ScribbleCL/organ_T2_spatial_start30_20260908`. Logs are `logs/<run>.log` and `runs/<run>/train.jsonl`; outputs are `runs/<run>/best.pt`, `last.pt`, `summary.json` and `manifest.json`. Run and coordinator exit codes are written in the root. The tmux session is `organ-t2-spatial30-20260908` and is independent of SSH/Codex session lifetime.

The actual NAS mount, 30 TiB free capacity, a small write/read probe, and GPU4/5 free memory (24,124 MiB each) were checked before launch. Existing runs and other processes are preserved. Only code and the prospective protocol are published at launch; no datasets, annotations, model tensors or patient-level outputs are included.
