# Class-CL ZS-DER++ + MiB launch record

Date: 2026-09-02

## Implementation gate

- Method: `zs-derpp-mib`
- Backbone: ZScribbleSeg U-Net
- Current-task objective: sparse PCE with MiB unbiased background handling, ZS global consistency, and MiB knowledge distillation after the first task
- Replay objective: DER++ feature MSE plus sparse PCE and ZS global consistency under each sample's historical task/class space
- Smoke run: `c6smk3`, Class T1 to T2, two train batches per task
- Static/runtime audit: PASS; DER++ feature loss, replay PCE, replay global loss, MiB KD, and current-task ZS global loss were all exercised
- Sparse-label gate: PASS; T1 uses global labels 1 to 3, T2 uses 4 to 5, and T3 uses 6 to 7, with 0 for background and -100 for ignored pixels

## Validation-only sweep

Both candidates use seed 42, Class T1 to T2, 20 epochs per task, SGD learning rate 0.03, ZS global weight 0.1, DER++ buffer 64, replay minibatch 4, alpha 0.5, and beta 0.5.

| Run ID | MiB KD weight | Device role | Status at publication |
|---|---:|---|---|
| `c6m3` | 1.0 | sweep worker 1 | running |
| `c7m4` | 10.0 | sweep worker 2 | running |

The original runs `c6m1` and `c7m2` completed T1 but failed at the first T2 replay step because batch 4, replay minibatch 8, and the MiB teacher exceeded 24 GB GPU memory by 242 MiB. They are retained as failed-run evidence. Reducing only the replay minibatch to 4 leaves the buffer capacity and objective unchanged and lowers the observed training footprint to about 15.5 GB.

Selection uses the mean 3D validation Dice over all seen tasks after T2 (`final_seen_validation_mean`). Test results are not used for hyperparameter selection. After both candidates pass the output audit, the selected coefficient is automatically used by formal run `v6c43` for Class T1 to T3, 80 epochs per task.

## Related formal run

Organ-CL formal ZS-DER++ run `u5k2n` is active with seed 42, T1 to T4, and 80 epochs per task.

Completed metrics and the final Class-CL matrix will be added after the detached jobs finish.
