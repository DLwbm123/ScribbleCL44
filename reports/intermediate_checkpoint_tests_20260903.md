# Intermediate checkpoint tests

Date: 2026-09-03

Both tests load a stable best-validation checkpoint and evaluate every task seen by that checkpoint on its 3D test split. The evaluator checks checkpoint modification time and size before and after loading, and does not modify active training, checkpoints, or datasets. Test Dice values below are diagnostic only and were not used for checkpoint or hyperparameter selection.

| Setting | Checkpoint | Seen-task Dice | Seen mean |
|---|---|---|---:|
| Class-CL ZS-DER++ + MiB | `s02_best.pt` after T2 | T1: 0.7173; T2: 0.6771 | 0.6972 |
| Organ-CL ZS-DER++ | `s04_best.pt` during T4 | T1: 0.5352; T2: 0.0007; T3: 0.8534; T4: 0.7918 | 0.5453 |

## Interpretation

The Class-CL intermediate checkpoint retains both seen tasks at this point.

The Organ-CL checkpoint has severe forgetting of T2. Its persisted matrix independently confirms the pattern: T2 was 0.6637 immediately after T2, then 0.0018 after T3. The near-zero T2 score during T4 is therefore a method outcome, not an evaluator mismatch. The intermediate test did not terminate or alter the active Organ training run.

Raw medical data and checkpoints are excluded from the public repository.
