# T2 forgetting after one T3 epoch

This user-requested amendment supersedes the 20-epoch T3 sweep and automatic 80-epoch promotion. T1 and T2 retain their original 20-epoch training and validation selection. After T2, each candidate restores its saved paired model/replay checkpoint, initializes the T3 transition with seed 44, trains exactly one T3 epoch, evaluates T2 on validation, and stops. No formal training is launched automatically.

One candidate had already completed two T3 epochs when the request arrived, so its best checkpoint no longer represented epoch one. To avoid mixing historical and reconstructed trajectories, all four candidates use the same new transition-seed protocol. This is a fresh one-epoch T3 branch from the saved T2 state, not an exact replay of the original uninterrupted first epoch: the historical stage checkpoints do not contain RNG state. T1/T2 are not retrained.

The original 20-epoch T3 learning-rate horizon is retained. Stopping after one epoch therefore does not silently turn the first epoch into a full decay-to-zero schedule. Each run records the original horizon, executed T3 epochs, transition seed, and source run in its manifest.

The original sweep controller is paused to prevent obsolete automatic promotion. The new bounded controller waits only for each outstanding T2 stage checkpoint, stops that source worker, launches its one-epoch branch on the same authorized GPU, and exits after all requested evaluations. Source checkpoints and old T3 logs remain preserved.

Outputs are `runs/e1_<candidate>/t2_forgetting.json` and the aggregate `t3_epoch1_forgetting.json`. Each contains T2 Dice before T3, after one T3 epoch, absolute drop, and retention ratio. Completion checks require exactly one new epoch row (stage index 2, epoch index 0, 356 updates) and exact agreement of the inherited T2 validation baseline with the source stage record. No test split is read or used for selection.

Runtime entry: `run_t3_epoch1_stop.py`, using an isolated `epoch1_source` copy of the shared runner. This report records the amended launch protocol; it does not claim all four evaluations are complete.
