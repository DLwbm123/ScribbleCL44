# Class benchmark replay comparisons — running

The user replaced the proposed MiB component experiment with benchmark-style ER and feature-DER comparisons. The cancelled component plan did not launch any training.

Both new comparisons are running on GPU2. At 2026-09-12 14:34 CST, ER had completed two T1 epochs / 750 updates (loss 0.523402), and feature-DER had completed one T1 epoch / 375 updates (loss 0.397940). Both first real-data updates were finite. The process titles and child commands were checked to be neutral job/run names.

Each run trains T1/T2/T3 for 80 epochs from scratch, with Class's existing full splits and annotations. Seed 42, batch 4, LR .03, SGD momentum .9 and weight decay .0001, PCE/Global/Spatial=1/.1/0, reservoir 64 and replay minibatch 4. ER uses sparse replay PCE=1 and feature-DER uses feature MSE=.5; each disables the other's replay loss and replay-global loss. Neither uses MiB. Checkpoints are selected on current-task foreground validation; background-inclusive Dice and WCD are recorded for reporting.

The small cross-task checks passed both methods and exact cached/uncached data parity on synthetic inputs. Peak CUDA reserved memory was 8556/8676 MiB; production admission requires 16000 MiB free. H5 arrays are cached in host RAM to avoid repeated strided NFS reads. All large files, logs and temporary files stay on the designated remote-home mount.

NFS returned EBUSY during multiprocessing worker temporary-directory cleanup. These are finalizer warnings: finite training continued into subsequent epochs, and no restart was performed. They are retained in the private logs rather than described as a clean log.

The detached pipeline is already training; it does not depend on the Codex session and has no automatic experiment expansion. Source overlay and reproduction notes: class_replay_runtime/. Scalar evidence: results/class_replay_comparisons_20260912/startup.json. This report makes no claim that the experiments are complete.
