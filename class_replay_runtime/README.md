# Class ER and feature-DER benchmark comparisons

This is a four-file overlay for the existing Class runtime. Copy the files from class_runtime/ into a campaign's source/ directory, then overlay these Python files. The training loop remains runner_core.main("class"). The prototype MiB component plan was cancelled before any training was launched.

Two fixed fresh runs: ZS-ER and ZS-DER, seed 42, tasks T1/T2/T3, 80 epochs per task, batch 4, workers 4, SGD LR .03, momentum .9, weight decay .0001. Current-image PCE/Global/Spatial = 1/.1/0. Reservoir capacity 64, replay minibatch 4. No extra sweep or automatic method expansion.

- ER: replay sparse PCE coefficient 1; feature and replay-global coefficients zero. Historical sparse labels are classified over all currently seen classes.
- Feature-DER: shared U-Net feature MSE coefficient .5; replay PCE/global coefficients zero. This follows the feature-based DER used by the benchmark implementation, rather than claiming a standard logit-DER reproduction.
- Shared-backbone BN statistics are frozen for the new controls' historical replay/feature capture. The existing Class augmented-image reservoir and task/source metadata are reused.
- Neither method uses MiB. Their objectives differ from the first task, so incompatible previously trained T1 models are not reused.
- These are scribble-supervised adaptations. The original benchmark buffer was 32; the present Class/Organ/Domain replay-control budget is 64.

Checkpoint selection remains current-task foreground validation Dice across all 80 epochs. Background-inclusive per-case Dice, full stage evaluations, whole-class Dice, fixed-model parameter counts, and original-source replay counts are retained for benchmark metrics. RMA uses the established common independent references when results are aggregated. Existing Class checkpoint behavior is preserved: selected model weights plus the stage-end reservoir are carried onward.

The H5 cache follows the existing Organ loading pattern: arrays are read into host RAM once per dataset instance and released on close. Synthetic-input checks assert exact cached/uncached image and label parity, including the global sparse labels and shifted validation labels of all three tasks. Both methods execute two updates per task across all three tasks, with finite-loss, active/disabled replay-branch, and historical-source assertions. These small checks are engineering tests, not scientific Dice results. A slow private-file fixture extraction attempt was stopped before training; its artifacts are not reused.

Provide private data/sparse paths in the campaign root's paths.json. All outputs, temporary files and caches stay on the designated remote-home mount. Start the existing Python interpreter with the neutral relative entry run.py. Runtime arguments are passed in environment; process titles are job/run. The detached coordinator waits for checks/passed.json and then admits both runs on GPU2 when measured memory plus headroom is available. No continuous Codex monitoring is configured.
