# Organ T2: recover the existing UCL independent baseline

Status: first-step parity passed; two baseline runs launched on GPUs 4/5 and reached the training forward path with approximately 7.2 GiB GPU memory each. No completed recovery result is claimed.

The Domain-D and Organ-T2 UCL H5 images, dense labels, and patient split arrays were compared directly on the server and match. Loading the historical Domain-D formal checkpoint into the Organ model reproduced validation foreground Dice 0.5988888759 and test Dice 0.5370209892. That historical checkpoint was selected on test for a platform demo. It is not a held-out baseline.

The failed Organ independent run used LR .003, Global .1, Spatial 0, gradient clipping at 1, and different scribbles. Its validation/test Dice were .0710013605/.0495519224. Foreground occupied 40.4083% of its validation predictions. Clipping affected 96.0714% of updates in the first 20 epochs and 47.3810% across all 80. These changes were not a matched reproduction of Domain-D.

## Shared implementation and bounded protocol

`run_t2_reference.py` directly imports the preserved Domain runner at ScribbleCL commit `c25b9b7ce4e379a0d82cea50c4d632534c00ee57`. It uses that checkout's `DomainModel` or `OrganModel` and calls its independent shared stage loop, with UCL mapped to Organ task T2. The existing reference implementation provides the optimizer, losses, augmentations, spatial branch and evaluation; no loop is copied. This is an Organ T2 adapter, not a claim that the separate numerically repaired Organ runner is now step-equivalent. Original dispatcher manifests remain intact; `adapter.json` records model and annotation provenance.

All runs: seed 42, 80 epochs, 166 slices, batch 4, 42 steps per epoch, LR .03 with polynomial exponent .9, momentum .9, optimizer decay 0 plus manual gradient decay 1e-4, PCE/Global/Spatial 1/1/.01, Spatial first active at one-based epoch 11, 8 workers. Gradients are not clipped. Non-finite gradients or updated parameters stop the run; the guard does not alter finite updates. Validation selects the checkpoint every epoch; test evaluation occurs only after selection. Consequently the selection policy differs from the historical platform demo, consistently across every new run.

| Phase | Run | Model container | Annotation |
|---|---|---|---|
| 1 | domain_control | DomainModel | Historical Domain-D pattern_f5_b10 |
| 1 | organ_reference | OrganModel, first binary head | Same Domain-D annotation |
| 2 | organ_fg20_reference_recipe | Same OrganModel | Organ foreground coverage 19.6609% |
| 2 | organ_fg40_reference_recipe | Same OrganModel | Organ foreground coverage 40.4353% |

Phase 1 runs in parallel. Phase 2 starts only if both jobs complete all 3,360 steps, each best validation Dice is at least .45, and their absolute validation gap is at most .05. These predeclared thresholds are a practical recovery gate, not a significance test or a guarantee of equivalence. A failed gate stops the pipeline for review; it does not trigger a retry, another sweep, or CL training. At most four 80-epoch runs are allowed. Phase 2 changes only annotation input relative to organ_reference.

GPUs 4-7 may be shared with other processes when at least 12 GiB is free. Preferred assignments are 4/5 for phase 1 and 6/7 for phase 2, with fallback to another sufficiently free authorized GPU. No existing process is stopped. Runtime data and outputs remain on NAS.

## First-step validation

The existing Domain parity capture helper was reused with one real augmented UCL training batch. Domain, repeated Domain and Organ passes had zero differences in initial parameters, loss, gradients and updated model state after output-head key normalization. PCE was .8066117167, Global .1874822974 and total loss .9940940142 in all three passes. The capture helper uses synthetic loader length 76 for both sides; this check does not validate a complete learning-rate trajectory. Actual production loaders have 42 batches and the pipeline verifies 3,360 completed steps. Spatial is inactive at the checked first step. CUDA grid-sample backward can remain nondeterministic; a passing fixed-batch check does not guarantee identical full training trajectories.

## Runtime

- NAS root: `/data_nas/jiangsuiyang/ScribbleCL/organ_T2_reference_recovery_20260908`
- `reference_source/`: clean archive of the explicit reference commit.
- `source/`: adapter and bounded pipeline.
- `checks/parity.json`: scalar parity evidence; capture checkpoints remain private.
- `pipeline.json`: phase and aggregate results.
- `logs/<run>.log`, `runs/<run>/train.jsonl`: execution and epoch logs.
- `runs/<run>/best.pt`, `last.pt`: private checkpoints.
- tmux session: `organ-t2-reference-20260908`.

The NFS mount, 30 TiB free capacity and a small write/read probe passed before creating outputs. No data, annotation arrays, model tensors, patient-level metrics or credentials are part of the public delivery. This launch does not schedule Codex monitoring; query live progress when requested.
