# T2 container discrepancy: controlled eight-update diagnostic

The fixed-batch parity check enabled deterministic execution settings, but the production adapter originally did not. This was an implementation gap in the experiment's repeatability controls. A controlled diagnostic reproduced early trajectory divergence even between two runs of the same DomainModel, eliminating the need for a model-container difference to produce that divergence.

## Executed diagnostic

Six short runs executed on physical GPU6: Domain baseline, Domain repeat, and Organ for each of two execution regimes. Each called the real preserved shared training loop for eight SGD updates, with the production 42-batch loader / 80-epoch learning-rate horizon and eight data workers. No validation or test samples were evaluated. Spatial was inactive throughout. Initialization, augmented input batches, PuzzleMix masks, gradients, and updated state dictionaries were compared in memory; no patient arrays or model tensors were exported.

All conditions used seed 42 and the historical Domain-D annotation. Both regimes fixed `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process startup, set four CPU threads, and disabled cuDNN benchmarking. The first regime left cuDNN determinism and deterministic-algorithm checking off; the second enabled cuDNN determinism and `torch.use_deterministic_algorithms(True, warn_only=True)`. The trace's `production_defaults` name refers to those nondeterministic PyTorch flags, not an exact reproduction of every original environment setting: the original launch did not explicitly set the CuBLAS workspace variable.

| Regime | Comparison | Initial max difference | First gradient max difference | State max difference after update 8 |
|---|---|---:|---:|---:|
| Determinism off | Domain repeat vs Domain | 0 | 1.6219914e-5 | .43002677 |
| Determinism off | Organ vs Domain | 0 | 4.7460198e-6 | .20514578 |
| Determinism on, warn-only | Domain repeat vs Domain | 0 | 0 | 0 |
| Determinism on, warn-only | Organ vs Domain | 0 | 0 | 0 |

State differences include BN running buffers, not just learned parameters; the table must not be described as a .43 weight change. Input image/annotation batches were identical for every compared update. In the enabled regime, losses, gradients, masks and updated model states matched exactly at every one of the eight updates, after renaming `heads.0.*` to `head.*`. In the disabled regime, the initial loss and PuzzleMix mask matched while gradients already differed. The Organ-vs-Domain PuzzleMix mask first differed at update 7, consistent with early numerical differences affecting later model-dependent augmentation. Domain-vs-Domain masks remained equal over these eight updates despite state divergence.

This establishes a concrete early divergence mechanism under uncontrolled execution; it does not quantify the fraction of the historical 80-epoch Dice gap explained by that mechanism or identify a specific CUDA kernel. Changing the execution controls removed divergence in the tested window. Neither model architecture nor Spatial activation is needed to reproduce the observed early divergence. The production logs also differed by epoch 1, before Spatial activation.

## Correction and remaining boundary

The adapter now applies the same settings before reference/CUDA initialization for ordinary training and parity checking, and records them in `adapter.json`. Historical runs and their runtime sources are preserved. No new 80-epoch training was launched as part of this diagnosis.

This is not a promise of strict end-to-end determinism: the installed CUDA `grid_sample` backward lacks a deterministic implementation and the mode is explicitly warn-only. Eight matching updates do not establish agreement over 80 epochs or across GPUs. The earlier single-run 11-vs-30 timing comparison therefore remains exploratory, rather than a clean estimate of activation timing alone.

Public evidence: [trace.json](../results/organ_t2_container_diagnostic_20260908/trace.json) and [diagnose_t2_container.py](../diagnose_t2_container.py). The diagnostic was invoked with `CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8`, the existing py38-named Python environment, and reference source commit `c25b9b7ce4e379a0d82cea50c4d632534c00ee57`.

Private runtime directory: `/data_nas/jiangsuiyang/ScribbleCL/organ_T2_spatial_start30_20260908/checks/container_trace`. Only scalar diagnostic evidence and source are published; raw medical data, model tensors and private captures remain excluded.
