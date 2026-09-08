# Class independent and positive-Spatial experiments

This source snapshot comes from the historical Class `q2m8v` checkout used by run `v6c43` (ZS-DER++ + MiB). It is separate from the actively running Organ implementation. All training uses this snapshot's shared `main.py --setting-run` → `runner.py` → `runner_core.main('class')` loop; no independent training loop is introduced.

Additions for the current experiments:

- `--class-independent-task T1|T2|T3` initializes a fresh Class backbone with only that task's foreground block, maps global training scribble labels to local contiguous labels, and keeps dense validation/test labels local. It requires `zs-sequential`, so there is no teacher, replay or preceding-task checkpoint.
- `--validation-only` skips test and whole-heart evaluation during sweeps.
- `--selection-min-epoch` limits checkpoint selection to a one-based epoch window. Spatial pilots select from epochs 35–40; ordinary and formal runs select throughout training.
- The final my-gpu deployment uses the **original, unchunked Spatial implementation**. A channel-chunk variant was tested on the 24-GB server after an OOM; its numerical test passed, but it is not used for these migrated runs. That failed attempt and diagnostic variant are preserved on the original host.
- `run_class_job.py` sets deterministic cuDNN/CuBLAS and warn-only deterministic algorithms. The known grid-sample backward limitation remains. These execution settings and per-epoch validation are not asserted to reproduce every number of the older run, which validated every 200 updates.

`run_sweeps.py` accepts `--root`, `--data-root`, `--sparse-root`, `--formal-epochs` (default 80), `--gpus` (default 2 3), and `--minimum-free-mib` (default 32000). Place this directory at `$RUN_ROOT/source` and run the coordinator in tmux using the existing Python environment. The my-gpu launch uses A100 GPUs 2/3 (40 GB each), waits for adequate memory without stopping other jobs, checks candidate completeness, selects only by validation, and queues one formal job per independent task and one formal Class CL job. Any failed candidate blocks automatic formal launch for its group while other groups continue. The lock prevents accidental duplicate coordinators.

Exactly eight sweep candidates are scheduled: LR .01/.03 for each of three independent tasks, 20 epochs each; Spatial .001/.01 for Class T1/T2/T3, 40 epochs per task and Spatial starting at epoch 35. MiB KD=1 is reused from the already completed 20-epoch KD sweep. No additional KD or Global sweep is run. The formal jobs use the chosen LR/coefficient and the specified formal epoch count, with test evaluation after validation selection. Final independent model aliases are `$RUN_ROOT/best_models/T1.pt`, `T2.pt`, `T3.pt`; each points to its own formal run's `s01_best.pt`, not a shared model. Foreground global IDs are T1=(1,2,3), T2=(4,5), T3=(6,7); checkpoint outputs are local background+foreground channels.

Run the small contract check with `python test_independent_mapping.py`. Medical images, annotations, model states, patient-level logs and runtime paths stay outside Git. This snapshot retains the historical Class loop's model-checkpoint/terminal-replay transition behavior; it does not silently apply the later Organ replay-state repair.

The destination reuses the installed Python 3.12.7 / PyTorch 2.6.0+cu124 environment. The historical 24-GB host used Python 3.10 / Torch 2.2.1+cu121. Cross-host bitwise training parity is not claimed. The same existing MMWHS data are reused after count/readability checks; the three original annotation archives were transferred without creating a local dataset copy.

The destination uses the already established my-gpu GCO compatibility wrapper, which dispatches to its installed `gco.cut_grid_graph` API with the same floating-point costs and swap algorithm. No package reinstall or Spatial-formula change is made. The first destination smoke failed at the old API call before training and is preserved.

Publication cleanup removes an unused duplicate module and normalizes legacy trailing whitespace; the active runtime source is left unchanged.
