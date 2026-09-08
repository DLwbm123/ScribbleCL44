# Class-CL independent and positive-Spatial sweeps — 2026-09-08

Status: **sweeps started on my-gpu at 2026-09-08 20:17:01 Asia/Shanghai (12:17:01 UTC), GPUs2/3 only**. The initial jiangsuiyang Class coordinator and jobs have stopped, with all partial outputs preserved. The 24-GB-host channel-chunk Spatial change has been removed from the final source: my-gpu uses the original Spatial implementation on A100 40-GB cards. The original-Spatial three-stage real-data preflight passed, and the initial T1/T2 independent workers passed their startup checks; no completed sweep or formal result is claimed. The user authorized automatic formal training after validation-based selection. The existing Organ run on jiangsuiyang GPU7 is preserved.

## Bounded plan

| Group | Candidates | Sweep length | Selection | Formal run |
|---|---|---|---|---|
| Independent T1 | LR .01, .03 | 20 epochs each | Current-task validation Dice | One fresh 80-epoch run |
| Independent T2 | LR .01, .03 | 20 epochs each | Current-task validation Dice | One fresh 80-epoch run |
| Independent T3 | LR .01, .03 | 20 epochs each | Current-task validation Dice | One fresh 80-epoch run |
| Class ZS-DER++ + MiB | Spatial .001, .01 | T1/T2/T3, 40 epochs/task | Final seen-task validation mean; each task checkpoint restricted to epochs 35–40 | One fresh T1/T2/T3 run, 80 epochs/task |

Eight candidates total, followed automatically by four formal jobs (three independent models and one CL sequence). Ordinary sweep length is 20 epochs. Spatial starts at one-based epoch 35 of **each task** (warmup argument 33), so a 20-epoch sweep would never activate it. The Spatial trials therefore run to 40, providing six active epochs per task. The selection window prevents an entirely pre-Spatial checkpoint from deciding the coefficient. Formal checkpoints can be selected at any epoch; the selected epoch and whether it precedes Spatial will be reported at completion.

Formal epochs default to the existing Class baseline's 80; the optional question about 60 vs 80 had no response when this configuration was prepared, so the stated 80-epoch assumption is used. Full Class data are retained: each task has 1,500 training / 200 validation / 900 test slices. The Organ dataset-halving request is not applied to Class data.

Common controls: seed42, batch4, workers4, SGD momentum .9 and weight decay 1e-4, poly LR exponent .9, PCE1, Global .1, validation every375 updates (one full epoch). Independent runs use Spatial0 and no replay/distillation. CL uses LR .03, KD1, DER++ alpha/beta .5/.5, replay capacity64/minibatch4. KD1 is reused from the completed validation-selected KD sweep; no redundant ordinary CL sweep is added. Sweep test evaluation is disabled; formal test metrics are evaluated after validation checkpoint selection. No guarantee of exceeding the historical .6975 score is implied.

## Implementation and checks

The source snapshot is [class_runtime](../class_runtime/README.md), copied from the historical Class runner that produced `v6c43`, with independent-task and selection-window controls added to the same training loop. Each independent task starts a new process/model; T1 has background+3 foreground outputs, T2/T3 background+2. Sparse global labels map to local labels without changing background/ignore pixels; dense validation/test labels remain local. The synthetic mapping/head tests passed. Original data and scribbles are reused and passed shape/label inventory checks.

Real independent T3 smoke (two updates) completed. Initial CL Spatial smoke reached T1/T2 but failed at T3 with OOM because the neighborhood product expanded all eight channels. A channel-chunk modification passed numerical checks and a three-stage smoke there, but the user requested using my-gpu. The destination therefore restores the original all-channel Spatial code and checks it on A100 40 GB. The failed 24-GB smoke and its diagnostic patch remain preserved on the old host. Only this Class experiment is moved; active Organ code and processes are untouched. Raw smoke losses are diagnostics, not quality metrics.

Deterministic cuDNN/CuBLAS and warn-only algorithms are enabled, with the existing grid-sample limitation. The historical Class transition behavior is retained, including restoring selected model weights with the terminal replay state; no claim of matching the newer guarded Organ protocol is made. The published source and report distinguish the two implementations.

## Background execution and artifacts

Destination: `my-gpu`, NAS project root `/remote-home/wangbomin/ScribbleCL/class_independent_spatial_sweep_20260908`. Running tmux session: `class-sweeps-20260908`; A100 GPU pool2/3. GPU0 is explicitly excluded by the user; GPU1 is also outside this queue. The scheduler requires at least 32,000 MiB free before a new job and never terminates other processes or changes a batch size to bypass a failure. A failed candidate blocks only its group's formal launch; remaining groups continue. No Codex recurring monitor is created.

- `plan.json`: bounded candidates and formal settings.
- `events.jsonl`, `progress.json`: persistent progress and selections.
- `logs/<run>.log`, `<run>.exitcode`: individual run logs/status.
- `jobs/<run>/`: manifests, epoch logs, selected checkpoints and summaries.
- `best_models/T1.pt`, `T2.pt`, `T3.pt`: the three independent validation-selected model aliases after their formal runs finish.
- `pipeline.exitcode`: 0 only after all four formal jobs complete their checks.

Public scope is source, launch/selection logic and this prospective protocol. Datasets, scribble arrays, checkpoints and patient-level outputs are private. Final scalar results will be verified and published when completion is queried.

Destination environment: existing Python3.12.7 / Torch2.6.0+cu124, compared with Python3.10 / Torch2.2.1+cu121 on the original server. Runtime differences can change training trajectories; restoring the original Spatial formula does not establish bitwise cross-host equivalence. No environment reinstall was performed. MMWHS data were reused on the destination after shape/readability checks; only the three small original scribble archives were transferred.

The destination uses the already established my-gpu GCO compatibility wrapper, which dispatches to its installed `gco.cut_grid_graph` API with the same floating-point costs and swap algorithm. No package reinstall or Spatial-formula change is made. The first destination smoke failed at the old API call before training and is preserved.

The initial source-host Class jobs were stopped before migration; their partial outputs remain preserved and are not mixed with destination scores. The destination starts fresh with all eight scheduled candidates. Initial jobs are independent T1/T2 at LR .01; subsequent candidates and validation-selected formal jobs are queued. The runner and status times use UTC remotely; this report also gives Asia/Shanghai time.
