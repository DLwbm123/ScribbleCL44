# ScribbleCL current experiment audit for GPT Pro

## Executive verdict

As of 2026-09-06, Domain-CL and Class-CL each have complete seed-42 evidence,
while Organ-CL does not have a completed formal result. The strongest current
background-inclusive scores are `0.7687` for formal Domain ZS-DER++, `0.8058`
for Class ZS-MiB, and `0.7793` for formal Class ZS-DER++ + MiB. The Organ value
`0.7683` is from an interrupted checkpoint and must not be presented as a
successful formal experiment.

All headline values below come from one inference pass per checkpoint, scored
with and without background class `0`. Every run uses seed 42; there are no
multi-seed error bars or significance tests.

## Current result ledger

| Scenario | Method | Run | Status | Epochs/task | Foreground-only Dice | Background-inclusive Dice | Reporting role |
|---|---|---|---|---:|---:|---:|---|
| Domain | ZS-GPM | `m7v2q` | Complete, 6/6 stages | 150 | 0.3247 | 0.6576 | Continual baseline |
| Domain | ZS-DER++ | `t4m7b` | Complete, 6/6 stages | 80 | 0.5437 | 0.7687 | Current formal result |
| Domain | ZS-Joint | `y9h4m` | Complete short run | 5 | 0.5628 | 0.7760 | Diagnostic upper bound only |
| Class | ZS-MiB | `zs_mib_global1_origscale_seed42` | Complete, 3/3 stages | 150 | 0.7362 | **0.8058** | Independent retained method |
| Class | ZS-DER++ + MiB | `v6c43` | Complete, 3/3 stages | 80 | 0.6975 | 0.7793 | Current formal replay result |
| Organ | ZS-DER++ | `u5k2n` | **Incomplete**, stopped during T4 epoch 21 | 80 | 0.5453 | 0.7683 | Diagnostic checkpoint only |

The Class methods have different schedules and objectives, so the observed
`0.0264` inclusive advantage of ZS-MiB over ZS-DER++ + MiB is descriptive, not
a controlled causal comparison. The five-epoch Domain Joint result is also not
a formal upper bound and must not be ranked as if it used the continual budget.

## Domain-CL

### Formal ZS-DER++ final checkpoint

| Domain | Foreground only | Background Dice | Background included |
|---|---:|---:|---:|
| A | 0.6608 | 0.9939 | 0.8273 |
| B | 0.7487 | 0.9954 | 0.8721 |
| C | 0.6055 | 0.9932 | 0.7994 |
| D | 0.2368 | 0.9945 | 0.6156 |
| E | 0.3750 | 0.9901 | 0.6826 |
| F | 0.6358 | 0.9951 | 0.8154 |
| **A-Dice** | **0.5437** | **0.9937** | **0.7687** |

The formal run used validation-selected A-to-B controls (`alpha=1.0`,
`beta=0.5`, buffer 64, replay minibatch 8, global weight 1.0). Its stored
foreground metrics are A-Dice `0.543748`, BWTR `-0.204754`, and E-FWT
`0.261256`. The negative BWTR and the low D/E foreground scores show that the
high inclusive Dice does not eliminate forgetting or domain-specific failure.

The current ZS-GPM baseline reaches `0.324679` foreground and `0.657588`
inclusive Dice. The five-epoch Joint diagnostic reaches `0.562822` foreground
and `0.776010` inclusive Dice, but is only a convergence check.

## Class-CL

| Method | Task | Foreground only | Background Dice | Background included |
|---|---|---:|---:|---:|
| ZS-MiB | T1 | 0.7735 | 0.9696 | 0.8225 |
| ZS-MiB | T2 | 0.7605 | 0.9623 | 0.8277 |
| ZS-MiB | T3 | 0.6745 | 0.9522 | 0.7671 |
| ZS-DER++ + MiB | T1 | 0.7382 | 0.9708 | 0.7964 |
| ZS-DER++ + MiB | T2 | 0.6957 | 0.9635 | 0.7850 |
| ZS-DER++ + MiB | T3 | 0.6587 | 0.9528 | 0.7567 |

ZS-MiB is a complete 150-epoch-per-task run and remains a separate valid
method. ZS-DER++ + MiB is the complete `v6c43` run selected from a validation-
only KD sweep: KD weight 1.0 reached validation mean `0.598135`, versus
`0.565759` for weight 10.0, before the 80-epoch formal run reached final
validation mean `0.742233`. Earlier sweep attempts with replay minibatch 8 ran
out of 24 GB GPU memory; they are excluded from result ranking.

## Organ-CL

| Task | Foreground only | Background Dice | Background included |
|---|---:|---:|---:|
| T1 | 0.5352 | 0.9919 | 0.7636 |
| T2 | 0.0007 | 0.9936 | 0.4971 |
| T3 | 0.8534 | 0.9878 | 0.9206 |
| T4 | 0.7918 | 0.9921 | 0.8920 |
| **Mean** | **0.5453** | **0.9914** | **0.7683** |

Run `u5k2n` has no final `s04.pt`, `s04_state.pt`, or `summary.json`; it stopped
during T4 epoch 21. T2 dropped from `0.6637` immediately after T2 to `0.0018`
after T3, and remains `0.0007` in the latest available T4 checkpoint. This is
severe catastrophic forgetting, not a successful Organ-CL result.

Subsequent retention gates did not remove the T2 collapse. Runtime audit found
and repaired replay-source, BatchNorm-state, checkpoint-state, and sparse-label
rotation issues, but post-repair Organ gates then failed numerically during T1.
The latest numerical-debug run `n9d3` captured non-finite replay feature targets
at `buffer_capture/feature_targets` (15,239,808 non-finite values; maximum finite
magnitude approximately `3.35e38`). Organ-CL is therefore blocked on numerical
stability before retention can be judged again.

## Metric interpretation

Including background raises every reported score because background Dice lies
between approximately 0.95 and 1.00. The masking effect is strongest when the
foreground fails: Organ T2 changes from `0.0007` foreground Dice to `0.4971`
inclusive Dice. For Domain and Organ, each task contains one foreground class,
so inclusive Dice is exactly the arithmetic mean of foreground and background.

Background-inclusive Dice is the requested headline definition. Foreground-
only Dice remains in this audit because removing it would hide clinically
relevant foreground failures. Existing checkpoints were selected by the older
foreground-only validation definition; the inclusive values are paired
retrospective evaluations and were not used for selection.

## Evidence cleanup and provenance

The current public tree removes superseded Domain artifacts that reported
ZS-GPM `0.1056` and ZS-DER++ `0.0945`, plus divergent Joint diagnostic outputs.
They were produced by earlier configurations and are replaced by `m7v2q`,
`t4m7b`, and the canonical short Joint result. Stale launch, intermediate,
sweep, and pre-latest-review reports were also removed from the current
Class/Organ tree. Git history retains recoverability; raw server experiments
were not deleted. ZS-MiB was explicitly retained as a separate valid method.

Public artifacts exclude medical data, sparse annotations, checkpoints, raw
logs, credentials, and private paths. They include manifests, matrices,
summaries, paired metric results, and this audit.

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Domain ZS-DER++ has a complete seed-42 result | `t4m7b` manifest, six-stage summary, final checkpoint re-evaluation | Supported |
| Class ZS-MiB and ZS-DER++ + MiB both have complete seed-42 results | Three-stage summaries and paired checkpoint re-evaluations | Supported |
| Organ ZS-DER++ is validated | No final summary; T2 collapse; numerical failures after repair | **Not supported** |
| Background-inclusive Dice is higher | Paired predictions for every retained headline checkpoint | Supported |
| One Class method is generally superior | One seed and unequal schedules/objectives | **Needs new controlled experiment** |
| ScribbleCL is robust across settings | Single seed, negative BWTR, Organ failure | **Not supported** |

## Reviewer-risk assessment

| Dimension | Current assessment |
|---|---|
| Contribution | Potentially interesting sparse-label CL evidence, but novelty must be argued separately from these runs. |
| Writing clarity | Protocol and status boundaries are explicit in this audit. |
| Experimental strength | Domain and Class are usable single-seed evidence; Organ is a documented failure. |
| Evaluation completeness | Missing multi-seed variance, matched Class budgets, and a successful repaired Organ gate. |
| Method soundness | Organ non-finite features and severe forgetting are unresolved high-risk issues. |

## Questions for GPT Pro

1. Should foreground-only Dice remain the primary segmentation metric, with
   background-inclusive Dice reported only as requested supplementary evidence?
2. Is any direct Class comparison defensible with 150 epochs/task for ZS-MiB
   and 80 epochs/task for ZS-DER++ + MiB, or is a matched-budget rerun mandatory?
3. Which claims remain defensible given one seed, Domain BWTR `-0.2048`, and the
   failed Organ extension?
4. What is the smallest decisive Organ diagnostic for the non-finite feature
   targets before another T1-to-T3 retention gate is justified?
5. Are the current public aggregate artifacts sufficient for reviewer-facing
   audit, without publishing medical data or checkpoints?
