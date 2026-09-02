# Organ-CL ZS-DER++ T1-to-T2 parameter sweep

Six seed-42 pilots were trained for 20 epochs per task. Hyperparameters were ranked only by the T2-stage mean T1/T2 validation Dice.

The selected configuration is:

- global-consistency weight `0.1`
- feature replay weight `alpha=0.5`
- replayed sparse PCE plus global-consistency weight `beta=0.5`
- buffer size `64`
- replay minibatch size `8`
- learning rate `0.03`, training batch size `4`

It achieved **0.5430 mean validation Dice** and a post-selection **0.5395 mean test Dice** (T1 0.5174, T2 0.5616). All six runs passed the output-contract audit.

The result also shows that the Domain-CL optimum should not be copied unchanged: Organ-CL strongly preferred global weight 0.1 over 1.0 in this pilot. This configuration is the recommended candidate for an 80-epoch-per-task T1-to-T4 formal run.

The complete anonymous table is in `results/organ_zs_derpp_ab_sweep_20260902/metrics.csv`. Raw medical data and checkpoints are not published.
