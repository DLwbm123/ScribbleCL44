# ZS-DER++ Domain-CL A-to-B parameter sweep

## Result

The best validation-selected configuration was:

- feature replay weight `alpha=1.0`
- replayed sparse PCE plus global-consistency weight `beta=0.5`
- current/replay global weight `1.0`
- buffer size `64`
- replay minibatch size `8`

It achieved **0.6238 A/B mean validation Dice** at B epoch 15. The post-selection A/B test mean was **0.6369** (A 0.6232, B 0.6506), with BWTR -0.0247.

## Protocol

- Domain-CL A-to-B pilot, 20 epochs per task, seed 42.
- ZS-DER++ uses feature MSE on the shared U-Net decoder feature plus sparse PCE and global consistency on replay samples.
- Learning rate 0.03 and training batch size 4 were fixed.
- Hyperparameters were ranked only by B-stage A/B mean validation Dice.
- Seven completed runs passed the output-contract audit.
- The replay-minibatch-16 candidate exceeded 24 GB GPU memory and was excluded as infeasible; it was not rerun with a different training batch size.

The selected configuration is the recommended candidate for the 150-epoch A-to-F formal run. The full anonymous result table is in `results/zs_derpp_ab_sweep_20260902/metrics.csv`.
