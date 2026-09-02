# Domain-CL joint-training short convergence study

## Outcome

Joint ZScribbleSeg training converged within five epochs. The selected recipe (`lr=0.04`, batch size 4) achieved **0.5662 mean validation Dice** and **0.5628 mean A-F test Dice**. Its loss decreased monotonically from 0.3166 to 0.1877.

The 200-step learning-rate gate showed that increasing the rate without limit was harmful: `lr=0.10` collapsed to 0.0141 mean test Dice. The useful short-run range was 0.03-0.04, with 0.04 consistently strongest in the full-epoch comparison.

## Protocol

- Scenario: Domain-CL tasks A-F, pooled for joint training.
- Method: ZScribbleSeg U-Net with sparse PCE and global consistency weight 1.0.
- Seed: 42.
- Batch size: 4; 2,266 training slices; 567 batches per full epoch.
- Selection: equal-weight mean A-F validation Dice; test data was not used for checkpoint selection.
- Runtime: Python 3.10, PyTorch 2.2.1+cu121.
- Audit: all eight anonymous runs passed the output-contract audit.

## Full-epoch comparison

| Run | LR | Epochs | Best val | Test mean | A | B | C | D | E | F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r8c4w | 0.02 | 3 | 0.4261 | 0.3730 | 0.3027 | 0.4554 | 0.4968 | 0.2337 | 0.4746 | 0.2750 |
| t5n9b | 0.03 | 3 | 0.4876 | 0.4858 | 0.4123 | 0.5003 | 0.5702 | 0.5618 | 0.5144 | 0.3560 |
| u2f7k | 0.04 | 3 | 0.5228 | 0.5414 | 0.4458 | 0.5911 | 0.6139 | 0.6129 | 0.5480 | 0.4365 |
| w6d3s | 0.03 | 5 | 0.5287 | 0.4998 | 0.4635 | 0.5375 | 0.5723 | 0.5386 | 0.5812 | 0.3056 |
| y9h4m | 0.04 | 5 | **0.5662** | **0.5628** | **0.6163** | 0.5365 | **0.6602** | 0.5964 | **0.6039** | 0.3635 |

## Interpretation

This is a convergence and implementation-validity check, not a final multi-seed result. Five epochs are sufficient to show that the joint path, sparse-label loader, optimizer, validation selection, checkpoint reload, and A-F evaluation work together. The next formal upper-bound run should retain `lr=0.04` and use the required longer schedule; no additional high-LR sweep is justified by these results.

Raw medical data and checkpoints are excluded. The CSV files contain the complete anonymous aggregate metrics and epoch curves needed to reproduce this report.
