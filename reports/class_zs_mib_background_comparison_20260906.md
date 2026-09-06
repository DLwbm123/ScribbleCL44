# Class-CL ZS-MiB background-Dice comparison

Date: 2026-09-06

The completed seed-42 ZS-MiB stage-3 checkpoint was evaluated once on the
original dense MMWHS test sets. The same predictions were scored both with the
historical foreground-only patient/class macro-average and with background
class `0` included in that macro-average.

| Test task | Foreground only | Background Dice | Background included | Difference |
|---|---:|---:|---:|---:|
| T1: MYO/LV/LA | 0.773496 | 0.969612 | 0.822525 | +0.049029 |
| T2: RA/RV | 0.760469 | 0.962291 | 0.827743 | +0.067274 |
| T3: AO/PA | 0.674523 | 0.952174 | 0.767073 | +0.092550 |
| **A-Dice** | **0.736163** | — | **0.805781** | **+0.069618** |

The foreground-only A-Dice reproduces the archived `0.7362` result. Including
background increases the reported A-Dice because the measured background Dice
is between `0.9522` and `0.9696` across the three tasks.

This is a paired checkpoint re-evaluation, not a retrained result. The selected
checkpoint was originally chosen by foreground-only validation Dice. Future
runs use background-inclusive validation selection and report
`dice_includes_background: true` in their manifests and summaries.
