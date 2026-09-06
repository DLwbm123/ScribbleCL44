# Continual sparse segmentation project

This checkout contains the current public Class-CL and Organ-CL runtime plus
aggregate, data-free experiment artifacts. Organ-CL follows left atrium,
prostate, liver, then brain tumor with one shared ZScribbleSeg U-Net backbone
and one frozen binary output head per observed task. Only `main.py` is a
supported training entry.

Current evaluations use an unweighted patient/class macro-average over
background class `0` and the task foreground classes. Foreground-only values
remain beside them to expose failures hidden by easy background.

See the [current complete experiment audit](reports/current_experiment_audit_20260906.md)
for Domain-CL, Class-CL, and Organ-CL status, results, limitations, and questions
for independent GPT Pro review.

Example static task-1 gate:

```bash
python main.py --setting-run \
  --data-root <benchmark-root> --sparse-root <sparse-root> \
  --output <anonymous-output> --max-task 1 --device cuda:0 \
  --epochs-per-task 80 --method pce-sequential
```

Omit `--max-task` for the complete T1 to T4 sequence. Supported methods are
`pce-sequential`, `zs-sequential`, `pce-ewc`, and `zs-ewc`. ZS methods enable
global consistency with weight 1 by default. Optional ZS ablation flags are
`--zs-global-weight`, `--zs-gd-loss`, `--zs-adversarial-perturbation`,
`--zs-spatial-loss-weight`, and `--zs-spatial-warmup-epochs`. EWC estimates
Fisher information from sparse PCE and regularizes only the shared backbone.
