# ScribbleCL44

Domain continual segmentation experiments with sparse scribble supervision. The official training entry is `main.py`; all formal methods use a U-Net backbone.

## Joint-training upper bound

The `zs-joint` method pools the six Domain-CL training tasks and selects one checkpoint by the equal-weight mean A-F validation Dice. A short seed-42 convergence study reached **0.5628 mean test Dice after five epochs**.

```bash
python main.py --setting-run \
  --data-root /path/to/domain_data \
  --sparse-root /path/to/sparse_annotations \
  --output runs/<anonymous_run_id> \
  --device cuda:0 --seed 42 \
  --method zs-joint --zs-global-weight 1.0 \
  --epochs-per-task 5 --batch-size 4 --lr 0.04 \
  --workers 4 --validate-every 567
```

Inputs are external and are not published. The expected task files and sparse-annotation filenames are declared in `runner_core.py`.

- [Short convergence report](reports/joint_short_convergence_20260902.md)
- [Aggregate metrics](results/joint_short_20260902/metrics.csv)
- [Training curves](results/joint_short_20260902/curves.csv)
- [ZS-DER++ A-to-B parameter sweep](reports/zs_derpp_ab_sweep_20260902.md)
- [Organ-CL ZS-DER++ T1-to-T2 parameter sweep](reports/organ_zs_derpp_ab_sweep_20260902.md)

The tested runtime used Python 3.10, PyTorch 2.2.1+cu121, and CUDA 12.1. Install a CUDA-specific PyTorch wheel appropriate for the target machine.
