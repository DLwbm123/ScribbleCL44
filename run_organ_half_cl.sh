#!/usr/bin/env bash
# Full T1/T2/T3 run using the real guarded loop and persisted checkpoints.
set -euo pipefail
RUN_ROOT=${1:?run root required}
PYTHON=${2:?existing Python interpreter required}
GPU=${3:-7}
case "$GPU" in 4|5|6|7) ;; *) echo 'GPU must be 4–7' >&2; exit 2;; esac
cd "$RUN_ROOT/source"
mkdir -p "$RUN_ROOT/logs"
trap 'TASK_EXIT=$?; printf "%s\n" "$TASK_EXIT" > "$RUN_ROOT/training.exitcode"' EXIT
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
"$PYTHON" -u -c '
import runpy
import torch
torch.set_num_threads(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
runpy.run_path("main.py", run_name="__main__")
' --setting-run --data-root "$RUN_ROOT/subset/data" --sparse-root "$RUN_ROOT/subset/sparse" \
  --output "$RUN_ROOT/run" --device cuda:0 --seed 42 \
  --epochs-per-task 80 --max-task 3 --batch-size 4 --workers 8 --lr .03 \
  --method zs-derpp --der-alpha .5 --der-beta .5 --der-buffer-size 128 --der-minibatch-size 4 \
  --pce-loss-weight 1 --zs-global-weight 1 --zs-spatial-loss-weight .01 --zs-spatial-warmup-epochs 28 \
  --validate-each-epoch --test-evaluation --t3-first-epoch-evaluation \
  --annotation-id organ_T13_half_train_seed42 \
  > "$RUN_ROOT/logs/train.log" 2>&1
