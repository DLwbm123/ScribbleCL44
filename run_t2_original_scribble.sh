#!/bin/bash
# One annotation-only contrast against the completed deterministic Organ baseline.
set -u
TASK_ROOT=/data_nas/jiangsuiyang/ScribbleCL/organ_T2_original_scribble_20260908
REFERENCE=/data_nas/jiangsuiyang/ScribbleCL/organ_T2_reference_recovery_20260908/reference_source
ADAPTER=/data_nas/jiangsuiyang/ScribbleCL/organ_T2_spatial30_deterministic_20260908/source/run_t2_reference.py
cd "$REFERENCE" || exit 1
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    /home/jiangsuiyang/anaconda3/envs/py38/bin/python -u "$ADAPTER" \
    --reference-source "$REFERENCE" \
    --data-root /home/jiangsuiyang/medical_continual_segmentation_data/CL_Benchmark/data \
    --annotation /data_nas/jiangsuiyang/ScribbleCL/organ_T2_coverage_20260908/annotations/fg20/organ/T2_v2_s2_seed42.npz \
    --model organ --spatial-start-epoch 30 --output "$TASK_ROOT/runs/organ_fg20" \
    > "$TASK_ROOT/logs/organ_fg20.log" 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > "$TASK_ROOT/organ_fg20.exitcode"
exit "$TASK_EXIT"
