#!/bin/bash
# Exactly two fresh 80-epoch runs; no subsequent experiments are scheduled.
set -u
TASK_ROOT="${1:-/data_nas/jiangsuiyang/ScribbleCL/organ_T2_spatial_start30_20260908}"
REFERENCE=/data_nas/jiangsuiyang/ScribbleCL/organ_T2_reference_recovery_20260908/reference_source
PYTHON=/home/jiangsuiyang/anaconda3/envs/py38/bin/python
cd "$REFERENCE" || exit 1
run_one() {
    local task_name="$1" task_model="$2" task_gpu="$3" task_exit
    CUDA_VISIBLE_DEVICES="$task_gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        "$PYTHON" -u "$TASK_ROOT/source/run_t2_reference.py" \
        --reference-source "$REFERENCE" \
        --data-root /home/jiangsuiyang/medical_continual_segmentation_data/CL_Benchmark/data \
        --annotation /home/jiangsuiyang/medical_continual_segmentation_domain_gptpro/data/sparse_annotations_pattern_f5_b10/domain/D_v2_s2_seed42.npz \
        --model "$task_model" --spatial-start-epoch 30 --output "$TASK_ROOT/runs/$task_name" \
        > "$TASK_ROOT/logs/$task_name.log" 2>&1
    task_exit=$?
    printf '%s\n' "$task_exit" > "$TASK_ROOT/$task_name.exitcode"
    return "$task_exit"
}
run_one domain_control domain 4 &
DOMAIN_PID=$!
run_one organ_reference organ 5 &
ORGAN_PID=$!
wait "$DOMAIN_PID"
DOMAIN_EXIT=$?
wait "$ORGAN_PID"
ORGAN_EXIT=$?
TASK_EXIT=0
if [ "$DOMAIN_EXIT" -ne 0 ] || [ "$ORGAN_EXIT" -ne 0 ]; then TASK_EXIT=1; fi
printf '%s\n' "$TASK_EXIT" > "$TASK_ROOT/coordinator.exitcode"
exit "$TASK_EXIT"
