# Bounded Class EWC T2 diagnostic

Status: launched on 2026-09-11 at 22:22 Asia/Shanghai; startup checked at 22:24. These are diagnostic runs, with no completed diagnostic results yet and no automatic formal continuation.

The completed EWC comparison lost its T1 foreground predictions after T2. Its T2 penalty with lambda=1 was approximately 0.000002 to 0.000039, while supervised losses were much larger. This motivates testing regularization strength, without assuming the coefficient is the only cause of forgetting.

| Job | EWC lambda | New training | Startup status |
|---|---:|---|---|
| q01 | 1 | T2, 10 epochs | Running |
| q02 | 10000 | T2, 10 epochs | Running |
| q03 | 1000 | T2, 10 epochs | Queued |

All jobs restore the selected T1 model and its matching Fisher from the completed zs_ewc job in class_comparisons_20260910_neutral_restart. This saves retraining T1. The source T1 validation foreground Dice is 0.717723. Model and anchor equality, Fisher scope and finite values, and preservation of the requested new coefficient passed CPU checks. The maximum source Fisher value is 0.0001022673.

Only lambda changes between the three arms. They use the same seed 42, full 1500-slice T2 training set, batch 4, workers 4, SGD LR 0.03, momentum 0.9, weight decay 0.0001, EWC gamma 0.1, PCE=1, Global=0.1, and Spatial=0. The original 80-epoch polynomial learning-rate budget is retained, but execution stops after 10 epochs (3750 updates). All arms reset to the same seeded setup; this is a controlled resumed comparison, not an exact continuation of the original random-number stream.

T1 and T2 validation foreground and background-inclusive Dice are recorded every epoch. The diagnostic checkpoint is selected by the mean of T1/T2 foreground validation Dice. The normal current-T2-validation checkpoint is also preserved separately. The new runs do not evaluate the test set. This distinguishes better retention from simply preventing T2 learning.

The shared Class training implementation now supports --class-t2-from and --diagnostic-epochs. No duplicate training loop is introduced. The existing formal source snapshot and its active processes remain intact. Two new jobs run concurrently on GPU2 with a 14000 MiB admission threshold; q03 follows a completed slot. Both first steps passed with finite loss, 96 restored Fisher tensors, and approximately 6254 MiB peak CUDA reservation each. Main and worker process command lines and GPU names were neutral.

For reproduction, copy probe_job_entry.py to a file named main in each q01/q02/q03 directory. Supply config.json with source pointing to the Class runtime directory and argv containing the existing run_class_job.py options plus --method zs-ewc, --epochs-per-task 80, --class-t2-from pointing to the completed source job, --diagnostic-epochs 10, --max-task 2, --validation-only, and the corresponding --ewc-lambda. Use --validate-every 375 and the fixed settings above. Launch the neutral entry with python -u main. The optional run_ewc_probe.py coordinator, installed as main in the parent run directory, reads plan.json containing jobs with name/ewc_lambda and a short NAS tmpdir.

Source copying initially encountered a NAS extended-attribute error. Ordinary write/read probes passed; copying file contents without extended attributes resolved deployment. Data, models, raw logs, patient-level metrics and private filesystem locations are excluded from this report.

[Shared runtime](../class_runtime/runner_core.py), [resume test](../class_runtime/test_class_ewc_resume.py), [coordinator](../class_runtime/run_ewc_probe.py), [neutral job entry](../class_runtime/probe_job_entry.py).
