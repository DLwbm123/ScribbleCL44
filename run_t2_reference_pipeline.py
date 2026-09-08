"""Run matched containers first, then annotation-only contrasts after a recovery gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--domain-annotation", type=Path, required=True)
    parser.add_argument("--organ-annotation", type=Path, required=True)
    parser.add_argument("--expanded-annotation", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    assert json.loads((root / "checks/parity.json").read_text())["passed"]
    state = {"status": "baseline_running", "runs": [], "max_training_runs": 4,
             "selection_split": "val", "recovery_min_val": .45, "container_max_gap": .05,
             "automatic_CL_training": False}
    gpu_lock, reserved = threading.Lock(), set()

    def save():
        temporary = root / "pipeline.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "pipeline.json")

    def run(job):
        name, model, annotation, gpu = job
        with gpu_lock:
            rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits"], text=True).splitlines()
            free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows}
            candidates = [i for i in dict.fromkeys([gpu, 4, 5, 6, 7]) if i not in reserved and free[i] >= 12288]
            if not candidates:
                raise RuntimeError("No authorized GPU has 12288 MiB free; existing processes preserved")
            gpu = candidates[0]
            reserved.add(gpu)
        output = root / "runs" / name
        command = [sys.executable, "-u", str(Path(__file__).with_name("run_t2_reference.py")),
                   "--reference-source", str(args.reference_source), "--data-root", str(args.data_root),
                   "--annotation", str(annotation), "--model", model, "--output", str(output)]
        (root / f"{name}.command.json").write_text(json.dumps({"gpu": gpu, "command": command}, indent=2) + "\n")
        with (root / "logs" / f"{name}.log").open("x") as log:
            process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4"), stdout=log, stderr=subprocess.STDOUT)
            (root / f"{name}.pid").write_text(str(process.pid) + "\n")
            code = process.wait()
        with gpu_lock:
            reserved.remove(gpu)
        (root / f"{name}.exitcode").write_text(str(code) + "\n")
        if code:
            raise RuntimeError(f"{name} exited {code}; see log")
        summary = json.loads((output / "summary.json").read_text())
        assert summary["complete"] and summary["completed_epochs"] == 80 and summary["iteration"] == 3360
        assert summary["selection_split"] == "val" and not summary["test_for_selection"]
        record = summary["records"][0]
        return {"name": name, "model": model, "gpu": gpu,
                "validation_dice": record["best_validation"]["foreground_mean"],
                "test_dice": record["test"]["foreground_mean"]}

    save()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            baseline = list(pool.map(run, [("domain_control", "domain", args.domain_annotation, 4),
                                          ("organ_reference", "organ", args.domain_annotation, 5)]))
        state["runs"] += baseline
        scores = [row["validation_dice"] for row in baseline]
        if min(scores) < .45 or abs(scores[0] - scores[1]) > .05:
            state.update(status="stopped_at_recovery_gate", reason="Baseline recovery or container agreement failed; no annotation contrasts launched")
            save()
            return
        state["status"] = "annotation_contrasts_running"
        save()
        with ThreadPoolExecutor(max_workers=2) as pool:
            state["runs"] += list(pool.map(run, [("organ_fg20_reference_recipe", "organ", args.organ_annotation, 6),
                                                ("organ_fg40_reference_recipe", "organ", args.expanded_annotation, 7)]))
        state.update(status="complete", completed_at=time.strftime("%Y-%m-%d %H:%M:%S %z"))
        save()
    except Exception as error:
        state.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
