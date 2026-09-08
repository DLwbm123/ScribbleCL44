"""Four fixed validation-only configurations, then conditional paired formal runs."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    prior = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    base = next(e["command"] for e in reversed(prior) if e.get("run") == "cl_fg20" and "command" in e)
    lock = threading.Lock()
    def event(**values):
        with lock, (root / "sweep_events.jsonl").open("a") as f:
            f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S %z"), **values)) + "\n")

    def run(name, gpu, lr, global_weight, epochs, variant="fg20", formal=False):
        while int(subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                   "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip()) < 18000:
            time.sleep(30)
        command = list(base)
        for flag, value in [("--output", root / "runs" / name), ("--lr", lr),
                            ("--zs-global-weight", global_weight), ("--epochs-per-task", epochs),
                            ("--sparse-root", root / "annotations" / variant),
                            ("--annotation-id", "organ_T2_nested_20260908_" + variant),
                            ("--der-minibatch-size", 4)]:
            command[command.index(flag) + 1] = str(value)
        if not formal:
            command.remove("--test-evaluation")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        with (root / "logs" / (name + ".log")).open("x") as log:
            process = subprocess.Popen(command, cwd=Path(__file__).parent, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        event(run=name, status="started", gpu=gpu, pid=process.pid, command=command)
        code = process.wait()
        event(run=name, status="completed" if code == 0 else "failed", exit_code=code)
        return dict(run=name, lr=lr, global_weight=global_weight, exit_code=code)

    configs = [("sw_lr01_g0", 6, .01, 0.), ("sw_lr03_g0", 7, .03, 0.),
               ("sw_lr01_g01", 4, .01, .1), ("sw_lr03_g01", 5, .03, .1)]
    event(status="sweep_started", config_count=4, epochs_per_task=20,
          selection="validation only; fixed baseline annotation; no test access")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, name, gpu, lr, glob, 20) for name, gpu, lr, glob in configs]
        records = [future.result() for future in futures]
    eligible = []
    for item in records:
        if item["exit_code"] != 0:
            continue
        summary = json.loads((root / "runs" / item["run"] / "summary.json").read_text())
        if summary["completed_stages"] != 3:
            continue
        matrix = summary["validation_matrix"]
        acquired, retained, t3 = matrix[1][1], matrix[2][1], matrix[2][2]
        item.update(t2_acquired=acquired, t2_retained=retained, t3_acquired=t3,
                    final_seen_validation_mean=summary["final_seen_validation_mean"])
        item["eligible"] = (acquired >= .3 and t3 >= .5 and retained >= .8 * acquired
                            and acquired - retained <= .1)
        if item["eligible"]:
            eligible.append(item)
    selected = max(eligible, key=lambda x: x["final_seen_validation_mean"]) if eligible else None
    result = dict(records=records, selected=selected,
                  promotion_rule="T2 acquired >= .3; T3 acquired >= .5; T2 retention >= 80%; T2 drop <= .1; rank eligible by final seen validation mean")
    (root / "sweep_selection.json").write_text(json.dumps(result, indent=2) + "\n")
    if selected is None:
        event(status="no_configuration_passed", formal_launched=False)
        return
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, "selected_" + variant, gpu, selected["lr"],
                               selected["global_weight"], 80, variant, True)
                   for variant, gpu in [("fg20", 6), ("fg40", 7)]]
        for future in futures:
            future.result()
    event(status="selected_formal_controller_finished")


if __name__ == "__main__":
    main()
