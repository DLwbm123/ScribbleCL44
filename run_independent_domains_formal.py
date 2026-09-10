#!/usr/bin/env python3
"""Run independent B-F with Domain A's selected configuration on GPUs 4-7."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
import subprocess
import sys
import time

from run_independent_a_spatial_sweep import run_training, write_json


def gpu_idle(gpu):
    output = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}",
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    memory, utilization = map(int, output.strip().split(","))
    return memory < 100 and utilization == 0


def gpu_free_memory():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    return dict(tuple(map(int, line.split(","))) for line in output.strip().splitlines())


def run_gpu_queue(tasks, run_task, min_free_memory_mib=12288):
    pending = Queue()
    for task in tasks:
        pending.put(task)

    allocation_lock = Lock()
    reserved_until = {gpu: 0.0 for gpu in (4, 5, 6, 7)}

    def worker():
        rows = []
        while not pending.empty():
            with allocation_lock:
                free = gpu_free_memory()
                available = [gpu for gpu in (4, 5, 6, 7)
                             if free.get(gpu, 0) >= min_free_memory_mib
                             and time.monotonic() >= reserved_until[gpu]]
                gpu = max(available, key=lambda gpu: free[gpu]) if available else None
                if gpu is not None:
                    try:
                        task = pending.get_nowait()
                    except Empty:
                        break
                    # Allow CUDA startup allocations to become visible before sharing again.
                    reservation = time.monotonic() + 60
                    reserved_until[gpu] = reservation
            if gpu is None:
                time.sleep(10)
                continue
            try:
                rows.append(run_task(task, gpu))
            finally:
                pending.task_done()
                with allocation_lock:
                    if reserved_until[gpu] == reservation:
                        reserved_until[gpu] = 0.0
        return rows

    with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
        futures = [pool.submit(worker) for _ in tasks]
        return [row for future in futures for row in future.result()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-a", type=Path)
    parser.add_argument("--demo-test-selection", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-free-memory-mib", type=int, default=12288)
    args = parser.parse_args()
    if args.min_free_memory_mib < 1:
        parser.error("--min-free-memory-mib must be positive")
    if not 0 <= args.warmup_epochs < 80:
        parser.error("--warmup-epochs must be between 0 and 79")
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    if not args.demo_test_selection:
        if args.completed_a is None:
            parser.error("standard mode requires --completed-a")
        prior = json.loads((args.completed_a / "summary.json").read_text())
        assert prior["complete"] and prior["task_order"] == ["A"] and prior["completed_epochs"] == 80
    tasks = list(range(1 if args.demo_test_selection else 2, 7))
    args.output.mkdir(parents=True, exist_ok=False)
    args.adopt_running = {}
    protocol = {
        "status": "running", "gpus": [4, 5, 6, 7], "domains": list("ABCDEF"),
        "min_free_memory_mib": args.min_free_memory_mib, "allow_gpu_sharing": True,
        "completed_a": str(args.completed_a) if args.completed_a else None,
        "fresh_domains": ["ABCDEF"[task - 1] for task in tasks],
        "epochs": 80, "seed": 42, "spatial_weight": 0.01,
        "spatial_first_epoch_index": args.warmup_epochs, "configuration_selected_on": "A validation",
        "selection_uses_test": args.demo_test_selection,
        "selection_split": "test" if args.demo_test_selection else "val",
        "result_usage": "platform_demo" if args.demo_test_selection else "standard_evaluation", "training_implementation": "shared_stage_loop",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(args.output / "pipeline.json", protocol)
    def run_task(task, gpu):
        domain = "ABCDEF"[task - 1]
        print(f"Starting Domain {domain} on GPU {gpu}", flush=True)
        try:
            row = run_training(args, domain, 0.01, 80, gpu, True, task=task,
                               warmup=args.warmup_epochs - 1, demo_test_selection=args.demo_test_selection)
            return {"domain": domain, "status": "complete", **row}
        except Exception as error:
            row = {"domain": domain, "gpu": gpu, "status": "failed", "error": repr(error)}
            write_json(args.output / f"{domain}.failure.json", row)
            return row

    rows = sorted(run_gpu_queue(tasks, run_task, args.min_free_memory_mib), key=lambda row: row["domain"])
    protocol["results"] = rows
    protocol["status"] = "complete" if len(rows) == len(tasks) and all(r["status"] == "complete" for r in rows) else "failed"
    write_json(args.output / "pipeline.json", protocol)
    if protocol["status"] != "complete":
        raise RuntimeError("One or more domains failed; see pipeline.json")


def self_check():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    module = sys.modules[__name__]
    for demo, fail in ((False, False), (False, True), (True, False), (True, True)):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "summary.json", {"complete": True, "task_order": ["A"], "completed_epochs": 80})
            calls = []

            def train(args, name, weight, epochs, gpu, test, task, warmup, demo_test_selection):
                assert name == "ABCDEF"[task - 1] and weight == 0.01 and epochs == 80 and test
                assert warmup == 9 and demo_test_selection == demo
                calls.append(task)
                if fail and task == 3:
                    raise RuntimeError("expected test failure")
                return {"gpu": gpu}

            argv = [__file__, "--data-root", directory, "--sparse-root", directory,
                    "--completed-a", directory, "--output", str(root / "output"), "--warmup-epochs", "10"]
            if demo:
                argv.append("--demo-test-selection")
            with patch.object(module, "run_training", train), patch.object(module, "gpu_free_memory", return_value={4: 14000, 5: 14000, 6: 14000, 7: 14000}), patch.object(sys, "argv", argv):
                try:
                    main()
                except RuntimeError:
                    assert fail
                else:
                    assert not fail
            result = json.loads((root / "output/pipeline.json").read_text())
            assert sorted(calls) == list(range(1 if demo else 2, 7))
            assert result["selection_uses_test"] == demo
            assert result["status"] == ("failed" if fail else "complete")
            assert result["spatial_first_epoch_index"] == 10
    print("Domain queue and failure handling: PASS")


if __name__ == "__main__":
    self_check() if sys.argv[1:] == ["--self-check"] else main()
