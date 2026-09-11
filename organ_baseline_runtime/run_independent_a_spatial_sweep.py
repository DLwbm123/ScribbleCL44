#!/usr/bin/env python3
"""Domain-A spatial sweep (6 x 20 epochs), then a fresh 80-epoch selected run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

WEIGHTS = (0.0, 0.01, 0.05, 0.1, 0.3, 1.0)
WARMUP = 4  # Runner uses epoch > warmup: five warm-up epochs, active from epoch index 5.


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def select_candidate(rows):
    if len(rows) != len(WEIGHTS) or {r["spatial_weight"] for r in rows} != set(WEIGHTS):
        raise ValueError("selection requires all six distinct completed candidates")
    if not all(math.isfinite(r["validation_foreground"]) for r in rows):
        raise ValueError("non-finite validation metric")
    return max(rows, key=lambda r: (r["validation_foreground"], -r["spatial_weight"]))


def adopted_exit_status(pid, parent, output):
    """Read a retained child's exit status while its original coordinator is stopped."""
    proc = Path("/proc") / str(pid)
    while True:
        fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        if int(fields[1]) != parent:
            raise RuntimeError("adopted process parent changed")
        if fields[0] == "Z":
            return os.waitstatus_to_exitcode(int(fields[49]))
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if str(output) + " " not in command:
            raise RuntimeError("adopted process output mismatch")
        time.sleep(2)


def run_training(args, name, weight, epochs, gpu, test, task=1, warmup=WARMUP, demo_test_selection=False, lr=0.03, global_weight=1.0, validate_every=200):
    output = args.output / name
    command = [
        sys.executable, "-u", "main.py", "--setting-run",
        "--data-root", str(args.data_root), "--sparse-root", str(args.sparse_root),
        "--output", str(output), "--device", "cuda:0", "--seed", "42",
        "--independent-reference", "--independent-task", str(task),
        "--method", "zs-sequential", "--epochs-per-task", str(epochs),
        "--batch-size", "4", "--lr", str(lr), "--workers", "8", "--validate-every", str(validate_every),
        "--pce-loss-weight", "1", "--zs-global-weight", str(global_weight),
        "--zs-spatial-loss-weight", str(weight), "--zs-spatial-warmup-epochs", str(warmup),
    ]
    if demo_test_selection:
        assert test
        command.append("--independent-demo-test-selection")
    if not test:
        command.append("--independent-skip-test")
    if name in args.adopt_running:
        original = json.loads((args.output / (name + ".command.json")).read_text())
        assert original == {"gpu": gpu, "command": command}
        start = (args.output / (name + ".command.json")).stat().st_mtime
        returncode = adopted_exit_status(args.adopt_running[name], args.adopt_parent, output)
    else:
        write_json(args.output / (name + ".command.json"), {"gpu": gpu, "command": command})
        start = time.time()
        with (args.output / (name + ".log")).open("x") as log:
            result = subprocess.run(
                command, cwd=Path(__file__).resolve().parent,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4"},
                stdout=log, stderr=subprocess.STDOUT,
            )
        returncode = result.returncode
    (args.output / (name + ".exitcode")).write_text(str(returncode) + "\n")
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    record = summary["records"][0]
    assert manifest["status"] == "complete" and summary["complete"]
    assert summary["completed_epochs"] == epochs and summary["iteration"] == summary["batches_per_epoch"] * epochs
    assert summary["train_samples"] > 0 and summary["task_order"] == ["ABCDEF"[task - 1]]
    if task == 1:
        assert summary["train_samples"] == 301 and summary["batches_per_epoch"] == 76
    assert summary["score_split"] == ("test" if test else "val")
    assert summary["test_evaluated"] == test
    assert (record["test"] is not None) == (test and not demo_test_selection)
    if demo_test_selection:
        assert summary["test_for_selection"] and summary["result_usage"] == "platform_demo"
        assert summary["selection_split"] == "test" and not summary["held_out_test_evaluated"]
    assert manifest["learning_rate"] == lr and manifest["zs_global_weight"] == global_weight
    assert manifest["validate_every"] == validate_every
    assert manifest["zs_spatial_loss_weight"] == weight
    assert manifest["zs_spatial_warmup_epochs"] == warmup
    train_rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    epoch_rows = [r for r in train_rows if "loss" in r]
    active = [r for r in epoch_rows if r["zs_em_mixture_ratios"] is not None]
    assert len(epoch_rows) == epochs
    assert len(active) == (max(0, epochs - warmup - 1) if weight > 0 else 0)
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["zs_spatial_loss"]) for r in epoch_rows)
    if weight > 0 and epochs > warmup + 1:
        assert active[0]["epoch"] == warmup + 1 and any(r["zs_spatial_loss"] > 0 for r in active)
    best_selection = record["best_selection"] if demo_test_selection else record["best_validation"]
    row = {
        "run": name, "gpu": gpu, "spatial_weight": weight, "epochs": epochs,
        "learning_rate": lr, "global_weight": global_weight, "validate_every": validate_every,
        "validation_foreground": best_selection["foreground_mean"],
        "validation_inclusive": best_selection["inclusive_mean"],
        "best_epoch": best_selection["epoch"],
        "best_iteration": best_selection["iteration"],
        "spatial_active_epochs": len(active),
        "spatial_first_epoch_index": warmup + 1, "runner_warmup_argument": warmup,
        "mean_active_spatial_loss": sum(r["zs_spatial_loss"] for r in active) / len(active) if active else 0.0,
        "elapsed_seconds": time.time() - start, "test_evaluated": test,
    }
    if demo_test_selection:
        row["demo_selection_foreground"] = row.pop("validation_foreground")
        row["demo_selection_inclusive"] = row.pop("validation_inclusive")
        row.update(test_for_selection=True, selection_split="test", result_usage="platform_demo", held_out_test_evaluated=False)
    if test and not demo_test_selection:
        row["test"] = record["test"]
        if task == 1:
            row["target_difference"] = record["test"]["foreground_mean"] - 0.7261413306722405
            row["target_reached"] = row["target_difference"] >= 0
    write_json(args.output / (name + ".result.json"), row)
    print(json.dumps(row), flush=True)
    return row


def self_check():
    rows = [{"spatial_weight": w, "validation_foreground": 0.5,
             "test_foreground": 100 * w} for w in WEIGHTS]
    assert select_candidate(rows)["spatial_weight"] == 0  # Ties prefer simpler, never test.
    rows[2]["validation_foreground"] = 0.6
    assert select_candidate(list(reversed(rows)))["spatial_weight"] == 0.05
    try:
        select_candidate(rows[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete sweep must not launch formal training")
    if sys.platform == "linux":
        parent = subprocess.Popen([
            sys.executable, "-c",
            "import subprocess,sys; child=subprocess.Popen([sys.executable,'-c',"
            "'import time; time.sleep(1); raise SystemExit(7)', '/tmp/sweep-adoption-check']);"
            "print(child.pid,flush=True); child.wait()",
        ], stdout=subprocess.PIPE, text=True)
        child = int(parent.stdout.readline())
        os.kill(parent.pid, signal.SIGSTOP)
        try:
            assert adopted_exit_status(child, parent.pid, Path("/tmp/sweep-adoption-check")) == 7
        finally:
            parent.kill()
            parent.wait()
    print("selection_self_check_passed")


def main():
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--adopt-running", nargs="*", default=[], metavar="RUN=PID")
    parser.add_argument("--adopt-parent", type=int)
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU indices must be distinct and non-negative")
    args.adopt_running = dict(item.split("=", 1) for item in args.adopt_running)
    args.adopt_running = {name: int(pid) for name, pid in args.adopt_running.items()}
    if not args.output.is_absolute():
        parser.error("--output must be an absolute path on experiment storage")
    if args.adopt_running:
        if args.adopt_parent is None:
            parser.error("--adopt-running requires a stopped --adopt-parent")
        parent = Path("/proc") / str(args.adopt_parent)
        assert (parent / "stat").read_text().rsplit(") ", 1)[1].split()[0] == "T"
        assert str(args.output).encode() in (parent / "cmdline").read_bytes()
        original_protocol = json.loads((args.output / "pipeline.json").read_text())
        assert original_protocol["status"] == "sweep_running"
        assert set(args.adopt_running) <= {f"sweep_s{i:02d}" for i in range(len(WEIGHTS))}
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "sweep_running", "scenario": "domain", "task": "A", "seed": 42,
        "spatial_weights": WEIGHTS, "sweep_epochs": 20, "formal_epochs": 80,
        "warmup_epochs": 5, "runner_warmup_argument": WARMUP,
        "pce_weight": 1, "global_weight": 1, "lr": 0.03, "batch_size": 4,
        "workers": 8, "omp_num_threads": 4, "validate_every": 200,
        "optimizer_weight_decay": 0, "manual_gradient_decay": 1e-4,
        "sparse_protocol": args.sparse_root.parent.name, "gpus": args.gpus,
        "selection_metric": "best_foreground_validation_dice",
        "selection_uses_test": False, "formal_initialization": "fresh_seed42",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True,
        ).strip(),
    }
    if args.adopt_running:
        protocol.update(original_source_commit=original_protocol["source_commit"],
                        adopted_runs=args.adopt_running, superseded_coordinator=args.adopt_parent)
    write_json(args.output / "pipeline.json", protocol)
    try:
        def worker(gpu, indices):
            return [
                run_training(args, f"sweep_s{index:02d}", WEIGHTS[index], 20, gpu, False)
                for index in indices
            ]
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, gpu, range(i, len(WEIGHTS), len(args.gpus)))
                       for i, gpu in enumerate(args.gpus)]
            rows = [row for future in futures for row in future.result()]
        if args.adopt_running:
            os.kill(args.adopt_parent, signal.SIGKILL)  # Both retained children have exited and been validated.
        best = select_candidate(rows)
        write_json(args.output / "sweep_summary.json", {
            "complete": True, "selection_uses_test": False,
            "selection_metric": "best_foreground_validation_dice",
            "candidates": sorted(rows, key=lambda r: r["spatial_weight"]), "best": best,
        })
        protocol.update(status="formal_running", selected_spatial_weight=best["spatial_weight"])
        write_json(args.output / "pipeline.json", protocol)
        formal = run_training(args, "formal80", best["spatial_weight"], 80, 6, True)
        protocol.update(status="complete", formal_result=formal)
        write_json(args.output / "pipeline.json", protocol)
    except Exception as error:
        protocol.update(status="failed", error=repr(error))
        write_json(args.output / "pipeline.json", protocol)
        raise


if __name__ == "__main__":
    main()
