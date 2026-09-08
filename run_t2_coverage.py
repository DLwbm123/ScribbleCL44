"""Background four-run contrast with a required real-data DER++ integration gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cl-only", action="store_true", help="leave the running independent pair alone")
    parser.add_argument("--replay-minibatch-size", type=int, default=4)
    args = parser.parse_args()
    if args.replay_minibatch_size not in (4, 8):
        parser.error("this contrast supports replay minibatches 4 or 8")
    root = args.root.resolve()
    jobs = []
    events = root / "events.jsonl"

    def record(**event):
        with events.open("a") as stream:
            stream.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S %z"), **event)) + "\n")

    def launch(name, gpu, variant, independent=False, gate=False):
        while True:
            free = int(subprocess.check_output([
                "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits",
            ], text=True).strip())
            required = 8000 if independent else (18000 if args.replay_minibatch_size == 4 else 24000)
            if free >= required:
                break
            record(run=name, status="waiting_for_memory", gpu=gpu, free_mib=free)
            time.sleep(30)
        command = [sys.executable, "main.py", "--setting-run",
                   "--data-root", str(args.data_root),
                   "--sparse-root", str(root / "annotations" / variant),
                   "--output", str(root / "runs" / name), "--device", "cuda:0",
                   "--seed", "42", "--epochs-per-task", "2" if gate else "80",
                   "--batch-size", "4", "--workers", "4", "--lr", ".003",
                   "--grad-clip-norm", "1", "--validate-each-epoch",
                   "--zs-global-weight", ".1", "--zs-spatial-loss-weight", "0",
                   "--annotation-id", "organ_T2_nested_20260908_" + variant,
                   "--method", "zs-sequential" if independent else "zs-derpp"]
        if independent:
            command += ["--organ-task", "T2"]
        else:
            command += ["--max-task", "2" if gate else "3", "--der-alpha", ".5",
                        "--der-beta", ".5", "--der-buffer-size", "128",
                        "--der-minibatch-size", str(args.replay_minibatch_size)]
        if gate:
            command += ["--max-train-batches", "32"]
        else:
            command += ["--test-evaluation"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        with (root / "logs" / (name + ".log")).open("x") as stream:
            process = subprocess.Popen(command, cwd=Path(__file__).parent, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
        record(run=name, status="started", gpu=gpu, pid=process.pid, command=command)
        return process

    if not args.cl_only:
        for name, gpu, variant in [("ind_fg20", 4, "fg20"), ("ind_fg40", 5, "fg40")]:
            jobs.append((name, launch(name, gpu, variant, independent=True)))
    gate_name = f"stability_gate_mb{args.replay_minibatch_size}"
    gate = launch(gate_name, 6, "fg20", gate=True)
    code = gate.wait()
    gate_ok = code == 0
    if gate_ok:
        summary = json.loads((root / "runs" / gate_name / "summary.json").read_text())
        rows = [json.loads(line) for line in (root / "runs" / gate_name / "train.jsonl").read_text().splitlines()]
        gate_ok = (summary["completed_stages"] == 2
                   and any(row.get("derpp_feature_loss", 0) > 0 for row in rows)
                   and not (root / "runs" / gate_name / "FIRST_NONFINITE.json").exists())
    record(run=gate_name, status="passed" if gate_ok else "failed", exit_code=code,
           scope="128 real-data optimizer steps across T1/T2; does not prove 80-epoch stability or retention")
    if gate_ok:
        for name, gpu, variant in [("cl_fg20", 6, "fg20"), ("cl_fg40", 7, "fg40")]:
            jobs.append((name, launch(name, gpu, variant)))
    for name, process in jobs:
        code = process.wait()
        record(run=name, status="completed" if code == 0 else "failed", exit_code=code)
    record(status="controller_finished", cl_launched=gate_ok)


if __name__ == "__main__":
    main()
