"""Stop the old sweep after T2, run one standardized T3 epoch, evaluate and exit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--remaining-gpu", type=int, choices=[4, 5, 6, 7],
                        help="run the single remaining candidate on an available authorized GPU")
    args = parser.parse_args()
    root = args.root.resolve()
    request = json.loads((root / "early_stop_request.json").read_text())
    if args.remaining_gpu is not None:
        unfinished = [job for job in request["jobs"]
                      if not (root / "runs" / ("e1_" + job["run"]) / "t2_forgetting.json").exists()]
        if len(unfinished) != 1:
            parser.error("--remaining-gpu requires exactly one unfinished candidate")
    lock = threading.Lock()
    def event(**values):
        with lock, (root / "epoch1_events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S %z"), **values)) + "\n")

    def finish(job):
        name, pid, gpu = job["run"], job["pid"], job["gpu"]
        result_path = root / "runs" / ("e1_" + name) / "t2_forgetting.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        if args.remaining_gpu is not None:
            gpu = args.remaining_gpu
        source = root / "runs" / name
        while True:
            try:
                stages = json.loads((source / "stages.json").read_text())
                ready = len(stages) >= 2 and stages[1]["stage"] == 1
            except (FileNotFoundError, json.JSONDecodeError):
                ready = False
            if ready:
                break
            if not Path("/proc", str(pid)).exists():
                event(run=name, status="failed_before_T2_checkpoint")
                return dict(run=name, status="failed_before_T2_checkpoint")
            time.sleep(3)
        proc = Path("/proc", str(pid), "cmdline")
        # A stopped worker can remain as a zombie while its old controller is
        # paused; its empty cmdline needs no signal and is not a PID reuse.
        if proc.exists() and proc.read_bytes():
            argv = proc.read_bytes().split(b"\0")
            if b"--output" not in argv or argv[argv.index(b"--output") + 1] != str(source).encode():
                raise RuntimeError("PID no longer belongs to the expected experiment")
            os.kill(pid, signal.SIGTERM)
            try:
                os.kill(pid, signal.SIGCONT)  # Deliver pending TERM to a previously paused worker.
            except ProcessLookupError:
                pass
        reason = dict(status="stopped_at_user_request", replacement="e1_" + name,
                      reason="reuse paired T2 state; standardized T3 transition seed 44; one epoch only")
        (source / "STOP_REASON_EPOCH1.json").write_text(json.dumps(reason, indent=2) + "\n")
        event(run=name, **reason)
        while int(subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip()) < 18000:
            time.sleep(3)
        output = root / "runs" / ("e1_" + name)
        command = list(job["command"])
        command[command.index("--output") + 1] = str(output)
        command += ["--t3-one-epoch-from", str(source)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        with (root / "logs" / ("e1_" + name + ".log")).open("x") as log:
            process = subprocess.Popen(command, cwd=Path(__file__).parent, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        event(run="e1_" + name, status="started", gpu=gpu, pid=process.pid, command=command)
        code = process.wait()
        if code:
            event(run="e1_" + name, status="failed", exit_code=code)
            return dict(run=name, status="failed", exit_code=code)
        summary = json.loads((output / "summary.json").read_text())
        epochs = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()
                  if "epoch_seconds" in json.loads(line)]
        assert len(epochs) == 1 and epochs[0]["stage"] == 2 and epochs[0]["epoch"] == 0
        assert epochs[0]["iteration"] == 356
        matrix = summary["validation_matrix"]
        before, after = matrix[1][1], matrix[2][1]
        assert abs(before - stages[1]["validation_evaluated"]["T2"]["benchmark_mean"]) < 1e-9
        result = dict(run=name, status="completed", t3_epochs=1, transition_seed=44,
                      evaluation_split="validation", t2_before=before, t2_after=after,
                      absolute_drop=before-after, retention_ratio=after/before if before > 0 else None)
        (output / "t2_forgetting.json").write_text(json.dumps(result, indent=2) + "\n")
        event(**result)
        return result

    event(status="epoch1_protocol_started", automatic_formal_launch=False,
          protocol="reuse original best T2 paired states; common transition seed 44; retain 20-epoch LR horizon")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(finish, request["jobs"]))
    (root / "t3_epoch1_forgetting.json").write_text(json.dumps(results, indent=2) + "\n")
    old = request.get("controller_pid")
    proc = Path("/proc", str(old), "cmdline")
    if old and proc.exists():
        cmd = proc.read_bytes()
        if b"run_t2_sweep.py" in cmd and str(root).encode() in cmd:
            os.kill(old, signal.SIGKILL)
    event(status="stopped_after_requested_evaluations", automatic_formal_launch=False)


if __name__ == "__main__":
    main()
