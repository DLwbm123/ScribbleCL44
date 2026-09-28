"""Finite ER/DER timing queue; each restores T1 and performs 30 T2 updates."""
import json
import os
from pathlib import Path
import subprocess
import time
from setproctitle import setproctitle

setproctitle("run")
ROOT = Path(__file__).resolve().parent.parent
PLAN = json.loads((ROOT / "plan.json").read_text())
state = dict(status="running", started_at=time.time(), jobs={})


def save():
    (ROOT / "receipt.json").write_text(json.dumps(state, indent=2) + "\n")


try:
    save()
    for method in ("zs-er", "zs-der"):
        free = int(subprocess.check_output([
            "nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
        ], text=True))
        if free < 22000:
            raise RuntimeError("GPU0 has less than 22000 MiB free")
        output = ROOT / "jobs" / method
        args = ["--data-root", PLAN["data"], "--sparse-root", PLAN["sparse"],
                "--output", str(output), "--device", "cuda:0", "--method", method,
                "--seed", "42", "--epochs-per-task", "80", "--task-epochs", "80", "1", "60",
                "--max-task", "2", "--batch-size", "4", "--workers", "0", "--lr", ".03",
                "--pce-loss-weight", "1", "--zs-global-weight", ".1", "--zs-spatial-loss-weight", "0",
                "--zs-spatial-warmup-epochs", "33", "--validate-every", "375", "--cache-h5",
                "--der-buffer-size", "64", "--der-minibatch-size", "4",
                "--der-alpha", "0" if method == "zs-er" else ".5",
                "--der-beta", "1" if method == "zs-er" else "0", "--with-mib", "--mib-kd-weight", "10",
                "--resume-from", str(Path(PLAN["resume"]) / method), "--validation-only",
                "--max-train-batches", "30"]
        if method == "zs-der":
            args += ["--safe-numerics", "--grad-clip-norm", "5"]
        source = PLAN["source_er" if method == "zs-er" else "source_der"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", JOB_ARGS=json.dumps(args),
                   PYTHONPATH=source + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   TMPDIR=str(ROOT / "tmp"), PYTHONUNBUFFERED="1")
        with (ROOT / "logs" / (method + ".log")).open("x") as log:
            child = subprocess.Popen(["./main", "-u", "benchmark.py"], cwd=ROOT / "control", env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state["jobs"][method] = dict(status="running", pid=child.pid)
            save()
            code = child.wait()
        (ROOT / "logs" / (method + ".exitcode")).write_text(str(code) + "\n")
        if code:
            raise RuntimeError(f"{method} exited {code}")
        timing = json.loads((output / "timing.json").read_text())
        state["jobs"][method].update(status="complete", timing=timing)
        save()
    state.update(status="complete", finished_at=time.time())
    save()
except Exception as exc:
    state.update(status="failed", error=str(exc), finished_at=time.time())
    save()
    raise
