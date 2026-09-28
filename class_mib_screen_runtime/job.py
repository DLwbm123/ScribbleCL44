"""One fixed paired T2 screen per GPU; no retries or formal continuation."""
import json
import math
import os
from pathlib import Path
import subprocess
import time
from setproctitle import setproctitle

setproctitle("run")
ROOT = Path(__file__).resolve().parent.parent
PLAN = json.loads((ROOT / "plan.json").read_text())
METHOD = os.environ["METHOD"]
GPU = os.environ["GPU"]
STATE = dict(status="running", method=METHOD, gpu=GPU, jobs={}, started_at=time.time())


def save():
    path = ROOT / "receipts" / (METHOD + ".json")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(STATE, indent=2) + "\n")
    temp.replace(path)


def run(name, mib, smoke=False):
    free = int(subprocess.check_output([
        "nvidia-smi", "-i", GPU, "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True))
    if free < 22000:
        raise RuntimeError("Less than 22000 MiB free; no waiting or automatic retry")
    output = ROOT / "jobs" / name
    source = ROOT / ("source_er" if METHOD == "zs-er" else "source_der")
    epochs = 1 if smoke else 5
    args = ["--data-root", PLAN["data"], "--sparse-root", PLAN["sparse"],
            "--output", str(output), "--device", "cuda:0", "--method", METHOD,
            "--seed", "42", "--epochs-per-task", "80", "--task-epochs", "80", str(epochs), "60",
            "--max-task", "2", "--batch-size", "4", "--workers", "0", "--lr", ".03",
            "--pce-loss-weight", "1", "--zs-global-weight", ".1", "--zs-spatial-loss-weight", "0",
            "--zs-spatial-warmup-epochs", "33", "--validate-every", "2" if smoke else "375",
            "--cache-h5", "--der-buffer-size", "64", "--der-minibatch-size", "4",
            "--der-alpha", "0" if METHOD == "zs-er" else ".5",
            "--der-beta", "1" if METHOD == "zs-er" else "0",
            "--resume-from", str(ROOT / "resume" / METHOD), "--validation-only"]
    if mib:
        args += ["--with-mib", "--mib-kd-weight", "10"]
    if METHOD == "zs-der":
        args += ["--safe-numerics", "--grad-clip-norm", "5"]
    if smoke:
        args += ["--max-train-batches", "2"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, JOB_ARGS=json.dumps(args),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TMPDIR=str(ROOT / "tmp"), PYTHONUNBUFFERED="1")
    with (ROOT / "logs" / (name + ".log")).open("x") as log:
        child = subprocess.Popen(["./main", "-u", "entry.py"], cwd=source, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        STATE["jobs"][name] = dict(status="running", pid=child.pid, started_at=time.time())
        save()
        code = child.wait()
    (ROOT / "logs" / (name + ".exitcode")).write_text(str(code) + "\n")
    if code:
        STATE["jobs"][name].update(status="failed", exitcode=code)
        raise RuntimeError(f"{name} exited {code}")
    result = json.loads((output / "summary.json").read_text())
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row.get("stage") == 1 and "loss" in row]
    assert result["completed_stages"] == 2
    assert [row["epoch"] for row in rows] == list(range(epochs))
    assert all(math.isfinite(row["loss"]) for row in rows)
    assert all((row["mib_kd_loss"] > 0) if mib else (row["mib_kd_loss"] == 0) for row in rows)
    startup = json.loads((output / "startup.json").read_text())
    assert startup["stored_examples"] == 64
    assert startup["replay_global_loss"] == 0
    if METHOD == "zs-er":
        assert startup["feature_loss"] == 0 and startup["replay_pce_loss"] > 0
    else:
        assert startup["feature_loss"] > 0 and startup["replay_pce_loss"] == 0
    STATE["jobs"][name].update(status="complete", exitcode=0, finished_at=time.time(),
                               validation=result["stage_rows"][-1]["seen_validation"],
                               startup=startup)
    save()
    return result["stage_rows"][-1]["seen_validation"]


try:
    save()
    run(METHOD + "-smoke", True, smoke=True)
    control = run(METHOD + "-control5", False)
    candidate = run(METHOD + "-mib5", True)
    STATE.update(status="complete", finished_at=time.time(), control=control, candidate=candidate,
                 mean_validation_gain=(sum(candidate.values()) - sum(control.values())) / 2)
    save()
except Exception as error:
    STATE.update(status="failed", error=str(error), finished_at=time.time())
    save()
    raise
