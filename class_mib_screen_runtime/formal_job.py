"""One fresh Class baseline + MiB run with the explicitly recorded task budget."""
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
EPOCHS = int(PLAN.get("epochs_per_task", 50))
LR = float(PLAN.get("learning_rate", .03))
TASKS = int(PLAN.get("max_task", 3))
VALIDATION_ONLY = bool(PLAN.get("validation_only", False))
assert EPOCHS > 0 and math.isfinite(LR) and LR > 0
assert 1 <= TASKS <= 3
METHOD, GPU = os.environ["METHOD"], os.environ["GPU"]
STATE = dict(status="starting", method=METHOD, gpu=GPU, started_at=time.time())


def save():
    path = ROOT / "receipts" / (METHOD + ".json")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(STATE, indent=2) + "\n")
    temp.replace(path)


try:
    save()
    free = int(subprocess.check_output([
        "nvidia-smi", "-i", GPU, "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True))
    if free < 22000:
        raise RuntimeError("Less than 22000 MiB free; no automatic retry")
    output = ROOT / "jobs" / METHOD
    source = ROOT / ("source_der" if METHOD == "zs-der" else "source_er")
    args = ["--data-root", PLAN["data"], "--sparse-root", PLAN["sparse"],
            "--output", str(output), "--device", "cuda:0", "--method", METHOD,
            "--seed", "42", "--epochs-per-task", str(EPOCHS), "--task-epochs", *[str(EPOCHS)] * 3,
            "--max-task", str(TASKS), "--batch-size", "4", "--workers", "0", "--lr", str(LR),
            "--pce-loss-weight", "1", "--zs-global-weight", ".1", "--zs-spatial-loss-weight", "0",
            "--zs-spatial-warmup-epochs", "33", "--validate-every", "375", "--cache-h5",
            "--der-buffer-size", "64", "--der-minibatch-size", "4",
            "--der-alpha", "0" if METHOD == "zs-er" else ".5",
            "--der-beta", "1" if METHOD == "zs-er" else "0",
            "--with-mib", "--mib-kd-weight", "10"]
    if METHOD == "zs-der":
        args += ["--safe-numerics", "--grad-clip-norm", "5"]
    if VALIDATION_ONLY:
        args += ["--validation-only"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, JOB_ARGS=json.dumps(args),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TMPDIR=str(ROOT / "tmp"), PYTHONUNBUFFERED="1")
    if PLAN.get("two_stage_preflight", False):
        check_args = args.copy()
        check_output = ROOT / "checks" / METHOD
        for key, value in (("--output", str(check_output)), ("--epochs-per-task", "1"), ("--max-task", "2")):
            check_args[check_args.index(key) + 1] = value
        start = check_args.index("--task-epochs") + 1
        check_args[start:start + 3] = ["1"] * 3
        check_args += ["--validation-only", "--max-train-batches", "2", "--fisher-batches", "2",
                       "--gpm-examples", "2", "--gpm-max-patches-per-layer", "64",
                       "--gpm-max-matrix-elements", "40000"]
        with (ROOT / "logs" / (METHOD + "-check.log")).open("x") as log:
            child = subprocess.Popen(["./main", "-u", "entry.py"], cwd=source,
                                     env=dict(env, JOB_ARGS=json.dumps(check_args)),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            STATE.update(status="preflight", pid=child.pid)
            save()
            check_code = child.wait()
        if check_code:
            raise RuntimeError(f"Two-stage preflight exited {check_code}; formal run not started")
        check = json.loads((check_output / "summary.json").read_text())
        check_rows = [json.loads(line) for line in (check_output / "train.jsonl").read_text().splitlines()]
        check_rows = [row for row in check_rows if "loss" in row]
        assert check["completed_stages"] == 2 and len(check_rows) == 2
        assert all(math.isfinite(row["loss"]) for row in check_rows)
        assert check_rows[1]["mib_kd_loss"] > 0 and not check["replay"]
        if METHOD == "zs-ewc":
            assert check_rows[1]["ewc_penalty"] > 0
        if METHOD == "zs-gpm":
            assert check_rows[1]["gpm_gradient_ratio"] is not None
        STATE["preflight_passed"] = True
        save()
    with (ROOT / "logs" / (METHOD + ".log")).open("x") as log:
        child = subprocess.Popen(["./main", "-u", "entry.py"], cwd=source, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        STATE.update(status="running", pid=child.pid, free_mib_before_launch=free)
        save()
        code = child.wait()
    (ROOT / "logs" / (METHOD + ".exitcode")).write_text(str(code) + "\n")
    if code:
        raise RuntimeError(f"Training exited {code}")
    result = json.loads((output / "summary.json").read_text())
    assert result["completed_stages"] == TASKS and result["task_epochs"] == [EPOCHS] * 3
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    for stage in range(TASKS):
        epochs = [r for r in rows if r.get("stage") == stage and "loss" in r]
        assert [r["epoch"] for r in epochs] == list(range(EPOCHS))
        assert all(math.isfinite(r["loss"]) for r in epochs)
        assert all(r["mib_kd_loss"] > 0 if stage else r["mib_kd_loss"] == 0 for r in epochs)
        assert (output / f"s{stage+1:02d}_state.pt").stat().st_size > 0
        if not VALIDATION_ONLY:
            assert all(math.isfinite(v) for v in result["matrix"][stage][:stage+1])
    if TASKS == 3 and not VALIDATION_ONLY:
        assert math.isfinite(result["whole_class_dice"]["benchmark_mean"])
    STATE.update(status="complete", exitcode=0, finished_at=time.time(),
                 final_seen_mean=result["final_seen_mean"],
                 final_seen_validation_mean=result["final_seen_validation_mean"])
    save()
except Exception as error:
    STATE.update(status="failed", error=str(error), finished_at=time.time())
    save()
    raise
