"""Finite GPU3 screening and validation-gated continuation. No automatic retries."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from setproctitle import setproctitle

setproctitle("run")
ROOT = Path(__file__).resolve().parent.parent
PLAN = json.loads((ROOT / "plan.json").read_text())
STATE = {"status": "running", "started_at": time.time(), "jobs": {}, "gates": {}}


def save():
    temp = ROOT / "progress.tmp"
    temp.write_text(json.dumps(STATE, indent=2) + "\n")
    temp.replace(ROOT / "progress.json")


def run(name, method, improved, epochs, max_task, resume, smoke=False):
    free = int(subprocess.check_output(["nvidia-smi", "-i", "3", "--query-gpu=memory.free",
                                       "--format=csv,noheader,nounits"], text=True))
    if free < 16000:
        STATE["jobs"][name] = {"status": "blocked", "reason": "GPU3 free memory below 16000 MiB"}
        save()
        return None
    source = ROOT / ("source_er" if method == "zs-er" and not improved else "source_der")
    output = ROOT / "jobs" / name
    args = ["--data-root", PLAN["data"], "--sparse-root", PLAN["sparse"],
            "--output", str(output), "--device", "cuda:0", "--method", method,
            "--seed", "42", "--epochs-per-task", "80", "--task-epochs", "80", str(epochs), "60",
            "--max-task", str(max_task), "--batch-size", "4", "--workers", "0",
            "--lr", ".03", "--pce-loss-weight", "1", "--zs-global-weight", ".1",
            "--zs-spatial-loss-weight", "0", "--zs-spatial-warmup-epochs", "33",
            "--validate-every", "2" if smoke else "375", "--cache-h5",
            "--der-buffer-size", "64", "--der-minibatch-size", "4",
            "--der-alpha", "0" if method == "zs-er" else ".5",
            "--der-beta", "1" if method == "zs-er" else "0", "--resume-from", str(resume)]
    if improved:
        args += ["--improved-replay"]
    if improved or method == "zs-der":
        args += ["--safe-numerics", "--grad-clip-norm", "5"]
    if smoke or epochs == 5:
        args += ["--validation-only"]
    if smoke:
        args += ["--max-train-batches", "2"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", JOB_ARGS=json.dumps(args),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TMPDIR=str(ROOT / "tmp"), PYTHONUNBUFFERED="1")
    with (ROOT / "logs" / (name + ".log")).open("x") as log:
        p = subprocess.Popen([sys.executable, "-u", "entry.py"], cwd=source, env=env,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        STATE["jobs"][name] = {"status": "running", "pid": p.pid, "started_at": time.time()}
        save()
        code = p.wait()
    (ROOT / "logs" / (name + ".exitcode")).write_text(str(code) + "\n")
    if code:
        STATE["jobs"][name].update(status="failed", exitcode=code, finished_at=time.time())
        save()
        return None
    summary = json.loads((output / "summary.json").read_text())
    rows = [json.loads(l) for l in (output / "train.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "loss" in r]
    assert summary["completed_stages"] == max_task
    for stage in range(max_task):
        stage_rows = [r for r in rows if r["stage"] == stage]
        assert [r["epoch"] for r in stage_rows] == list(range([80, epochs, 60][stage]))
        assert all(math.isfinite(r["loss"]) for r in stage_rows)
        assert (output / ("s%02d_state.pt" % (stage + 1))).stat().st_size > 0
    STATE["jobs"][name].update(status="complete", exitcode=0, finished_at=time.time(),
                               validation=summary["stage_rows"][-1]["seen_validation"])
    save()
    return summary


def main():
    save()
    ready = []
    for method in ("zs-er", "zs-der"):
        prefix = ROOT / "resume" / method
        if run(method + "-smoke", method, True, 1, 2, prefix, smoke=True) is not None:
            ready.append(method)
    selected = []
    for method in ready:
        prefix = ROOT / "resume" / method
        control = run(method + "-control5", method, False, 5, 2, prefix)
        candidate = run(method + "-improved5", method, True, 5, 2, prefix)
        if control is None or candidate is None:
            STATE["gates"][method] = {"passed": False, "reason": "paired screening incomplete"}
            save()
            continue
        base = control["stage_rows"][-1]["seen_validation"]
        score = candidate["stage_rows"][-1]["seen_validation"]
        initial = candidate["stage_rows"][0]["seen_validation"]["T1"]
        passed = (score["T1"] >= .5 * initial and score["T2"] >= max(.15, .8 * base["T2"])
                  and sum(score.values()) / 2 >= sum(base.values()) / 2 + .02)
        STATE["gates"][method] = {"passed": passed, "control": base, "candidate": score,
                                 "T1_retention": score["T1"] / initial}
        save()
        if passed:
            selected.append(method)
    for method in selected:
        prefix = ROOT / "resume" / method
        second = run(method + "-T2-60", method, True, 60, 2, prefix)
        if second is None:
            continue
        score = second["stage_rows"][-1]["seen_validation"]
        initial = second["stage_rows"][0]["seen_validation"]["T1"]
        if score["T1"] < .5 * initial or score["T2"] < .15:
            STATE["gates"][method]["T3_started"] = False
            STATE["gates"][method]["reason"] = "60-epoch T2 failed retention/acquisition gate"
            save()
            continue
        STATE["gates"][method]["T3_started"] = True
        save()
        run(method + "-T3-60", method, True, 60, 3, ROOT / "jobs" / (method + "-T2-60"))
    STATE.update(status="finished", finished_at=time.time())
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        STATE.update(status="failed", error=str(exc), finished_at=time.time())
        save()
        raise
