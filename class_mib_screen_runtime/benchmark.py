"""Time the unchanged T2 training loop; exclude five warmup intervals and validation."""
import json
import os
from pathlib import Path
import statistics
import sys
import time
from setproctitle import setproctitle

setproctitle("job")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import runner_core

torch.set_num_threads(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
stamps = []
original = torch.optim.SGD.zero_grad


def timed_zero_grad(self, *args, **kwargs):
    torch.cuda.synchronize()
    stamps.append(time.perf_counter())
    return original(self, *args, **kwargs)


torch.optim.SGD.zero_grad = timed_zero_grad
sys.argv = ["main", *json.loads(os.environ["JOB_ARGS"])]
started = time.perf_counter()
runner_core.main("class")
wall = time.perf_counter() - started
durations = [b - a for a, b in zip(stamps, stamps[1:])][5:]
assert len(stamps) == 30 and len(durations) == 24
output = Path(sys.argv[sys.argv.index("--output") + 1])
rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
rows = [r for r in rows if r.get("stage") == 1 and "loss" in r]
assert len(rows) == 1 and rows[0]["iteration"] == 30 and rows[0]["mib_kd_loss"] > 0
result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
              updates=30, warmup_intervals=5, measured_intervals=len(durations),
              median_seconds_per_batch=statistics.median(durations),
              mean_seconds_per_batch=statistics.mean(durations),
              images_per_second=4 / statistics.mean(durations), durations_seconds=durations,
              total_seconds_including_load_and_evaluation=wall,
              peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
(output / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
print("TIMING", json.dumps(result), flush=True)
