"""Neutral entry; runtime arguments travel through the environment."""
import json
import os
import sys
from setproctitle import setproctitle

setproctitle("job")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import runner_core

torch.set_num_threads(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
sys.argv = ["main", *json.loads(os.environ["JOB_ARGS"])]
runner_core.main("class")

