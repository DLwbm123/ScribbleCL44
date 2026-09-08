"""Deterministic runtime setup; all training remains in the shared Class main."""
import os
import runpy
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
torch.set_num_threads(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
runpy.run_path(str(Path(__file__).with_name('main.py')), run_name='__main__')
