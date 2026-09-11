"""Deterministic runtime setup; all training remains in the shared Class main."""
import os
import runpy
import sys
from pathlib import Path


def formal_training_paused(argv):
    if '--output' not in argv:
        return False
    output = Path(argv[argv.index('--output') + 1]).resolve()
    return output.name.endswith('_formal') and (output.parent.parent/'SWEEP_ONLY.json').exists()


def main():
    from setproctitle import setproctitle
    setproctitle('job')
    if formal_training_paused(sys.argv):
        raise SystemExit('Formal training paused by user; sweep-only control is active.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    runpy.run_path(str(Path(__file__).with_name('main.py')), run_name='__main__')


if __name__ == '__main__':
    main()
