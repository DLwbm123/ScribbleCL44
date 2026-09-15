"""Detached single-job supervisor; the job specification is passed by environment."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from setproctitle import setproctitle
from run import ROOT, complete, write

setproctitle('run')
spec = json.loads(os.environ['RUN_SPEC'])
method, gpu = spec['method'], spec['gpu']
receipt = ROOT / 'receipts' / (method + '.json')
free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
if free < 16000:
    write(receipt, dict(status='blocked', reason='less than 16000 MiB free', gpu=gpu))
    raise SystemExit(1)
with (ROOT / 'logs' / (method + '.log')).open('x') as log:
    child = subprocess.Popen([sys.executable, '-u', 'run.py'], cwd=Path(__file__).resolve().parent,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JOB=json.dumps(spec['args'])),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    status = dict(status='running', method=method, gpu=gpu, pid=child.pid,
                  supervisor_pid=os.getpid(), started_at=time.time())
    write(receipt, status)
    code = child.wait()
(ROOT / 'logs' / (method + '.exitcode')).write_text(str(code) + '\n')
try:
    if code:
        raise RuntimeError(f'training exited {code}')
    result = complete(Path(spec['args'][spec['args'].index('--output') + 1]), 80)
    write(receipt, dict(status, status='complete', **result, finished_at=time.time()))
except Exception as error:
    write(receipt, dict(status, status='failed', error=str(error), finished_at=time.time()))
    raise
