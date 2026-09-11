"""Run three bounded EWC diagnostics using the existing shared training loop."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from setproctitle import setproctitle


def main():
    setproctitle('run')
    root = Path.cwd()
    plan = json.loads((root / 'plan.json').read_text())
    event_lock, launch_lock = threading.Lock(), threading.Lock()
    outcomes = {}

    def event(**values):
        with event_lock, (root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(), **values)) + '\n')

    def execute(job):
        name = job['name']
        work = root / name
        output = work / 'output'
        with (work / 'run.log').open('x') as log:
            with launch_lock:
                while int(subprocess.check_output([
                    'nvidia-smi', '-i', '2', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True)) < 14000:
                    time.sleep(30)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                           TMPDIR=plan['tmpdir'], XDG_CACHE_HOME=str(root / 'cache'),
                           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                process = subprocess.Popen([sys.executable, '-u', 'main'], cwd=work, env=env,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                event(name=name, status='started', pid=process.pid, gpu=2, ewc_lambda=job['ewc_lambda'])
                # Wait for real first-batch allocation before admitting another job.
                while process.poll() is None and not (output / 'startup.json').exists():
                    time.sleep(1)
            code = process.wait()
        (work / 'exitcode').write_text(str(code) + '\n')
        result = dict(status='failed', exitcode=code, ewc_lambda=job['ewc_lambda'])
        if code == 0:
            rows = [json.loads(x) for x in (output / 'train.jsonl').read_text().splitlines()]
            train = [x for x in rows if 'loss' in x]
            probes = [x['probe_validation'] for x in rows if 'probe_validation' in x]
            assert len(train) == 10 and train[-1]['iteration'] == 3750 and len(probes) == 10
            result.update(status='complete', best_balanced_validation=json.loads((output / 'probe_best.json').read_text()))
        event(name=name, **result)
        with event_lock:
            outcomes[name] = result
            (root / 'progress.json').write_text(json.dumps(outcomes, indent=2) + '\n')

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute, plan['jobs']))
    success = all(v['status'] == 'complete' for v in outcomes.values())
    (root / 'pipeline.exitcode').write_text('0\n' if success else '1\n')
    event(status='complete' if success else 'finished_with_failures')


if __name__ == '__main__':
    main()
