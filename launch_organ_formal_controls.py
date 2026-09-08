"""Launch three matched Organ continuation controls beside the selected small-alpha run."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


CONFIGS = [('alpha0', 4, 0., .01), ('alpha01', 5, .1, .01), ('no_spatial', 6, .05, 0.)]


def commands(base, output):
    jobs = []
    for name, gpu, alpha, spatial in CONFIGS:
        command = list(base)
        for flag, value in [('--output', output / name), ('--organ-t2-feature-alpha', alpha),
                            ('--zs-spatial-loss-weight', spatial),
                            ('--annotation-id', 'organ_T13_half_train_seed42_control_' + name)]:
            assert command.count(flag) == 1, flag
            command[command.index(flag) + 1] = str(value)
        jobs.append(dict(name=name, gpu=gpu, t2_alpha=alpha, spatial_weight=spatial, command=command))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    reference = json.loads((root / 'small_alpha_checks_20260908/formal_launch.json').read_text())
    output = root / 'formal_controls_20260908'
    jobs = commands(reference['command'], output)
    assert len({j['gpu'] for j in jobs}) == 3
    assert all(j['command'][j['command'].index('--epochs-per-task') + 1] == '60' and
               j['command'][j['command'].index('--t2-from') + 1] == str(root / 'run60')
               for j in jobs)
    if args.dry_run:
        print(json.dumps(dict(source=reference['source'], jobs=jobs), indent=2))
        return
    if output.exists():
        raise FileExistsError(output)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    free = {int(gpu): int(memory) for gpu, memory in (r.split(',') for r in rows.splitlines())}
    if any(free[j['gpu']] < 16000 for j in jobs):
        raise RuntimeError(f'Each selected GPU requires 16000 MiB free: {free}')
    if shutil.disk_usage(root).free < 60 * 1024**3:
        raise OSError('Require 60 GiB free for three formal continuation runs')
    output.mkdir()
    probe = output / '.write_probe'
    probe.write_text('ok')
    assert probe.read_text() == 'ok'
    probe.unlink()
    (output / 'logs').mkdir()
    launched = []
    for job in jobs:
        name = job['name']
        script = output / (name + '.sh')
        script.write_text('#!/bin/bash\nset -euo pipefail\n' +
            "trap 'TASK_EXIT=$?; echo \"$TASK_EXIT\" > " +
            shlex.quote(str(output / (name + '.exitcode'))) + "' EXIT\n" +
            shlex.join(job['command']) + '\n')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        log_path = output / 'logs' / (name + '.log')
        with log_path.open('x') as log:
            process = subprocess.Popen(['bash', str(script)], cwd=reference['source'], env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launched.append(dict(**job, wrapper_pid=process.pid, log=str(log_path)))
        (output / 'launch.json').write_text(json.dumps(dict(
            time=time.strftime('%Y-%m-%d %H:%M:%S %z'), source=reference['source'],
            training_code_commit='65ab7ef8783a687ac9787f5419563f5491422b4e',
            reference_output=reference['output'], jobs=launched), indent=2) + '\n')
        print(json.dumps(launched[-1]), flush=True)


if __name__ == '__main__':
    main()
