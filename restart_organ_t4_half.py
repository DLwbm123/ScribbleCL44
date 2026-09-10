"""Restart only T4 independent; retain three running jobs and aggregate the amended campaign."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from launch_organ_metrics_formal import commands, check_completed, collect_metrics, write_json
from run_independent_domains_formal import gpu_free_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    root, parent = args.root.resolve(), args.parent.resolve()
    assert root.is_relative_to(Path('/data_nas')) and parent.is_relative_to(Path('/data_nas'))
    assert (root/'amendment.json').is_file()
    with (root/'coordinator.started').open('x') as f:
        f.write(str(os.getpid()))
    jobs = {j['name']: j for j in commands(root)}
    job = jobs['ind_T4']
    cmd = job['command']
    cmd[cmd.index('--annotation-id')+1] = 'organ_T134_half_train_seed42'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
    for key, directory in {'TMPDIR':'tmp', 'XDG_CACHE_HOME':'cache', 'TORCH_HOME':'cache/torch',
                           'CUDA_CACHE_PATH':'cache/cuda', 'TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor',
                           'MPLCONFIGDIR':'cache/matplotlib'}.items():
        (root/directory).mkdir(parents=True, exist_ok=True)
        env[key] = str(root/directory)
    outcomes = {name: dict(status='queued') for name in jobs}
    write_json(root/'launch_plan.json', dict(data_variant='T1/T3/T4 half, T2 full',
        adopted_jobs=['organ_retention','organ_no_replay','ind_T3'], parent=str(parent),
        restarted_job=job, min_free_mib=22000, class_jobs=0))
    while gpu_free_memory().get(7, 0) < 22000:
        time.sleep(10)
    with (root/'logs/ind_T4.log').open('x') as log:
        process = subprocess.Popen(cmd, cwd=root/'source', env=env, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        outcomes['ind_T4'] = dict(status='running', gpu=7, pid=process.pid)
        previous = None
        while True:
            inherited = json.loads((parent/'progress.json').read_text())['jobs']
            for name in jobs:
                if outcomes[name]['status'] in ('complete', 'failed'):
                    continue
                state = (dict(inherited[name], source=str(parent/'runs'/name)) if name != 'ind_T4'
                         else dict(outcomes[name]))
                if name == 'ind_T4' and process.poll() is not None:
                    (root/'logs/ind_T4.exitcode').write_text(str(process.returncode)+'\n')
                    state['status'] = 'complete' if process.returncode == 0 else 'failed'
                    state['exit_code'] = process.returncode
                if state['status'] == 'complete':
                    try:
                        state.update(check_completed(root, jobs[name]))
                        summary = json.loads((root/'runs'/name/'summary.json').read_text())
                        if name != 'ind_T3':
                            assert summary['stage_rows'][-1]['train_samples'] == 982
                    except Exception as error:
                        state.update(status='failed', error=str(error))
                outcomes[name] = state
            snapshot = json.dumps(outcomes, sort_keys=True)
            if snapshot != previous:
                write_json(root/'progress.json', dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                    data_variant='T1/T3/T4 half, T2 full', jobs=outcomes))
                previous = snapshot
            if all(v['status'] in ('complete', 'failed') for v in outcomes.values()):
                break
            time.sleep(30)
    collect_metrics(root, outcomes, data_variant='T1/T3/T4 half, T2 full')
    success = all(v['status'] == 'complete' for v in outcomes.values())
    (root/'pipeline.exitcode').write_text('0\n' if success else '1\n')
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
