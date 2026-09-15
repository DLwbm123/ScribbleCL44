"""Read-only checkpoint reevaluation; a finite JSON plan preserves each run's split.

Run from a private work directory containing plan.json as `python run.py`.
The supervisor runs one worker at a time. No training or checkpoint writes occur.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def worker(job):
    import numpy as np
    import torch

    sys.path.insert(0, job['code'])
    import runner_core as core
    if hasattr(core.H5Slices, 'cache_arrays'):
        core.H5Slices.cache_arrays = True
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    checkpoint = Path(job['checkpoint'])
    before = checkpoint.stat()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('checkpoint changed during load')
    state = state.get('model', state)
    scenario = job['scenario']
    if job.get('independent') and scenario == 'class':
        model = core.ClassModel((len(job['tasks'][0]['classes']),))
    else:
        model = core._build_model(scenario)
    for stage in range(job['stage']):
        model.activate_stage(stage)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    del state

    @torch.no_grad()
    def evaluate(task):
        dataset = core.H5Slices(Path(job['data_root']) / task['file'], job['split'],
                                label_shift=task['label_shift'])
        ends = np.asarray(dataset.ends)
        if len(ends) == 0 or ends[-1] + 1 != len(dataset) or np.any(np.diff(ends) <= 0):
            raise ValueError('invalid patient boundaries')
        # This is the project's existing compare_background_dice.py calculation.
        predictions, targets = [], []
        for images, labels in core._loader(dataset, 4, False, 0, 42):
            predictions.append(core.zs_forward(model, images.to(device), task['head'])[
                'pred_masks'].argmax(1).cpu().numpy())
            targets.append(labels.numpy())
        prediction, target = np.concatenate(predictions), np.concatenate(targets)
        classes = [0, *task['classes']]
        rows = []
        for start, end in zip([0, *(ends[:-1] + 1)], ends + 1):
            row = []
            for c in classes:
                p, y = prediction[start:end] == c, target[start:end] == c
                row.append(float((2 * (p & y).sum() + 1e-5) / (p.sum() + y.sum() + 1e-5)))
            rows.append(row)
        values = np.asarray(rows)
        result = dict(background_inclusive_mean=float(values.mean()),
                      background_dice=float(values[:, 0].mean()),
                      foreground_mean=float(values[:, 1:].mean()),
                      per_class_including_background=values.mean(0).tolist(),
                      slices=len(dataset), patients=len(ends), classes=classes)
        dataset.close()
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        assert abs(result['background_inclusive_mean'] -
                   np.mean(result['per_class_including_background'])) < 1e-12
        print(json.dumps(dict(task=task['code'], split=job['split'],
                              background_inclusive_mean=result['background_inclusive_mean'])), flush=True)
        return result

    scores = {task['code']: evaluate(task) for task in job['tasks']}
    differences = {t: abs(scores[t]['foreground_mean'] - expected)
                   for t, expected in job['expected_foreground'].items() if t in scores}
    parity = bool(differences) and max(differences.values()) <= job.get('tolerance', 1e-4)
    result = dict(status='complete' if parity else 'needs_review', job_id=job['id'],
                  run=job['run'], scenario=scenario, split=job['split'], stage=job['stage'],
                  checkpoint=job['checkpoint'], checkpoint_bytes=before.st_size,
                  checkpoint_mtime=before.st_mtime, code=job['code'], data_root=job['data_root'],
                  tasks=scores, background_inclusive_mean=float(np.mean([
                      value['background_inclusive_mean'] for value in scores.values()])),
                  foreground_max_abs_difference=max(differences.values()) if differences else None,
                  foreground_parity_passed=parity, compared_tasks=list(differences),
                  finished_at=time.time(), gpu_peak_mib=torch.cuda.max_memory_reserved()/1024**2)
    # WCD is a separate all-class evaluation, only for completed Class test runs.
    if job.get('whole_class') and parity:
        path = Path(job['data_root']) / 'MMWHS/whole_heart_test.h5'
        if path.exists():
            result['whole_class_dice'] = evaluate(dict(file='MMWHS/whole_heart_test.h5',
                code='whole', classes=list(range(1,8)), label_shift=0, head=None))
    return result


def main():
    try:
        from setproctitle import setproctitle
        setproctitle('run')
    except ImportError:
        pass
    plan = json.loads(Path('plan.json').read_text())
    if 'ITEM_INDEX' in os.environ:
        job = plan['jobs'][int(os.environ['ITEM_INDEX'])]
        try:
            result = worker(job)
        except Exception:
            result = dict(status='failed', job_id=job['id'], run=job['run'],
                          error=traceback.format_exc(), finished_at=time.time())
        write_json(Path('results') / (job['id'] + '.json'), result)
        print(json.dumps({k:result.get(k) for k in ['job_id','status','foreground_max_abs_difference',
                                                  'background_inclusive_mean','error']}), flush=True)
        raise SystemExit(0 if result['status'] == 'complete' else 1)
    Path('results').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    status = dict(status='running', started_at=time.time(), pid=os.getpid(), total=len(plan['jobs']), jobs={})
    write_json('progress.json', status)
    for i, job in enumerate(plan['jobs']):
        output = Path('results') / (job['id'] + '.json')
        if output.exists():
            status['jobs'][job['id']] = json.loads(output.read_text())['status']
            continue
        # Do not wait for unrelated training to free memory or change its state.
        free = int(subprocess.check_output(['nvidia-smi','-i',str(plan['gpu']),
                   '--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True))
        if free < plan['minimum_free_mib']:
            status.update(status='paused_low_memory', free_mib=free)
            break
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['gpu']), ITEM_INDEX=str(i),
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        with (Path('logs')/(job['id']+'.log')).open('w') as log:
            p = subprocess.Popen([sys.executable,'-u','run.py'],env=env,stdout=log,stderr=subprocess.STDOUT)
            status.update(current=job['id'], worker_pid=p.pid)
            write_json('progress.json', status)
            code = p.wait()
        status['jobs'][job['id']] = (json.loads(output.read_text())['status'] if output.exists()
                                    else f'worker_exit_{code}')
        write_json('progress.json', status)
    else:
        status['status'] = 'complete' if all(x == 'complete' for x in status['jobs'].values()) else 'finished_with_gaps'
    status['finished_at'] = time.time()
    write_json('progress.json', status)


if __name__ == '__main__':
    main()
