"""Bounded throughput measurement using the existing guarded Organ CL loop."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--sparse-root', type=Path, required=True)
    p.add_argument('--physical-gpu', type=int, required=True)
    p.add_argument('--spatial', choices=['off', 'on'], required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--train-batches', type=int, default=12)
    p.add_argument('--max-task', type=int, choices=[1, 2, 3])
    p.add_argument('--full-buffer-only', action='store_true')
    args = p.parse_args()
    if args.batch_size < 1 or args.train_batches < 5:
        p.error('batch-size must be positive and train-batches at least 5')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.physical_gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    sys.path.insert(0, str(args.source.resolve()))
    import runner_core as r
    from numerical_safety import _cpu_copy
    assert Path(r.__file__).resolve() == (args.source/'runner_core.py').resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    measurements, evaluations = [], []
    batch_start = None
    replay_tasks = []
    original_loader = r._loader
    original_add = r.DarkExperienceReplayPlus.add_data
    original_sample = r.DarkExperienceReplayPlus.sample
    original_evaluate = r.evaluate
    original_load = torch.load
    snapshots = {}

    def save(value, path, *a, **kw):
        # The real loop reloads only its selected paired state. Keep that state
        # in RAM for throughput measurement; redundant export writes are skipped.
        key = str(path)
        if key.endswith('_best_state.pt'):
            snapshots[key] = _cpu_copy(value)

    def load(path, *a, **kw):
        key = str(path)
        return snapshots.pop(key) if key in snapshots else original_load(path, *a, **kw)

    def loader(dataset, *a):
        actual = original_loader(dataset, *a)
        if dataset.split != 'train':
            return actual
        class TimedLoader:
            def __len__(self):
                return len(actual)
            def __iter__(self):
                nonlocal batch_start
                iterator = iter(actual)
                while True:
                    batch_start = time.monotonic()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        return
                    yield batch
        return TimedLoader()

    def sample(buffer, device):
        nonlocal replay_tasks
        result = original_sample(buffer, device)
        replay_tasks = torch.unique(result[3]).tolist()
        return result

    def add(buffer, examples, features, labels, task_ids, classes):
        before = len(buffer)
        original_add(buffer, examples, features, labels, task_ids, classes)
        torch.cuda.synchronize()
        measurements.append({'stage': int(task_ids[0]), 'seconds': time.monotonic()-batch_start,
            'buffer_before': before, 'replay_tasks': replay_tasks if before else []})

    def evaluate(*a, **kw):
        torch.cuda.synchronize()
        start = time.monotonic()
        result = original_evaluate(*a, **kw)
        torch.cuda.synchronize()
        evaluations.append({'slices': len(a[1].dataset), 'seconds': time.monotonic()-start})
        return result

    max_task = args.max_task or (3 if args.spatial == 'off' else 2)
    argv = ['benchmark', '--data-root', str(args.data_root),
        '--sparse-root', str(args.sparse_root),
        '--output', str(args.output/'run'), '--device', 'cuda:0', '--seed', '42',
        '--epochs-per-task', '1', '--max-task', str(max_task), '--max-train-batches', str(args.train_batches),
        '--batch-size', str(args.batch_size), '--workers', '8', '--lr', '.03', '--method', 'zs-derpp',
        '--der-alpha', '.5', '--der-beta', '.5', '--der-buffer-size', '128', '--der-minibatch-size', '4',
        '--pce-loss-weight', '1', '--zs-global-weight', '1', '--zs-spatial-loss-weight', '.01',
        '--zs-spatial-warmup-epochs', '80' if args.spatial == 'off' else '-1', '--validate-each-epoch']
    start = time.monotonic()
    telemetry_path = args.output/'private_gpu_pmon.log'
    with telemetry_path.open('w') as stream:
        telemetry = subprocess.Popen(['nvidia-smi', 'pmon', '-i', str(args.physical_gpu), '-s', 'u', '-d', '1'],
                                      stdout=stream, stderr=subprocess.STDOUT)
        failure = None
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        try:
            with patch.object(r, '_loader', loader), patch.object(r, 'evaluate', evaluate), \
                 patch.object(r.DarkExperienceReplayPlus, 'add_data', add), \
                 patch.object(r.DarkExperienceReplayPlus, 'sample', sample), patch.object(sys, 'argv', argv), \
                 patch.object(torch, 'save', save), patch.object(torch, 'load', load):
                r.main('organ')
        except Exception as error:
            failure = repr(error)
        finally:
            telemetry.terminate()
            telemetry.wait()
    other_sm = []
    for line in telemetry_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) != os.getpid() and parts[3].isdigit():
            other_sm.append(int(parts[3]))
    stages = []
    for stage in range(max_task):
        rows = [x for x in measurements if x['stage'] == stage]
        steady = [x for x in rows[4:] if not args.full_buffer_only or x['buffer_before'] == 128]
        if steady:
            stages.append({'task': f'T{stage+1}', 'updates_completed': len(rows), 'timed_updates_after_warmup': len(steady),
                'mean_step_seconds': statistics.mean(x['seconds'] for x in steady),
                'median_step_seconds': statistics.median(x['seconds'] for x in steady),
                'mean_replay_task_groups': statistics.mean(len(x['replay_tasks']) for x in steady),
                'buffer_size_before_last_insert': rows[-1]['buffer_before']})
    report = {'status': 'failed' if failure else 'complete', 'failure': failure,
        'batch_size': args.batch_size, 'replay_batch_size': 4,
        'full_buffer_only': args.full_buffer_only, 'completed_updates': len(measurements),
        'largest_buffer_before_completed_update': max((x['buffer_before'] for x in measurements), default=0),
        'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
        'peak_cuda_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
        'spatial': args.spatial, 'physical_gpu': args.physical_gpu, 'elapsed_seconds': time.monotonic()-start,
        'scope': f'real CL loop; 1 short epoch x {args.train_batches} updates per task; validation-only; no formal training',
        'checkpoint_storage': 'selected paired states in CPU RAM; final tensor exports skipped for timing',
        'timing_scope': 'batch retrieval through feature-target capture and buffer insertion; excludes evaluation/checkpoint writes',
        'other_process_sm_mean_when_observed': statistics.mean(other_sm) if other_sm else 0,
        'other_process_sm_max': max(other_sm, default=0), 'stages': stages, 'evaluation_seconds': evaluations}
    (args.output/'timing.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if failure:
        raise RuntimeError(failure)
    assert all(row['updates_completed'] == args.train_batches for row in stages) and len(stages) == max_task


if __name__ == '__main__':
    main()
