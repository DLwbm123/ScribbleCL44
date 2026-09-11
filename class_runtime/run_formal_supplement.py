"""Class comparison methods and fixed independent references, without a new sweep."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time

from run_sweeps import audit_run


JOBS = [
    dict(name='pce_sequential', method='pce-sequential', spatial=0, global_weight=0),
    dict(name='zs_sequential', method='zs-sequential', spatial=0, global_weight=.1),
    dict(name='dense_sequential', method='dense-sequential', spatial=0, global_weight=0),
    dict(name='zs_ewc', method='zs-ewc', spatial=0, global_weight=.1),
    dict(name='zs_gpm', method='zs-gpm', spatial=0, global_weight=.1),
    dict(name='main_s0', method='zs-derpp-mib', spatial=0, global_weight=.1),
    *[dict(name=f'ind_{task}_s0', method='zs-sequential', spatial=0, global_weight=.1, task=task)
      for task in ['T2', 'T3']],
]


def metrics(summary, references=None):
    rows = summary['stage_rows']
    matrix = [[row['evaluated'][f'T{i+1}']['background_inclusive_mean']
               for i in range(t+1)] for t, row in enumerate(rows)]
    params = summary['parameter_counts']
    replay = summary['derpp_buffer']
    counts = {} if replay is None else replay['replayed_unique_source_counts']
    if summary['method'].endswith('-gpm'):
        counts = {str(i): rows[i]['gpm']['examples'] for i in [0, 1]}
    return dict(
        background_inclusive_matrix=matrix,
        A_Dice=statistics.mean(matrix[-1]),
        BWTR=statistics.mean((matrix[-1][i]-matrix[i][i])/matrix[i][i] for i in range(2)),
        RMA=None if references is None else statistics.mean(matrix[i][i]/references[i-1] for i in [1, 2]),
        WCD=summary['whole_class_dice']['background_inclusive_mean'],
        MPE=statistics.mean((params[i]-params[i-1])/params[0] for i in [1, 2]),
        DRR=statistics.mean(counts.get(str(i), 0)/summary['source_train_sizes'][str(i)] for i in [0, 1]),
        DRR_raw_image_replay=replay is not None,
    )


def main():
    from setproctitle import setproctitle
    setproctitle('run')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--sparse-root', type=Path, required=True)
    parser.add_argument('--minimum-free-mib', type=int, default=36000)
    args = parser.parse_args()
    if args.minimum_free_mib < 1:
        parser.error('--minimum-free-mib must be positive')
    root, source = args.root.resolve(), Path(__file__).resolve().parent
    for name in ['jobs', 'logs']:
        (root/name).mkdir(exist_ok=True)
    (root/'coordinator.lock').write_text(str(os.getpid()))
    plan = dict(jobs=JOBS, gpus=[2, 3], minimum_free_mib=args.minimum_free_mib, epochs_per_task=80, seed=42, batch_size=4,
                learning_rate=.03, pce=1, global_weight=.1, spatial_first_epoch=35,
                selection='foreground validation Dice, all 80 epochs', reporting='background-inclusive',
                independent_reference='fixed common ZS T2/T3 reference; T1 excluded from RMA',
                priority='comparison methods first; positive-Spatial ablations deferred',
                DRR_unit='unique original training slice reused in a later task, grouped by source task')
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    gpus = queue.Queue()
    for gpu in [2, 3]:
        gpus.put(gpu)
    lock = threading.Lock()
    outcomes = {}

    def event(**values):
        with lock, (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(), **values))+'\n')

    def execute(job):
        gpu = gpus.get()
        try:
            while int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True)) < args.minimum_free_mib:
                time.sleep(30)
            output = root/'jobs'/job['name']
            if output.exists():
                raise FileExistsError(output)
            command = [sys.executable, '-u', str(source/'run_class_job.py'), '--setting-run',
                '--data-root', str(args.data_root), '--sparse-root', str(args.sparse_root),
                '--output', str(output), '--device', 'cuda:0', '--method', job['method'],
                '--seed', '42', '--epochs-per-task', '80', '--batch-size', '4', '--workers', '4',
                '--lr', '.03', '--pce-loss-weight', '1', '--zs-global-weight', str(job['global_weight']),
                '--zs-spatial-loss-weight', str(job['spatial']), '--zs-spatial-warmup-epochs', '33',
                '--validate-every', '375', '--mib-kd-weight', '1', '--der-alpha', '.5',
                '--der-beta', '.5', '--der-buffer-size', '64', '--der-minibatch-size', '4']
            if 'task' in job:
                command += ['--class-independent-task', job['task']]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                       TMPDIR=str((root/'tmp').resolve()), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            with (root/'logs'/f"{job['name']}.log").open('x') as log:
                process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                event(status='started', name=job['name'], gpu=gpu, pid=process.pid)
                code = process.wait()
            (root/'logs'/f"{job['name']}.exitcode").write_text(str(code)+'\n')
            if code:
                raise RuntimeError(f'exit {code}')
            audit_run(output, 80, 1 if 'task' in job else 3, job['spatial'] > 0)
            result = dict(status='complete', gpu=gpu)
        except Exception as error:
            result = dict(status='failed', gpu=gpu, error=str(error))
        finally:
            gpus.put(gpu)
        event(name=job['name'], **result)
        with lock:
            outcomes[job['name']] = result
            (root/'progress.json').write_text(json.dumps(outcomes, indent=2)+'\n')

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute, JOBS))
    results = {}
    for job in [j for j in JOBS if 'task' not in j]:
        if outcomes[job['name']]['status'] != 'complete':
            continue
        suffix = 's001' if job['spatial'] else 's0'
        ref_names = [f'ind_T{i}_{suffix}' for i in [2, 3]]
        refs = None
        if all(outcomes[name]['status'] == 'complete' for name in ref_names):
            refs = [json.loads((root/'jobs'/name/'summary.json').read_text())['stage_rows'][0]
                    ['evaluated'][f'T{i}']['background_inclusive_mean'] for i, name in zip([2, 3], ref_names)]
        summary = json.loads((root/'jobs'/job['name']/'summary.json').read_text())
        results[job['name']] = dict(**metrics(summary, refs), RMA_references=ref_names)
    (root/'benchmark_metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    success = all(v['status'] == 'complete' for v in outcomes.values())
    (root/'pipeline.exitcode').write_text('0\n' if success else '1\n')
    event(status='complete' if success else 'finished_with_failures')
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
