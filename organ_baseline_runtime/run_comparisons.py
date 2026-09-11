"""Run the five Domain-table controls through the shared Organ training entry.

Deploy this file as source/run.py and invoke the environment's Python as ./main.
Configuration travels in the environment, never in visible process arguments.
"""
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time

from launch_organ_metrics_formal import write_json
from run_independent_domains_formal import run_gpu_queue


METHODS = ('zs-gpm', 'zs-ewc', 'zs-sequential', 'dense-sequential', 'pce-sequential')


def job_arguments(root, method):
    return ['--setting-run', '--data-root', str(root/'inputs/data'),
            '--sparse-root', str(root/'inputs/sparse'), '--output', str(root/'runs'/method),
            '--method', method, '--device', 'cuda:0', '--seed', '42', '--batch-size', '4',
            '--workers', '8', '--epochs-per-task', '60', '--lr', '.03',
            '--organ-report-schedule', '--organ-t2-supervision-strategy',
            '--organ-t2-feature-alpha', '.05', '--organ-retention-strategy',
            '--zs-global-weight', '1' if method.startswith('zs-') else '0',
            '--zs-spatial-loss-weight', '0', '--validate-each-epoch', '--test-evaluation',
            '--cache-h5', '--numerical-debug', '--record-source-ids',
            '--annotation-id', 'organ_T134_half_train_seed42',
            '--ewc-lambda', '1', '--ewc-gamma', '.1', '--fisher-batches', '50',
            '--gpm-threshold', '.97', '--gpm-threshold-step', '.001', '--gpm-examples', '16',
            '--gpm-max-patches-per-layer', '4096', '--gpm-max-matrix-elements', '4000000']


def completed(output):
    summary = json.loads((output/'summary.json').read_text())
    assert summary['completed_stages'] == 4
    rows = [json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
    for stage, budget in enumerate((60, 60, 10, 10)):
        epochs = [row for row in rows if 'loss' in row and row['stage'] == stage]
        assert [row['epoch'] for row in epochs] == list(range(budget))
        assert all(math.isfinite(row['loss']) for row in epochs)
        assert (output/f's{stage+1:02d}_state.pt').stat().st_size > 0
    matrix = summary['matrix']
    params = [row['model_parameters'] for row in summary['stage_rows']]
    return dict(test_mean=summary['final_seen_mean'], validation_mean=summary['final_seen_validation_mean'],
                matrix=matrix, BWTR=sum((matrix[3][i]-matrix[i][i])/matrix[i][i] for i in range(3))/3
                if all(matrix[i][i] > 0 for i in range(3)) else None,
                MPE=sum(params[i]-params[i-1] for i in range(1,4))/(3*params[0]),
                DRR=None if summary['method']=='zs-gpm' else 0.,
                RMA=None, metric='foreground per-case macro Dice')


def main():
    if os.environ.get('RUN_JOB'):
        import torch
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        sys.argv = ['main.py', *json.loads(os.environ['RUN_JOB'])]
        runpy.run_path('main.py', run_name='__main__')
        return
    root = Path(__file__).resolve().parent.parent
    assert root.is_relative_to(Path('/data_nas'))
    assert (root/'checks/passed.json').is_file()
    with (root/'coordinator.started').open('x') as stream:
        stream.write(str(os.getpid()))
    jobs = [dict(name=method, arguments=job_arguments(root, method)) for method in METHODS]
    write_json(root/'plan.json', dict(jobs=jobs, gpus=[4,5,6,7], minimum_free_mib=22000,
        epochs=[60,60,10,10], lr=[.03,.03,.06,.06], zs_spatial=[.01,.01,0,0],
        checkpoint_selection='current-task foreground validation only',
        reference='existing provisional spatial0; comparison controls start from scratch'))
    outcomes = {method:dict(status='queued') for method in METHODS}
    lock = threading.Lock()
    def status(name, **fields):
        with lock:
            outcomes[name] = fields
            write_json(root/'progress.json',dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),jobs=outcomes))
    def execute(job, gpu):
        name = job['name']
        try:
            with (root/'logs'/f'{name}.log').open('x') as stream:
                process = subprocess.Popen(['./main','-u','run.py'], cwd=root/'source',
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), RUN_JOB=json.dumps(job['arguments'])),
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                status(name,status='running',gpu=gpu,pid=process.pid)
                code = process.wait()
            (root/'logs'/f'{name}.exitcode').write_text(str(code)+'\n')
            if code:
                raise RuntimeError(f'training exited {code}')
            status(name,status='complete',gpu=gpu,**completed(root/'runs'/name))
        except Exception as error:
            status(name,status='failed',gpu=gpu,error=str(error))
    run_gpu_queue(jobs, execute, min_free_memory_mib=22000)
    ok = all(row['status']=='complete' for row in outcomes.values())
    write_json(root/'comparison.json',dict(complete=ok, jobs=outcomes,
        reference=json.loads((root/'report_reference.json').read_text())))
    (root/'pipeline.exitcode').write_text('0\n' if ok else '1\n')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
