"""Validation-only 10-epoch screening, then one fresh 40-epoch run per method.

Deploy as source/run.py; launch via ./main -u run.py, with arguments in RUN_JOB.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

from launch_organ_metrics_formal import write_json
from run_independent_domains_formal import run_gpu_queue
from run_comparisons import job_arguments, main as training_entry

METHODS = ('zs-sequential', 'zs-ewc', 'zs-gpm')
RECIPES = (
    dict(name='c0', scales=[1.,.1,.1], bn_from=3, global_weight=1., ewc_lambda=1., gamma=.1, threshold=.97, step=.001),
    dict(name='c1', scales=[.1,.1,.1], bn_from=2, global_weight=1., ewc_lambda=100., gamma=1., threshold=.99, step=.001),
    dict(name='c2', scales=[.03,.03,.03], bn_from=2, global_weight=.1, ewc_lambda=1000., gamma=1., threshold=.999, step=0.),
    dict(name='c3', scales=[.01,.01,.01], bn_from=2, global_weight=1., ewc_lambda=1000., gamma=1., threshold=.999, step=0.),
)


def arguments(root, method, recipe, formal=False):
    args = job_arguments(root,method)
    args.remove('--organ-report-schedule')
    if not formal:
        args.remove('--test-evaluation')
    name = method+'_formal' if formal else method+'_'+recipe['name']
    for flag,value in {'--epochs-per-task':40 if formal else 10,
            '--output':root/'runs'/name, '--lr':.03,
            '--zs-global-weight':recipe['global_weight'], '--ewc-lambda':recipe['ewc_lambda'],
            '--ewc-gamma':recipe['gamma'], '--gpm-threshold':recipe['threshold'],
            '--gpm-threshold-step':recipe['step']}.items():
        args[args.index(flag)+1] = str(value)
    args += ['--organ-tuning','--organ-backbone-lr-scales',*map(str,recipe['scales']),
             '--organ-freeze-bn-from',str(recipe['bn_from'])]
    return args


def check_run(path, epochs, formal):
    summary = json.loads((path/'summary.json').read_text())
    manifest = json.loads((path/'manifest.json').read_text())
    assert summary['completed_stages']==4 and manifest['test_evaluation']==formal
    rows = [json.loads(line) for line in (path/'train.jsonl').read_text().splitlines()]
    for stage in range(4):
        records = [row for row in rows if 'loss' in row and row['stage']==stage]
        assert [row['epoch'] for row in records]==list(range(epochs))
        assert all(math.isfinite(row['loss']) for row in records)
        assert (path/f's{stage+1:02d}_state.pt').stat().st_size>0
    score = summary['final_seen_validation_mean']
    assert math.isfinite(score)
    if not formal:
        assert summary['final_seen_mean'] is None
        assert all(value is None for row in summary['matrix'] for value in row)
    return dict(validation_mean=score,validation_matrix=summary['validation_matrix'],
                test_mean=summary['final_seen_mean'] if formal else None,
                test_matrix=summary['matrix'] if formal else None)


def select_recipe(candidates):
    if len(candidates)!=len(RECIPES) or not all(row['status']=='complete' for row in candidates):
        raise ValueError('all four complete validation candidates are required')
    return max(candidates,key=lambda row: (row['validation_mean'],-int(row['recipe']['name'][1:])))['recipe']


def main():
    if os.environ.get('RUN_JOB'):
        training_entry()
        return
    root = Path(__file__).resolve().parent.parent
    assert root.is_relative_to(Path('/data_nas')) and (root/'checks/passed.json').exists()
    with (root/'coordinator.started').open('x') as f:
        f.write(str(os.getpid()))
    jobs = [dict(name=method+'_'+recipe['name'],method=method,recipe=recipe)
            for recipe in RECIPES for method in METHODS]
    outcomes = {job['name']:dict(status='queued',method=job['method'],recipe=job['recipe']) for job in jobs}
    for method in METHODS:
        outcomes[method+'_formal'] = dict(status='waiting_for_selection',method=method)
    selections = {}
    lock = threading.Lock()
    write_json(root/'plan.json',dict(methods=METHODS,recipes=RECIPES,screen_epochs_per_task=10,
        formal_epochs_per_task=40,gpus=[4,5,6,7],minimum_free_mib=22000,
        spatial_weight=0,lr=.03,selection='final T1-T4 foreground validation mean; c0-first tie break',
        screening_test_evaluation=False,formal_from_scratch=True,jobs=jobs))
    def save_status():
        write_json(root/'progress.json',dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),jobs=outcomes))
    save_status()
    def execute(job,gpu,formal=False):
        name,method,recipe = job['name'],job['method'],job['recipe']
        try:
            args=arguments(root,method,recipe,formal)
            with (root/'logs'/f'{name}.log').open('x') as log:
                p=subprocess.Popen(['./main','-u','run.py'],cwd=root/'source',
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),RUN_JOB=json.dumps(args)),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                with lock:
                    outcomes[name]=dict(status='running',gpu=gpu,pid=p.pid,method=method,recipe=recipe)
                    save_status()
                code=p.wait()
            (root/'logs'/f'{name}.exitcode').write_text(str(code)+'\n')
            if code:
                raise RuntimeError(f'process exited {code}')
            result=check_run(root/'runs'/name,40 if formal else 10,formal)
            with lock:
                outcomes[name]=dict(status='complete',gpu=gpu,method=method,recipe=recipe,**result)
                save_status()
        except Exception as error:
            with lock:
                outcomes[name]=dict(status='failed',gpu=gpu,method=method,recipe=recipe,error=str(error))
                save_status()
            return
        if formal:
            return
        # A method's last completed candidate selects its formal recipe exactly once.
        with lock:
            candidates=[outcomes[method+'_'+r['name']] for r in RECIPES]
            ready=all(row['status']=='complete' for row in candidates) and method not in selections
            if ready:
                selections[method]=select_recipe(candidates)
                write_json(root/'selections.json',dict(selected_on='validation only',recipes=selections,
                    candidates={m:[outcomes[m+'_'+r['name']] for r in RECIPES] for m in selections}))
        if ready:
            execute(dict(name=method+'_formal',method=method,recipe=selections[method]),gpu,formal=True)
    run_gpu_queue(jobs,execute,min_free_memory_mib=22000)
    with lock:
        for method in METHODS:
            if outcomes[method+'_formal']['status']=='waiting_for_selection':
                outcomes[method+'_formal']['status']='blocked_incomplete_screening'
        save_status()
    complete=all(outcomes[m+'_formal']['status']=='complete' for m in METHODS)
    write_json(root/'comparison.json',dict(complete=complete,selections=selections,jobs=outcomes))
    (root/'pipeline.exitcode').write_text('0\n' if complete else '1\n')
    if not complete:
        raise SystemExit(1)


if __name__=='__main__':
    main()
