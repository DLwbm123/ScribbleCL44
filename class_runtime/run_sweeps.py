"""Bounded Class sweeps followed by validation-selected formal training."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time


def audit_run(output, epochs, stages, spatial):
    manifest = json.loads((output/'manifest.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    rows = [json.loads(x) for x in (output/'train.jsonl').read_text().splitlines()]
    rows = [r for r in rows if 'loss' in r]
    assert summary['completed_stages'] == stages
    for stage in range(stages):
        r = [x for x in rows if x['stage'] == stage]
        assert [x['epoch'] for x in r] == list(range(epochs))
        assert r[-1]['iteration'] == epochs * 375
        assert all(math.isfinite(x['loss']) for x in r)
        if spatial:
            assert any(x['zs_spatial_loss'] > 0 for x in r[34:])
            assert summary['stage_rows'][stage]['best_validation']['epoch'] + 1 >= manifest['selection_min_epoch']
    score = float(summary['final_seen_validation_mean'])
    assert math.isfinite(score)
    return {'validation_score': score, 'test_mean': summary['final_seen_mean'],
            'matrix': summary['matrix'], 'epochs_per_task': epochs, 'stages': stages}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--sparse-root', type=Path, required=True)
    p.add_argument('--formal-epochs', type=int, default=80)
    p.add_argument('--gpus', type=int, nargs='+', default=[2, 3])
    p.add_argument('--minimum-free-mib', type=int, default=32000)
    args = p.parse_args()
    assert args.formal_epochs >= 35
    assert args.gpus and len(set(args.gpus)) == len(args.gpus) and min(args.gpus) >= 0
    root = args.root.resolve()
    source = Path(__file__).resolve().parent
    (root/'logs').mkdir(exist_ok=True)
    (root/'jobs').mkdir(exist_ok=True)
    (root/'best_models').mkdir(exist_ok=True)
    with (root/'coordinator.lock').open('x') as f:
        f.write(str(os.getpid()))
    gpus = queue.Queue()
    for gpu in args.gpus:
        gpus.put(gpu)
    lock = threading.Lock()

    def event(**values):
        with lock, (root/'events.jsonl').open('a') as f:
            f.write(json.dumps({'time': time.strftime('%Y-%m-%d %H:%M:%S %z'), **values})+'\n')

    def execute(job):
        gpu = gpus.get()
        try:
            while True:
                free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                if free >= args.minimum_free_mib:
                    break
                time.sleep(30)
            output = root/'jobs'/job['name']
            assert not output.exists(), 'Refusing to overwrite an existing run'
            formal = job['phase'] == 'formal'
            independent = job['group'].startswith('ind_')
            stages = 1 if independent else 3
            epochs = args.formal_epochs if formal else (20 if independent else 40)
            command = [sys.executable, '-u', str(source/'run_class_job.py'), '--setting-run',
                '--data-root', str(args.data_root), '--sparse-root', str(args.sparse_root),
                '--output', str(output), '--device', 'cuda:0', '--seed', '42',
                '--epochs-per-task', str(epochs), '--batch-size', '4', '--workers', '4',
                '--lr', str(job['lr']), '--pce-loss-weight', '1', '--zs-global-weight', '.1',
                '--zs-spatial-loss-weight', str(job['spatial']), '--zs-spatial-warmup-epochs', '33',
                '--validate-every', '375']
            if independent:
                command += ['--method', 'zs-sequential', '--class-independent-task', job['group'][4:]]
            else:
                command += ['--method', 'zs-derpp-mib', '--max-task', str(stages), '--mib-kd-weight', '1',
                    '--der-alpha', '.5', '--der-beta', '.5', '--der-buffer-size', '64', '--der-minibatch-size', '4']
            if not formal:
                command += ['--validation-only']
                if not independent:
                    command += ['--selection-min-epoch', '35']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
            with (root/'logs'/(job['name']+'.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                event(status='started', gpu=gpu, pid=process.pid, **job)
                code = process.wait()
            (root/'logs'/(job['name']+'.exitcode')).write_text(str(code)+'\n')
            if code:
                raise RuntimeError(f'run exited {code}; inspect its private log')
            result = dict(job, status='complete', **audit_run(output, epochs, stages, job['spatial'] > 0))
            event(**result)
            return result
        except Exception as error:
            result = dict(job, status='failed', failure=str(error))
            event(**result)
            return result
        finally:
            gpus.put(gpu)

    candidates = [dict(name=f'ind_{task}_lr{lr:g}', group=f'ind_{task}', phase='sweep', lr=lr, spatial=0)
                  for lr in [.01, .03] for task in ['T1', 'T2', 'T3']]
    candidates += [dict(name=f'cl_spatial_{weight:g}', group='cl_spatial', phase='sweep', lr=.03, spatial=weight)
                   for weight in [.001, .01]]
    plan = {'formal_epochs_per_task': args.formal_epochs, 'sweep_candidates': candidates,
            'selection_split': 'validation', 'spatial_first_epoch': 35,
            'cl_spatial_pilot_tasks': ['T1', 'T2', 'T3'], 'cl_spatial_pilot_epochs': 40,
            'cl_spatial_pilot_selection_window': [35, 40], 'independent_sweep_epochs': 20,
            'automatic_formal_training': True, 'gpu_pool': args.gpus}
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    outcomes, selections, decided = [], {}, set()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        pending = list(candidates)
        futures = {}
        while futures or pending:
            while pending and len(futures) < len(args.gpus):
                job = pending.pop(0)
                futures[pool.submit(execute, job)] = job
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                result = future.result()
                outcomes.append(result)
                group = job['group']
                if job['phase'] == 'sweep' and group not in decided:
                    group_results = [r for r in outcomes if r['group'] == group and r['phase'] == 'sweep']
                    if len(group_results) == 2:
                        decided.add(group)
                        if all(r['status'] == 'complete' for r in group_results):
                            best = max(group_results, key=lambda r:r['validation_score'])
                            selections[group] = best
                            formal = dict(name=group+'_formal', group=group, phase='formal', lr=best['lr'], spatial=best['spatial'])
                            pending.insert(0, formal)
                        else:
                            event(status='formal_blocked_by_failed_candidate', group=group)
                if job['phase'] == 'formal' and result['status'] == 'complete' and group.startswith('ind_'):
                    task = group[4:]
                    (root/'best_models'/(task+'.pt')).symlink_to(Path('../jobs')/job['name']/'s01_best.pt')
                state = {'status':'running' if futures or pending else 'finished', 'selections':selections, 'results':outcomes}
                temporary = root/'progress.tmp'
                temporary.write_text(json.dumps(state, indent=2)+'\n')
                temporary.replace(root/'progress.json')
    success = len([r for r in outcomes if r['phase']=='formal' and r['status']=='complete']) == 4
    (root/'pipeline.exitcode').write_text('0\n' if success else '1\n')
    event(status='complete' if success else 'finished_with_failures')
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
