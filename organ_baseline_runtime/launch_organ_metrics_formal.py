"""Four approved Organ jobs; reuse the existing memory-aware GPU 4-7 queue."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from run_independent_domains_formal import run_gpu_queue


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def commands(root):
    base = [sys.executable, '-u', '-c',
            'import torch; import runner_core; torch.set_num_threads(4); '
            'torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; '
            'torch.use_deterministic_algorithms(True,warn_only=True); runner_core.main("organ")',
            '--data-root', str(root/'inputs/data'), '--sparse-root', str(root/'inputs/sparse'),
            '--device', 'cuda:0', '--seed', '42', '--batch-size', '4', '--workers', '8', '--lr', '.03',
            '--pce-loss-weight', '1', '--zs-global-weight', '1', '--zs-spatial-loss-weight', '.01',
            '--zs-spatial-warmup-epochs', '28', '--validate-each-epoch', '--test-evaluation',
            '--numerical-debug', '--record-source-ids', '--cache-h5',
            '--annotation-id', 'organ_T13_half_train_seed42']
    parent = root.parent/'organ_T13_half_cl_20260908'
    shared = ['--epochs-per-task','60','--max-task','4','--organ-t2-supervision-strategy',
              '--organ-t2-feature-alpha','.05','--organ-retention-strategy',
              '--t3-first-epoch-evaluation','--der-buffer-size','128','--der-minibatch-size','4',
              '--der-alpha','.5','--der-beta','.5']
    definitions = [
        ('organ_retention', base + shared + ['--method','zs-derpp','--t3-from',str(root/'prefix_T2')], 2,60),
        ('organ_no_replay', base + shared + ['--method','zs-sequential','--t2-from',str(parent/'run60')],1,60),
        ('ind_T3', base + ['--method','zs-sequential','--organ-task','T3','--epochs-per-task','80',
                          '--grad-clip-norm','5'],0,80),
        ('ind_T4', base + ['--method','zs-sequential','--organ-task','T4','--epochs-per-task','80',
                          '--grad-clip-norm','5'],0,80),
    ]
    return [dict(name=name, command=cmd+['--output',str(root/'runs'/name)],
                 first_stage=first, epochs=epochs, stages=1 if name.startswith('ind_') else 4)
            for name,cmd,first,epochs in definitions]


def check_completed(root, job):
    output=root/'runs'/job['name']
    summary=json.loads((output/'summary.json').read_text())
    rows=[json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
    rows=[r for r in rows if 'loss' in r]
    assert summary['completed_stages']==job['stages']
    for stage in range(job['first_stage'],job['stages']):
        records=[r for r in rows if r['stage']==stage]
        assert [r['epoch'] for r in records]==list(range(job['epochs']))
        assert all(math.isfinite(r['loss']) for r in records)
        assert (output/f's{stage+1:02d}.pt').stat().st_size>0
    return dict(test_mean=summary['final_seen_mean'], validation_mean=summary['final_seen_validation_mean'])


def collect_metrics(root, outcomes, data_variant='T1/T3 half, T2/T4 full'):
    references=json.loads((root/'reference_T2.json').read_text())
    scores={'T2': references['test_foreground']}
    for task in ('T3','T4'):
        if outcomes['ind_'+task]['status']=='complete':
            summary=json.loads((root/'runs'/('ind_'+task)/'summary.json').read_text())
            scores[task]=summary['matrix'][0][0]
    write_json(root/'independent_references.json',dict(scores=scores,epochs=80,
        T2_source=references,complete=len(scores)==3,
        note='fixed independent reference; CL uses 60 epochs/task; T3/T4 independent use clip=5'))
    metrics={}
    for name in ('organ_retention','organ_no_replay'):
        if outcomes[name]['status']!='complete':continue
        s=json.loads((root/'runs'/name/'summary.json').read_text());m=s['matrix']
        params=[v['model_parameters'] for v in s['stage_rows']]
        metrics[name]={'A-Dice':sum(m[3])/4,
            'BWTR':sum((m[3][i]-m[i][i])/m[i][i] for i in range(3))/3
                if all(m[i][i]>0 for i in range(3)) else None,
            'RMA':sum(m[i][i]/scores['T'+str(i+1)] for i in range(1,4))/3
                if len(scores)==3 and all(v>0 for v in scores.values()) else None,
            'MPE':sum(params[i]-params[i-1] for i in range(1,4))/(3*params[0]),
            'DRR':0.0 if name=='organ_no_replay' else None,
            'DRR_note':'no raw/feature replay' if name=='organ_no_replay'
                else 'old T1/T2 buffers omit source IDs; new source counts in stage rows; do not invent historical counts',
            'matrix':m,'data_variant':data_variant}
    write_json(root/'benchmark_metrics.json',metrics)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();root=a.root.resolve();jobs=commands(root)
    if a.dry_run:
        print(json.dumps(jobs,indent=2));return
    if not (root/'preflight_passed.json').is_file():
        raise RuntimeError('preflight has not passed')
    if not root.is_relative_to(Path('/data_nas')):
        raise RuntimeError('campaign outputs must resolve inside /data_nas')
    for directory in ('inputs/data/Task_incre','inputs/sparse/organ','prefix_T2'):
        for path in (root/directory).iterdir():
            if not path.resolve().is_relative_to(Path('/data_nas')):
                raise RuntimeError(f'input/checkpoint resolves outside NAS: {path}')
    # Keep runtime caches and temporary files on the same NAS as checkpoints.
    for key, relative in {'TMPDIR':'tmp', 'XDG_CACHE_HOME':'cache',
                          'TORCH_HOME':'cache/torch', 'CUDA_CACHE_PATH':'cache/cuda',
                          'TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor',
                          'MPLCONFIGDIR':'cache/matplotlib'}.items():
        directory=root/relative
        directory.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(directory)
    with (root/'coordinator.started').open('x') as f:f.write(str(os.getpid()))
    lock=threading.Lock();outcomes={j['name']:dict(status='queued') for j in jobs}
    def status(name,**fields):
        with lock:
            outcomes[name]=fields
            write_json(root/'progress.json',dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),jobs=outcomes))
    def execute(job,gpu):
        name=job['name'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
            MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
        try:
            with (root/'logs'/(name+'.log')).open('x') as f:
                process=subprocess.Popen(job['command'],cwd=root/'source',env=env,stdout=f,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                status(name,status='running',gpu=gpu,pid=process.pid)
                code=process.wait()
            (root/'logs'/(name+'.exitcode')).write_text(str(code)+'\n')
            if code:raise RuntimeError(f'exit code {code}')
            status(name,status='complete',gpu=gpu,**check_completed(root,job))
        except Exception as error:
            status(name,status='failed',gpu=gpu,error=str(error))
        return outcomes[name]
    write_json(root/'launch_plan.json',dict(jobs=jobs,gpus=[4,5,6,7],min_free_mib=22000,
        data_variant='T1/T3 half, T2/T4 full',class_jobs=0,automatic_retries=False))
    run_gpu_queue(jobs,execute,min_free_memory_mib=22000)
    collect_metrics(root,outcomes)
    success=all(v['status']=='complete' for v in outcomes.values())
    (root/'pipeline.exitcode').write_text('0\n' if success else '1\n')
    if not success:raise SystemExit(1)


if __name__=='__main__':main()
