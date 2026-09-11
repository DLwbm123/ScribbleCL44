"""Four bounded replay controls. Invoke as ./main -u run.py; config stays in env."""
import json, math, os, subprocess, sys, threading, time
from pathlib import Path
from run_independent_domains_formal import run_gpu_queue
ROOT = Path(__file__).resolve().parent.parent
def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)
def arguments(scenario, method, check=False):
    inputs = ROOT/('checks/inputs' if check else 'inputs')/scenario
    output = ROOT/('checks' if check else 'runs')/(scenario+'_'+method)
    args = ['--data-root',str(inputs/'data'),'--sparse-root',str(inputs/'sparse'),
            '--output',str(output),'--method',method,'--device','cuda:0','--seed','42',
            '--batch-size','4','--workers','0' if check else '4',
            '--epochs-per-task','1' if check else ('40' if scenario=='organ' else '80'),
            '--lr','.03','--der-buffer-size','64','--der-minibatch-size','4',
            '--der-alpha','0' if method=='zs-er' else '.5',
            '--der-beta','1' if method=='zs-er' else '0',
            '--zs-global-weight','1','--zs-spatial-loss-weight','0',
            '--validate-each-epoch','--test-evaluation','--cache-h5',
            '--numerical-debug','--record-source-ids',
            '--annotation-id','organ_T134_half_train_seed42' if scenario=='organ' else 'domain_pattern_f5_b10']
    if scenario=='organ':
        args += ['--organ-tuning','--organ-t2-supervision-strategy','--organ-retention-strategy',
                 '--organ-backbone-lr-scales','.1','.1','.1','--organ-freeze-bn-from','2']
    if check:
        args += ['--max-train-batches','2','--max-task','4' if scenario=='organ' else '2']
    return args
def completed(output, scenario, epochs, stages):
    summary=json.loads((output/'summary.json').read_text())
    assert summary['completed_stages']==stages
    records=[json.loads(x) for x in (output/'train.jsonl').read_text().splitlines()]
    for stage in range(stages):
        rows=[r for r in records if r.get('stage')==stage and 'loss' in r]
        assert [r['epoch'] for r in rows]==list(range(epochs))
        assert all(math.isfinite(r['loss']) for r in rows)
        assert (output/f's{stage+1:02d}_state.pt').stat().st_size>0
        for metric in summary['stage_rows'][stage]['evaluated'].values():
            assert abs(metric['background_inclusive_dice']-
                       (metric['benchmark_mean']+metric['background_dice'])/2)<1e-12
    return dict(foreground_mean=summary['final_seen_mean'],
                background_inclusive_mean=summary['final_background_inclusive_mean'])
def main():
    if os.environ.get('RUN_JOB'):
        import torch, runner_core
        job=json.loads(os.environ['RUN_JOB'])
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        sys.argv=['main',*job['arguments']]
        runner_core.main(job['scenario'])
        output=Path(job['arguments'][job['arguments'].index('--output')+1])
        write(output/'memory.json',dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        return
    checks=json.loads((ROOT/'checks/passed.json').read_text())
    with (ROOT/'coordinator.started').open('x') as stream:stream.write(str(os.getpid()))
    jobs=[dict(name=scenario+'_'+method,scenario=scenario,arguments=arguments(scenario,method))
          for method in ('zs-der','zs-er') for scenario in ('organ','domain')]
    write(ROOT/'plan.json',dict(jobs=jobs,gpus=[3,4,5,6,7],minimum_free_mib=checks['minimum_free_mib'],
        selection='current-task foreground validation; test evaluated only after selection',
        epochs=dict(organ=[40]*4,domain=[80]*6),buffer=64,replay_batch=4,
        initialization='fresh; no compatible first-task replay checkpoint',additional_sweeps=0))
    outcomes={j['name']:dict(status='queued') for j in jobs}
    lock=threading.Lock()
    def status(name,**fields):
        with lock:
            outcomes[name]=fields
            write(ROOT/'progress.json',dict(updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'),jobs=outcomes))
    def execute(job,gpu):
        name=job['name']
        try:
            with (ROOT/'logs'/f'{name}.log').open('x') as stream:
                process=subprocess.Popen(['./main','-u','run.py'],cwd=ROOT/'source',
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),RUN_JOB=json.dumps(job)),
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                status(name,status='running',gpu=gpu,pid=process.pid,started_at=time.time())
                code=process.wait()
            (ROOT/'logs'/f'{name}.exitcode').write_text(str(code)+'\n')
            if code:raise RuntimeError(f'training exited {code}')
            status(name,status='complete',gpu=gpu,**completed(ROOT/'runs'/name,job['scenario'],
                40 if job['scenario']=='organ' else 80,4 if job['scenario']=='organ' else 6))
        except Exception as error:status(name,status='failed',gpu=gpu,error=str(error))
    run_gpu_queue(jobs,execute,min_free_memory_mib=checks['minimum_free_mib'],gpus=(3,4,5,6,7))
    ok=all(r['status']=='complete' for r in outcomes.values())
    write(ROOT/'comparison.json',dict(complete=ok,jobs=outcomes))
    (ROOT/'pipeline.exitcode').write_text('0\n' if ok else '1\n')
    if not ok:raise SystemExit(1)
if __name__=='__main__':main()
