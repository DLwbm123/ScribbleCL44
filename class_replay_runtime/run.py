"""Fixed Class ER/feature-DER jobs; all arguments travel through environment."""
import json, math, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(path)
def args_for(method,check=False):
    cfg=json.loads((ROOT/'paths.json').read_text())
    data=str(ROOT/'checks/input_v2/data') if check else cfg['data']
    sparse=str(ROOT/'checks/input_v2/sparse') if check else cfg['sparse']
    args=['--data-root',data,'--sparse-root',sparse,
          '--output',str(ROOT/('checks' if check else 'jobs')/method),
          '--device','cuda:0','--method',method,'--seed','42',
          '--epochs-per-task','1' if check else '80','--batch-size','4',
          '--workers','0' if check else '4','--lr','.03','--pce-loss-weight','1',
          '--zs-global-weight','.1','--zs-spatial-loss-weight','0','--cache-h5',
          '--zs-spatial-warmup-epochs','33','--validate-every','2' if check else '375',
          '--der-buffer-size','64','--der-minibatch-size','4',
          '--der-alpha','0' if method=='zs-er' else '.5',
          '--der-beta','1' if method=='zs-er' else '0']
    if check:args+=['--max-train-batches','2','--validation-only']
    return args
def complete(output,epochs):
    s=json.loads((output/'summary.json').read_text())
    rows=[json.loads(x) for x in (output/'train.jsonl').read_text().splitlines()]
    rows=[r for r in rows if 'loss' in r]
    assert s['completed_stages']==3
    for stage in range(3):
        es=[r for r in rows if r['stage']==stage]
        assert [r['epoch'] for r in es]==list(range(epochs))
        assert all(math.isfinite(r['loss']) for r in es)
        assert (output/f's{stage+1:02d}_state.pt').stat().st_size>0
    for row in rows:
        assert row['derpp_global_loss']==0
        assert row['derpp_feature_loss']==0 if s['method']=='zs-er' else row['derpp_pce_loss']==0
    key='derpp_pce_loss' if s['method']=='zs-er' else 'derpp_feature_loss'
    assert any(row[key]>0 for row in rows)
    assert s['derpp_buffer']['replayed_unique_source_counts']
    return dict(foreground_mean=s['final_seen_mean'],epochs_per_task=epochs)
def child(method,check,log):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',JOB=json.dumps(args_for(method,check)))
    return subprocess.Popen([sys.executable,'-u','run.py'],cwd=ROOT/'source',env=env,
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
def main():
    from setproctitle import setproctitle
    setproctitle('job' if os.environ.get('JOB') else 'run')
    if os.environ.get('JOB'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch,runner_core
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        sys.argv=['main',*json.loads(os.environ['JOB'])]
        runner_core.main('class')
        output=Path(sys.argv[sys.argv.index('--output')+1])
        write(output/'memory.json',dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        return
    if os.environ.get('CHECK_ONLY'):
        from check_replay import run_checks
        run_checks()
        return
    waiting_pid=os.environ.get('WAIT_CHECK_PID')
    if waiting_pid:
        while not (ROOT/'checks/passed.json').is_file():
            process=Path('/proc')/waiting_pid
            if not process.exists() or "State:\tZ" in (process/'status').read_text():
                write(ROOT/'progress.json',dict(status='blocked',reason='cross-task checks did not pass'))
                raise RuntimeError('cross-task check process exited without passing')
            time.sleep(10)
    check=json.loads((ROOT/'checks/passed.json').read_text())
    with (ROOT/'coordinator.started').open('x') as f:f.write(str(os.getpid()))
    plan=dict(methods=['zs-er','zs-der'],gpu=2,epochs_per_task=80,batch_size=4,
        replay_batch=4,buffer=64,global_weight=.1,spatial_weight=0,
        selection='current-task foreground validation, all 80 epochs',
        reporting='background-inclusive Dice and WCD; original-source DRR',
        initialization='fresh, because replay objectives differ from existing T1',
        minimum_free_mib=check['minimum_free_mib'],automatic_extra_runs=False)
    write(ROOT/'plan.json',plan)
    outcomes={m:dict(status='queued') for m in plan['methods']}
    launch_lock,record_lock=threading.Lock(),threading.Lock()
    def status(method,**fields):
        with record_lock:
            outcomes[method]=fields
            write(ROOT/'progress.json',dict(updated_at=time.time(),jobs=outcomes))
    def run(method):
        try:
            with (ROOT/'logs'/f'{method}.log').open('x') as log:
                with launch_lock:
                    while int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.free',
                              '--format=csv,noheader,nounits'],text=True))<plan['minimum_free_mib']:
                        time.sleep(15)
                    p=child(method,False,log)
                    status(method,status='running',gpu=2,pid=p.pid,started_at=time.time())
                    while p.poll() is None and not (ROOT/'jobs'/method/'startup.json').exists():
                        time.sleep(1)
                code=p.wait()
            (ROOT/'logs'/f'{method}.exitcode').write_text(str(code)+'\n')
            if code:raise RuntimeError(f'training exited {code}')
            status(method,status='complete',gpu=2,**complete(ROOT/'jobs'/method,80))
        except Exception as error:status(method,status='failed',gpu=2,error=str(error))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,plan['methods']))
    ok=all(r['status']=='complete' for r in outcomes.values())
    write(ROOT/'comparison.json',dict(complete=ok,jobs=outcomes))
    (ROOT/'pipeline.exitcode').write_text('0\n' if ok else '1\n')
    if not ok:raise SystemExit(1)
if __name__=='__main__':main()
