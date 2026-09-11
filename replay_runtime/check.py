"""Real-slice, bounded cross-task checks for both replay losses and metrics."""
import json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import h5py, numpy as np
from run import ROOT, arguments, completed, write
from runner_core import TASKS
def run_check():
    for scenario in ('organ','domain'):
        for task in TASKS[scenario]:
            root=ROOT/'checks/inputs'/scenario
            target=root/'data'/task.folder/task.filename
            target.parent.mkdir(parents=True,exist_ok=True)
            source=ROOT/'inputs'/scenario/'data'/task.folder/task.filename
            annotations=ROOT/'inputs'/scenario/'sparse'
            if scenario=='organ':annotations=annotations/scenario
            sparse=np.load(annotations/f'{task.code}_v2_s2_seed42.npz')['annotations']
            fg=np.flatnonzero((sparse>0).any(axis=(1,2)))
            selected=np.unique(np.r_[fg[:4],np.arange(min(8,len(sparse)))])[:8]
            with h5py.File(source,'r') as src,h5py.File(target,'w') as dst:
                for split in ('train','val','test'):
                    indices=selected if split=='train' else np.arange(min(4,src[f'{split}_images'].shape[2]))
                    for suffix in ('images','labels'):
                        dst[f'{split}_{suffix}']=src[f'{split}_{suffix}'][:,:,indices]
                    if split!='train':dst[f'patient_info_{split}']=np.array([len(indices)-1])
            out=root/'sparse'
            if scenario=='organ':out=out/scenario
            out.mkdir(parents=True,exist_ok=True)
            np.savez(out/f'{task.code}_v2_s2_seed42.npz',annotations=sparse[selected])
    def one(method,gpu):
        rows=[]
        for scenario in ('organ','domain'):
            name=scenario+'_'+method;job=dict(scenario=scenario,arguments=arguments(scenario,method,True))
            with (ROOT/'checks'/f'{name}.log').open('x') as log:
                p=subprocess.run(['./main','-u','run.py'],cwd=ROOT/'source',
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),RUN_JOB=json.dumps(job)),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            assert p.returncode==0,(name,p.returncode)
            output=ROOT/'checks'/name
            completed(output,scenario,1,4 if scenario=='organ' else 2)
            summary=json.loads((output/'summary.json').read_text())
            logs=[json.loads(x) for x in (output/'train.jsonl').read_text().splitlines()]
            epochs=[r for r in logs if 'loss' in r]
            key='derpp_pce_loss' if method=='zs-er' else 'derpp_feature_loss'
            assert any(r[key]>0 for r in epochs)
            zero_keys=('derpp_feature_loss','derpp_global_loss') if method=='zs-er' else ('derpp_pce_loss','derpp_global_loss')
            assert all(r[k]==0 for r in epochs for k in zero_keys)
            assert summary['derpp_buffer']['unique_replay_sources']
            memory=json.loads((output/'memory.json').read_text())
            rows.append(dict(name=name,status='passed',**memory))
        return rows
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,m,g) for m,g in [('zs-er',3),('zs-der',5)]]
        rows=[r for f in futures for r in f.result()]
    peak=max(r['peak_reserved_mib'] for r in rows)
    write(ROOT/'checks/passed.json',dict(results=rows,minimum_free_mib=max(8500,int(peak+1500))))
    print(json.dumps(rows),flush=True)
if __name__=='__main__':run_check()
