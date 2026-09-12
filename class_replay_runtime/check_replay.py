"""Small synthetic-input check of replay across all three Class tasks."""
import json
import h5py
import numpy as np
from run import ROOT,args_for,child,complete,write
from runner_core import TASKS,H5Slices
def run_checks():
    rng=np.random.default_rng(42)
    yy,xx=np.indices((256,256))
    for task in TASKS['class']:
        count=len(task.classes)
        label=np.zeros((256,256),dtype=np.int16)
        for c in range(1,count+1):
            label[30+c*20:50+c*20,40:200]=c
        images=np.stack([label.astype(np.float32)/count+rng.normal(0,.1,label.shape)
                         for _ in range(8)],axis=2).astype(np.float32)
        labels=np.repeat(label[:,:,None],8,axis=2)
        dst=ROOT/'checks/input_v2/data'/task.folder/task.filename
        dst.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(dst,'w') as out:
            for split,n in [('train',8),('val',4),('test',4)]:
                out[f'{split}_images']=images[:,:,:n]
                out[f'{split}_labels']=labels[:,:,:n]
                if split!='train':out[f'patient_info_{split}']=np.array([n-1])
        sparse=np.full((8,256,256),-100,dtype=np.int16)
        known=(xx%16==0)&(yy%8==0)
        global_labels=np.where(label>0,label+task.label_shift,0)
        sparse[:,known]=global_labels[known]
        target=ROOT/'checks/input_v2/sparse/class'
        target.mkdir(parents=True,exist_ok=True)
        np.savez(target/f'{task.code}_v2_s2_seed42.npz',annotations=sparse)
    # Check cache equivalence on every task's global sparse and shifted validation labels.
    import torch
    for task in TASKS['class']:
        data=ROOT/'checks/input_v2/data'/task.folder/task.filename
        sparse=ROOT/'checks/input_v2/sparse/class'/f'{task.code}_v2_s2_seed42.npz'
        for split in ('train','val'):
            H5Slices.cache_arrays=False
            plain=H5Slices(data,split,sparse if split=='train' else None,
                           label_shift=0 if split=='train' else task.label_shift)
            H5Slices.cache_arrays=True
            cached=H5Slices(data,split,sparse if split=='train' else None,
                           label_shift=0 if split=='train' else task.label_shift)
            for i in range(len(plain)):
                assert all(torch.equal(a,b) for a,b in zip(plain[i],cached[i]))
            plain.close();cached.close()
    H5Slices.cache_arrays=False
    results=[]
    for method in ('zs-er','zs-der'):
        with (ROOT/'checks'/f'{method}.log').open('x') as log:
            p=child(method,True,log);assert p.wait()==0,method
        complete(ROOT/'checks'/method,1)
        memory=json.loads((ROOT/'checks'/method/'memory.json').read_text())
        results.append(dict(method=method,status='passed',**memory))
    peak=max(r['peak_reserved_mib'] for r in results)
    write(ROOT/'checks/passed.json',dict(results=results,minimum_free_mib=max(16000,int(peak+2500))))
    print(json.dumps(results),flush=True)
