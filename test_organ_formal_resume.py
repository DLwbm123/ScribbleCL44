"""CPU check: formal retention policy, old-head isolation and replay accounting."""
import copy
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import h5py
import numpy as np
import torch
from runner_core import H5Slices, OrganModel, organ_task_strategy, evaluate, _loader
from cl_methods import DarkExperienceReplayPlus


def main():
    torch.set_num_threads(2)
    t2 = organ_task_strategy(True, 'T2', .5, .5, None, .05, True)
    assert t2 == dict(der_alpha=.05, der_beta=.5, grad_clip_norm=5., calibrate_head_bn=True)
    model = OrganModel()
    for stage in (1, 2, 3):
        model.activate_stage(stage)
        policy = organ_task_strategy(True, 'T'+str(stage+1), .5, .5, None, .05, True)
        model.freeze_backbone_bn = policy.get('freeze_backbone_bn', False)
        model.train()
        before = copy.deepcopy(model.state_dict())
        model.forward_logits(torch.randn(2, 1, 32, 32), stage).square().mean().backward()
        for key, value in before.items():
            old_head = key.startswith(tuple('heads.'+str(i)+'.' for i in range(stage)))
            fixed_stat = stage >= 2 and key.startswith('backbone.') and key.endswith(
                ('running_mean', 'running_var', 'num_batches_tracked'))
            if old_head or fixed_stat:
                assert torch.equal(value, model.state_dict()[key]), key
        if stage >= 2:
            assert policy['backbone_lr_scale'] == .1 and policy['grad_clip_norm'] == 5
        model.zero_grad(set_to_none=True)
    # Repeated retention observations must not alter model state or training RNG streams.
    model.activate_stage(2)
    model.train()
    dataset = torch.utils.data.TensorDataset(torch.randn(2,1,32,32), torch.zeros(2,32,32,dtype=torch.long))
    modes = [(module, module.training) for module in model.modules()]
    before = copy.deepcopy(model.state_dict())
    torch_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    score = evaluate(model, _loader(dataset, 1, False, 0, 42), np.array([1]), torch.device('cpu'), 2, (1,))
    assert 0 <= score['benchmark_mean'] <= 1
    assert all(torch.equal(value, model.state_dict()[key]) for key,value in before.items())
    assert all(module.training == mode for module,mode in modes)
    assert torch.equal(torch_rng, torch.get_rng_state()) and random.getstate() == python_rng
    current_rng = np.random.get_state()
    assert numpy_rng[0] == current_rng[0] and np.array_equal(numpy_rng[1],current_rng[1]) and numpy_rng[2:] == current_rng[2:]
    replay = DarkExperienceReplayPlus(3, 3, .5, .5)
    x = torch.ones(3, 1, 2, 2)
    replay.add_data(x, x, torch.zeros(3,2,2,dtype=torch.long), torch.zeros(3,dtype=torch.long),
                    2, source_ids=torch.tensor([2,2,5]))
    replay.consumer_stage = 1
    np.random.seed(42)
    replay.sample(torch.device('cpu'))
    assert replay.summary()['unique_replay_sources'] == {'0': 2}
    saved = replay.state_dict()
    restored = DarkExperienceReplayPlus(3, 3, .5, .5)
    restored.load_state_dict(saved)
    assert restored.source_ids == [2,2,5]
    legacy = dict(saved); legacy.pop('source_ids')
    restored.load_state_dict(legacy); restored.consumer_stage = 1
    restored.sample(torch.device('cpu'))
    assert restored.summary()['unknown_source_tasks'] == [0]
    with TemporaryDirectory() as directory:
        path=Path(directory)/'data.h5'; sparse=Path(directory)/'labels.npz'
        images=np.arange(256*256*2,dtype=np.float64).reshape(256,256,2)/11
        labels=(images%3>1).astype(np.uint8)
        with h5py.File(path,'w') as f:
            for split in ('train','val'):
                f[split+'_images']=images;f[split+'_labels']=labels
                f['patient_info_'+split]=[1]
        np.savez(sparse,annotations=labels.transpose(2,0,1).astype(np.int16))
        for split in ('train','val'):
            H5Slices.cache_arrays=False
            direct=H5Slices(path,split,sparse if split=='train' else None,return_index=True)
            H5Slices.cache_arrays=True
            cached=H5Slices(path,split,sparse if split=='train' else None,return_index=True)
            for index in (0,1):
                a,b=direct[index],cached[index]
                assert all(torch.equal(x,y) for x,y in zip(a[:2],b[:2])) and a[2]==b[2]
            direct.close();cached.close()
        H5Slices.cache_arrays=False
    print('PASS: retention policy, evaluation state/RNG isolation, source-ID dedup, legacy checkpoints and cached H5 equality')


if __name__ == '__main__':
    main()
