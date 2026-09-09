"""CPU check: task isolation, actual Organ head buffers, and strict paired replay restore."""
import copy
import torch
from runner_core import OrganModel, organ_task_strategy, calibrate_organ_head_bn
from cl_methods import DarkExperienceReplayPlus


def main():
    torch.set_num_threads(2)
    base = organ_task_strategy(False, 'T2', .5, .5, None)
    for task in ('T1', 'T3'):
        assert organ_task_strategy(True, task, .5, .5, None) == base
    for task in ('T1', 'T3'):
        assert organ_task_strategy(True, task, .5, .5, None, .05) == base
    assert organ_task_strategy(True, 'T2', .5, .5, None, .05)['der_alpha'] == .05
    t2 = organ_task_strategy(True, 'T2', .5, .5, None)
    assert t2 == dict(der_alpha=0., der_beta=.5, grad_clip_norm=5., calibrate_head_bn=True)
    model = OrganModel()
    model.activate_stage(1)
    model.train()
    modes = [m.training for m in model.modules()]
    before = copy.deepcopy(model.state_dict())
    momentum = [m.momentum for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    image = torch.randn(2, 1, 32, 32)
    result = calibrate_organ_head_bn(model, [(image, None), (image + 2, None)], 'cpu', 1)
    assert result['training_batches'] == 2 and result['bn_layers'] == 1
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert changed and all(k.startswith('heads.1.') and
        k.endswith(('running_mean', 'running_var', 'num_batches_tracked')) for k in changed), changed
    assert modes == [m.training for m in model.modules()]
    assert momentum == [m.momentum for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    try:
        calibrate_organ_head_bn(model, [], 'cpu', 1)
        raise AssertionError('empty loader accepted')
    except ValueError:
        pass
    assert modes == [m.training for m in model.modules()]
    replay = DarkExperienceReplayPlus(buffer_size=4, minibatch_size=2, alpha=0., beta=.5)
    state = replay.state_dict()
    replay.load_state_dict(state)
    replay.alpha, replay.beta = base['der_alpha'], base['der_beta']
    try:
        replay.load_state_dict(state)
        raise AssertionError('coefficient mismatch accepted')
    except ValueError:
        pass
    print('PASS: T1/T3 base policy, T2 head-only buffer updates, modes/momentum restored, strict replay restore')


if __name__ == '__main__':
    main()

# Optional native-loop integration using synthetic images, never patient data.
# Usage: python test_organ_task_strategy.py --smoke-root /nas/path
if __name__ == '__main__' and len(__import__('sys').argv) > 1:
    import json
    import sys
    from pathlib import Path
    import h5py
    import numpy as np
    import runner_core as runner
    root = Path(sys.argv[2])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'data/Task_incre').mkdir(parents=True)
    (root / 'sparse/organ').mkdir(parents=True)
    rng = np.random.default_rng(42)
    for task in runner.TASKS['organ'][:3]:
        labels = np.zeros((256, 256, 2), dtype=np.int16)
        labels[96:160, 96:160] = 1
        with h5py.File(root / 'data/Task_incre' / task.filename, 'w') as f:
            for split in ('train', 'val'):
                f[f'{split}_images'] = rng.normal(size=labels.shape).astype('float32')
                f[f'{split}_labels'] = labels
            f['patient_info_val'] = np.array([1])
        sparse = np.full((2, 256, 256), runner.IGNORE_INDEX, dtype=np.int16)
        sparse[:, 110:114, 100:150] = 1
        sparse[:, 20:24, 20:100] = 0
        np.savez(root / 'sparse/organ' / f'{task.code}_v2_s2_seed42.npz', annotations=sparse)
    common = ['check', '--data-root', str(root / 'data'), '--sparse-root', str(root / 'sparse'),
              '--device', 'cuda:0', '--max-task', '3', '--epochs-per-task', '1', '--batch-size', '2',
              '--workers', '0', '--lr', '.001', '--method', 'zs-derpp', '--der-buffer-size', '4',
              '--der-minibatch-size', '2', '--organ-t2-supervision-strategy', '--validate-each-epoch']
    sys.argv = common + ['--output', str(root / 'run')]
    runner.main('organ')
    rows = json.loads((root / 'run/stages.json').read_text())
    assert [r['derpp']['alpha'] for r in rows] == [.5, 0., .5]
    assert [r['task_strategy']['calibrate_head_bn'] for r in rows] == [False, True, False]
    sys.argv = common + ['--output', str(root / 'resume'), '--t3-one-epoch-from', str(root / 'run')]
    runner.main('organ')
    resumed = json.loads((root / 'resume/stages.json').read_text())
    assert resumed[2]['derpp']['alpha'] == .5
    sys.argv = common + ['--output', str(root / 'resume_t2'), '--t2-from', str(root / 'run'),
                         '--organ-t2-feature-alpha', '.05', '--epochs-per-task', '2', '--organ-t2-epochs', '1']
    runner.main('organ')
    resumed_t2 = json.loads((root / 'resume_t2/stages.json').read_text())
    assert [r['derpp']['alpha'] for r in resumed_t2] == [.5, .05, .5]
    assert resumed_t2[0] == rows[0]
    epoch_rows = [json.loads(line) for line in (root / 'resume_t2/train.jsonl').read_text().splitlines()]
    epoch_rows = [r for r in epoch_rows if 'epoch_seconds' in r]
    assert sum(r['stage'] == 1 for r in epoch_rows) == 1
    assert sum(r['stage'] == 2 for r in epoch_rows) == 2
    manifest = json.loads((root / 'resume_t2/manifest.json').read_text())
    assert manifest['organ_t2_executed_epochs'] == 1 and manifest['epochs_per_task'] == 2
    print('PASS: native T1/T2/T3, paired selection, T3 resume, and T2 resume with small alpha; preserved T1 record')
