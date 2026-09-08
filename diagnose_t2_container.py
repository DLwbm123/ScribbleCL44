"""Eight-update forensic probe of the real shared loop; no validation/test access."""
import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.reference))
import runner_core as r
import torch

torch.set_num_threads(4)
r.TASKS['domain'] = (r.Task('T2', 'Task_incre', 'UCL.h5', (1,)),)
annotation = Path('/home/jiangsuiyang/medical_continual_segmentation_domain_gptpro/data/sparse_annotations_pattern_f5_b10/domain/D_v2_s2_seed42.npz')
r._sparse_path = lambda *a: annotation
args.output.mkdir(parents=True, exist_ok=False)

def canon(name):
    return name.replace('heads.0.', 'head.')

def snapshot(model, gradients=False):
    source = {k: p.grad for k, p in model.named_parameters() if p.grad is not None} if gradients else model.state_dict()
    return {canon(k): v.detach().cpu().clone() for k, v in source.items()}

def difference(a, b):
    assert a.keys() == b.keys()
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)

class Finished(Exception):
    pass

def run(name, kind, baseline=None):
    captured = {'inputs': [], 'gradients': [], 'states': [], 'masks': [], 'losses': []}
    report = {'name': name, 'steps': []}
    model = None
    index = 0
    mask_index = 0
    latest_loss = None
    real_loader, real_sgd = r._loader, torch.optim.SGD
    real_mix, real_backward = r.mixup_process, torch.Tensor.backward

    def build(scenario):
        nonlocal model
        model = r.DomainModel() if kind == 'domain' else r.OrganModel()
        initial = snapshot(model)
        if baseline is None:
            captured['initial'] = initial
        else:
            report['initial_max_abs_diff'] = difference(initial, baseline['initial'])
        return model

    def loader(dataset, *a):
        original = real_loader(dataset, *a)
        if dataset.split != 'train':
            return original
        class Proxy:
            def __len__(self):
                return len(original)
            def __iter__(self):
                for batch in original:
                    if baseline is None:
                        captured['inputs'].append(tuple(v.clone() for v in batch))
                    else:
                        report['steps'].append({'step': index + 1, 'batch_identical': all(torch.equal(v, b) for v, b in zip(batch, baseline['inputs'][index]))})
                    yield batch
        return Proxy()

    def mix(*a, **kw):
        nonlocal mask_index
        result = real_mix(*a, **kw)
        mask = result[3].detach().cpu()
        if baseline is None:
            captured['masks'].append(mask.clone())
        else:
            report['steps'][-1]['mix_mask_identical'] = torch.equal(mask, baseline['masks'][mask_index])
        mask_index += 1
        return result

    def backward(tensor, *a, **kw):
        nonlocal latest_loss
        latest_loss = float(tensor.detach())
        return real_backward(tensor, *a, **kw)

    class SGD(real_sgd):
        def step(self, *a, **kw):
            nonlocal index
            gradients = snapshot(model, True)
            result = super().step(*a, **kw)
            state = snapshot(model)
            if baseline is None:
                captured['gradients'].append(gradients)
                captured['states'].append(state)
                captured['losses'].append(latest_loss)
            else:
                report['steps'][-1].update(loss=latest_loss, loss_abs_diff=abs(latest_loss-baseline['losses'][index]),
                    gradient_max_abs_diff=difference(gradients, baseline['gradients'][index]),
                    state_max_abs_diff=difference(state, baseline['states'][index]))
            index += 1
            if index == 8:
                raise Finished
            return result

    argv = ['probe', '--data-root', '/home/jiangsuiyang/medical_continual_segmentation_data/CL_Benchmark/data',
        '--sparse-root', str(annotation.parent), '--output', str(args.output/name), '--device', 'cuda:0',
        '--seed', '42', '--independent-reference', '--independent-task', '1', '--method', 'zs-sequential',
        '--epochs-per-task', '80', '--batch-size', '4', '--lr', '.03', '--workers', '8',
        '--validate-every', '42', '--pce-loss-weight', '1', '--zs-global-weight', '1',
        '--zs-spatial-loss-weight', '.01', '--zs-spatial-warmup-epochs', '28', '--independent-skip-test']
    with ExitStack() as stack:
        for obj, key, value in [(r, '_build_model', build), (r, '_loader', loader),
                (r, 'mixup_process', mix), (torch.optim, 'SGD', SGD),
                (torch.Tensor, 'backward', backward), (sys, 'argv', argv)]:
            stack.enter_context(patch.object(obj, key, value))
        try:
            r.main('domain')
        except Finished:
            pass
        else:
            raise AssertionError('probe failed to stop after eight updates')
    assert index == 8
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return captured if baseline is None else report

results = {'scope': 'six eight-update runs on one GPU; real training loader; no validation or test', 'conditions': []}
for deterministic in [False, True]:
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    prefix = 'deterministic' if deterministic else 'production_defaults'
    baseline = run(prefix+'_domain_base', 'domain')
    condition = {'cudnn_deterministic': deterministic, 'deterministic_warn_only': deterministic,
                 'domain_repeat': run(prefix+'_domain_repeat', 'domain', baseline),
                 'organ': run(prefix+'_organ', 'organ', baseline)}
    results['conditions'].append(condition)
    (args.output/'trace.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(condition), flush=True)
    del baseline
    gc.collect()
