"""Train-image-only BN calibration probe; no optimizer, no test-set access."""
import argparse
import copy
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root/'source'))
    import torch
    from runner_core import OrganModel, TASKS, H5Slices, _loader, _evaluate_task
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = OrganModel().cuda()
    model.activate_stage(1)
    model.load_state_dict(torch.load(args.case/'diagnostic_final_model.pt', map_location='cuda'))
    task = TASKS['organ'][1]
    data = H5Slices(args.root/'subset/data'/task.folder/task.filename, 'train',
                    args.root/'subset/sparse/organ/T2_v2_s2_seed42.npz', augment=False)
    loader = _loader(data, 4, False, 0, 42)
    results = {}
    for scope in ('none', 'head', 'backbone', 'both'):
        probe = copy.deepcopy(model).eval()
        selected = []
        if scope in ('head', 'both'):
            selected.extend(probe.heads['1'].modules())
        if scope in ('backbone', 'both'):
            selected.extend(probe.backbone.modules())
        layers = [layer for layer in selected if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
        if layers:
            with torch.no_grad():
                for image, _ in loader:
                    probe.forward_logits(image.cuda(), 1)
        scores = {t.code: _evaluate_task(probe, 'organ', t, i,
                    args.root/'subset/data', 'val', 4, torch.device('cuda'))
                  for i, t in enumerate(TASKS['organ'][:2])}
        results[scope] = {
            'validation': {t: row['benchmark_mean'] for t, row in scores.items()},
            'prediction_fg_fraction': {t: row['prediction_fg_fraction'] for t, row in scores.items()},
            'calibrated_bn_layers': len(layers),
        }
        del probe
    assert abs(results['head']['validation']['T1'] - results['none']['validation']['T1']) < 1e-6
    data.close()
    result = {'calibration_data': 'T2 training images only; no augmentation; no labels used',
              'optimizer_steps': 0, 'parameter_updates': 0, 'results': results,
              'scope': 'diagnosis only; no calibrated model saved or formal training resumed'}
    (args.case/'bn_calibration_probe.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
