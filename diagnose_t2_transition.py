"""Bounded T2 diagnosis from a saved T1 paired state; never resumes a formal run.

Uses a declared transition seed because the checkpoint omits the original RNG.
Executes the existing runner with bounded entry/exit and existing control flags.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=42)
    parser.add_argument('--lr', type=float, default=.03)
    parser.add_argument('--clip', type=float)
    parser.add_argument('--clean-bn-writer', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--transition-seed', type=int, default=43)
    parser.add_argument('--freeze-backbone-bn', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.steps <= 420
    assert 0 < args.lr <= .03
    assert args.clip is None or 0 < args.clip < float('inf')
    assert 0 <= args.transition_seed < 2**32
    assert not (args.clean_bn_writer and args.freeze_backbone_bn)
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(exist_ok=False)
    probe = output/'write_probe'
    probe.write_text('ok')
    assert probe.read_text() == 'ok'
    probe.unlink()
    source = root/'source'
    sys.path.insert(0, str(source))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    import numerical_safety
    record_success = numerical_safety.NumericalAudit.record_success
    capture_pre_step = numerical_safety.NumericalAudit.capture_pre_step
    live_model = None
    initial_bn = {}

    def remember_model(audit, model, optimizer, **kwargs):
        nonlocal live_model
        if live_model is None and args.freeze_backbone_bn:
            initial_bn.update({name: value.detach().cpu().clone()
                               for name, value in model.backbone.named_buffers()})
        live_model = model
        return capture_pre_step(audit, model, optimizer, **kwargs)

    numerical_safety.NumericalAudit.capture_pre_step = remember_model

    def bounded_record(audit):
        record_success(audit)
        if audit.iteration >= args.steps:
            result = {**audit.metadata(), 'steps_completed': audit.iteration}
            if args.freeze_backbone_bn:
                result['backbone_buffers_unchanged'] = all(
                    torch.equal(initial_bn[name], value.detach().cpu())
                    for name, value in live_model.backbone.named_buffers())
                assert result['backbone_buffers_unchanged']
            if args.evaluate:
                scores = {
                    task.code: module._evaluate_task(live_model, 'organ', task, index,
                        root/'subset/data', 'val', 4, torch.device('cuda:0'))
                    for index, task in enumerate(module.TASKS['organ'][:2])}
                result['validation'] = {task: row['benchmark_mean'] for task, row in scores.items()}
                result['validation_prediction_fg_fraction'] = {
                    task: row['prediction_fg_fraction'] for task, row in scores.items()}
            (output/'BOUNDED_FINITE.json').write_text(json.dumps(result)+'\n')
            raise SystemExit(0)

    numerical_safety.NumericalAudit.record_success = bounded_record
    text = (source/'runner_core.py').read_text()
    entry = '    for stage, task in enumerate(tasks[:last_stage + 1]):\n'
    assert text.count(entry) == 1
    replacement = entry + '''        if stage == 0:
            saved = torch.load(DIAGNOSTIC_ROOT / "run60" / "s01_state.pt", map_location="cpu")
            assert saved["stage"] == 0 and saved["method"] == args.method
            model.load_state_dict(saved["model"], strict=True)
            derpp.load_state_dict(saved["continual"])
            del saved
            torch.manual_seed(DIAGNOSTIC_TRANSITION_SEED)
            np.random.seed(DIAGNOSTIC_TRANSITION_SEED)
            random.seed(DIAGNOSTIC_TRANSITION_SEED)
            continue
'''
    text = text.replace(entry, replacement)
    settings = ['--data-root', str(root/'subset/data'),
                '--sparse-root', str(root/'subset/sparse'), '--output', str(output/'run'),
                '--device', 'cuda:0', '--seed', '42', '--epochs-per-task', '60',
                '--max-task', '2', '--batch-size', '4', '--workers', '8', '--lr', str(args.lr),
                '--method', 'zs-derpp', '--der-alpha', '.5', '--der-beta', '.5',
                '--der-buffer-size', '128', '--der-minibatch-size', '4',
                '--pce-loss-weight', '1', '--zs-global-weight', '1',
                '--zs-spatial-loss-weight', '.01', '--zs-spatial-warmup-epochs', '28',
                '--validate-every', '999999', '--numerical-debug',
                '--annotation-id', f'diagnostic_only_from_T1_transition{args.transition_seed}']
    if args.clip is not None:
        settings += ['--grad-clip-norm', str(args.clip)]
    if args.clean_bn_writer:
        settings += ['--zs-clean-bn-writer']
    sys.argv = ['diagnose_t2_transition.py'] + settings
    metadata = {'source_checkpoint': 'run60/s01_state.pt', 'transition_seed': args.transition_seed,
                'exact_original_rng_replay': False, 'maximum_steps': args.steps,
                'lr': args.lr, 'clip': args.clip, 'clean_bn_writer': args.clean_bn_writer,
                'freeze_backbone_bn': args.freeze_backbone_bn,
                'spatial_active': False, 'formal_training': False,
                'evaluation': 'validation only' if args.evaluate else 'none'}
    (output/'diagnostic_protocol.json').write_text(json.dumps(metadata, indent=2)+'\n')
    namespace = {'__name__': 'diagnostic_runner', '__file__': str(source/'runner_core.py'),
                 'DIAGNOSTIC_ROOT': root, 'DIAGNOSTIC_TRANSITION_SEED': args.transition_seed}
    # Dataclasses resolve their defining module through sys.modules.
    import types
    module = types.ModuleType('diagnostic_runner')
    module.__dict__.update(namespace)
    sys.modules[module.__name__] = module
    exec(compile(text, str(source/'runner_core.py'), 'exec'), module.__dict__)
    if args.freeze_backbone_bn:
        original_train = module.OrganModel.train

        def frozen_backbone_train(model, mode=True):
            original_train(model, mode)
            for layer in model.backbone.modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    layer.eval()
            return model

        module.OrganModel.train = frozen_backbone_train
    module.main('organ')


if __name__ == '__main__':
    main()
