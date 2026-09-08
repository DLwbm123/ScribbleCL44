"""Bounded T2 diagnosis from a saved T1 paired state; never resumes a formal run.

Uses transition seed 43 because the saved checkpoint omits the original RNG.
Executes the existing runner in memory, changing only entry/exit boundaries.
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
    args = parser.parse_args()
    assert 1 <= args.steps <= 42
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

    def bounded_record(audit):
        record_success(audit)
        if audit.iteration >= args.steps:
            (output/'BOUNDED_FINITE.json').write_text(json.dumps(audit.metadata())+'\n')
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
            torch.manual_seed(43)
            np.random.seed(43)
            random.seed(43)
            continue
'''
    text = text.replace(entry, replacement)
    settings = ['--data-root', str(root/'subset/data'),
                '--sparse-root', str(root/'subset/sparse'), '--output', str(output/'run'),
                '--device', 'cuda:0', '--seed', '42', '--epochs-per-task', '60',
                '--max-task', '2', '--batch-size', '4', '--workers', '8', '--lr', '.03',
                '--method', 'zs-derpp', '--der-alpha', '.5', '--der-beta', '.5',
                '--der-buffer-size', '128', '--der-minibatch-size', '4',
                '--pce-loss-weight', '1', '--zs-global-weight', '1',
                '--zs-spatial-loss-weight', '.01', '--zs-spatial-warmup-epochs', '28',
                '--validate-each-epoch', '--numerical-debug',
                '--annotation-id', 'diagnostic_only_from_T1_seed43']
    sys.argv = ['diagnose_t2_transition.py'] + settings
    metadata = {'source_checkpoint': 'run60/s01_state.pt', 'transition_seed': 43,
                'exact_original_rng_replay': False, 'maximum_steps': args.steps,
                'lr': .03, 'spatial_active': False, 'formal_training': False}
    (output/'diagnostic_protocol.json').write_text(json.dumps(metadata, indent=2)+'\n')
    namespace = {'__name__': 'diagnostic_runner', '__file__': str(source/'runner_core.py'),
                 'DIAGNOSTIC_ROOT': root}
    # Dataclasses resolve their defining module through sys.modules.
    import types
    module = types.ModuleType('diagnostic_runner')
    module.__dict__.update(namespace)
    sys.modules[module.__name__] = module
    exec(compile(text, str(source/'runner_core.py'), 'exec'), module.__dict__)
    module.main('organ')


if __name__ == '__main__':
    main()
