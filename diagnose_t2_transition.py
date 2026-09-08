"""Bounded T2 diagnosis from a saved T1 paired state; never resumes a formal run.

Uses a declared transition seed because the checkpoint omits the original RNG.
Executes the existing runner with bounded entry/exit and existing control flags.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def class_balanced_pce(logp, labels):
    """Equal mean contribution of each present class; preserve ignored pixels."""
    import torch
    from numerical_safety import sparse_pce_from_log_probs
    original = sparse_pce_from_log_probs(logp, labels)  # validates shape/range/finite inputs
    terms = [sparse_pce_from_log_probs(logp, labels.masked_fill(labels.ne(k), -100))
             for k in range(logp.shape[1]) if bool(labels.eq(k).any())]
    return torch.stack(terms).mean() if terms else original


def balance_probe(audit, model, current, feature, supervision, labels, replay):
    """Compare weighted gradient contributions on the shared backbone only."""
    import torch
    parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    vectors = []
    for loss in (current, feature, supervision):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
                                  for p, g in zip(parameters, gradients)]))
    norms = [float(torch.linalg.vector_norm(v, dtype=torch.float64)) for v in vectors]
    cosine = [None if not norms[0] or not norms[k] else
              float((vectors[0]/norms[0]).dot(vectors[k]/norms[k])) for k in (1, 2)]
    audit.note('balance', 'backbone_gradient_contributions',
               current_norm=norms[0], feature_norm=norms[1], replay_supervision_norm=norms[2],
               current_feature_cosine=cosine[0], current_replay_supervision_cosine=cosine[1],
               marked_foreground_pixels=int(labels.gt(0).sum()),
               marked_background_pixels=int(labels.eq(0).sum()),
               replay_T1_examples=0 if replay is None else int(replay[3].eq(0).sum()),
               replay_T2_examples=0 if replay is None else int(replay[3].eq(1).sum()))


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
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--balanced-current-pce', action='store_true')
    parser.add_argument('--probe-balance', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.steps <= 420
    assert 0 < args.lr <= .03
    assert args.clip is None or 0 < args.clip < float('inf')
    assert 0 <= args.transition_seed < 2**32
    assert not (args.clean_bn_writer and args.freeze_backbone_bn)
    assert 0 <= args.alpha <= .5 and 0 <= args.beta <= .5
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
            if args.save_model:
                torch.save(live_model.state_dict(), output/'diagnostic_final_model.pt')
            raise SystemExit(0)

    numerical_safety.NumericalAudit.record_success = bounded_record
    text = (source/'runner_core.py').read_text()
    entry = '    for stage, task in enumerate(tasks[:last_stage + 1]):\n'
    assert text.count(entry) == 1
    replacement = entry + '''        if stage == 0:
            saved = torch.load(DIAGNOSTIC_ROOT / "run60" / "s01_state.pt", map_location="cpu")
            assert saved["stage"] == 0 and saved["method"] == args.method
            model.load_state_dict(saved["model"], strict=True)
            # Restore the historical buffer under its original coefficient contract,
            # then apply the explicitly declared T2-only loss intervention.
            derpp.alpha, derpp.beta = saved["continual"]["alpha"], saved["continual"]["beta"]
            derpp.load_state_dict(saved["continual"])
            derpp.alpha, derpp.beta = args.der_alpha, args.der_beta
            del saved
            torch.manual_seed(DIAGNOSTIC_TRANSITION_SEED)
            np.random.seed(DIAGNOSTIC_TRANSITION_SEED)
            random.seed(DIAGNOSTIC_TRANSITION_SEED)
            continue
'''
    text = text.replace(entry, replacement)
    if args.balanced_current_pce:
        old = 'partial_ce = pce_loss(outputs, target)'
        assert text.count(old) == 1
        text = text.replace(old, 'partial_ce = DIAGNOSTIC_BALANCED_PCE(outputs["log_probabilities"], label)')
    if args.probe_balance:
        old = '                    loss.backward()\n                    gradient_norm = numerics.check_gradients(model)'
        assert text.count(old) == 1
        text = text.replace(old, '''                    if iteration + 1 in {1, 5, 42, 126, 420}:
                        DIAGNOSTIC_BALANCE_PROBE(numerics, model,
                            args.pce_loss_weight * partial_ce + args.zs_global_weight * global_loss,
                            derpp_feature, args.der_beta * (derpp_pce + args.zs_global_weight * derpp_global),
                            label, replay)
''' + old)
    settings = ['--data-root', str(root/'subset/data'),
                '--sparse-root', str(root/'subset/sparse'), '--output', str(output/'run'),
                '--device', 'cuda:0', '--seed', '42', '--epochs-per-task', '60',
                '--max-task', '2', '--batch-size', '4', '--workers', '8', '--lr', str(args.lr),
                '--method', 'zs-derpp', '--der-alpha', str(args.alpha), '--der-beta', str(args.beta),
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
                'alpha': args.alpha, 'beta': args.beta,
                'balanced_current_pce': args.balanced_current_pce,
                'probe_balance': args.probe_balance,
                'replay_forwards_and_buffer_updates_retained': True,
                'spatial_active': False, 'formal_training': False,
                'evaluation': 'validation only' if args.evaluate else 'none'}
    (output/'diagnostic_protocol.json').write_text(json.dumps(metadata, indent=2)+'\n')
    namespace = {'__name__': 'diagnostic_runner', '__file__': str(source/'runner_core.py'),
                 'DIAGNOSTIC_ROOT': root, 'DIAGNOSTIC_TRANSITION_SEED': args.transition_seed,
                 'DIAGNOSTIC_BALANCED_PCE': class_balanced_pce,
                 'DIAGNOSTIC_BALANCE_PROBE': balance_probe}
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
