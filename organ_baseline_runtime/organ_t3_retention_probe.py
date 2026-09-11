"""Bounded observers/controls for the shared runner; no alternate training loop.

All patient outputs and endpoint tensors are private server artifacts.
"""
from contextlib import contextmanager
from collections import Counter
import copy
import json
import random
import sys
import time
import numpy as np
import torch

NODES = (0, 1, 5, 10, 25, 50, 100, 178)
GRAD_NODES = (1, 10, 50, 178)
BASELINE = {'T1': .6475301369585638, 'T2': .6991301629991079}
TOLERANCE = 1e-6


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def validate_controls(args, scenario):
    required = dict(method='zs-derpp', batch_size=4, epochs_per_task=60, max_task=3,
        seed=42, der_buffer_size=128, der_minibatch_size=4, organ_t2_feature_alpha=.05,
        pce_loss_weight=1., zs_global_weight=1., zs_spatial_loss_weight=.01,
        zs_spatial_warmup_epochs=28, lr=.03)
    if scenario != 'organ' or any(getattr(args, k) != v for k, v in required.items()):
        raise ValueError('T3 probe requires the fixed, authorized Organ protocol')
    if (not args.t3_one_epoch_from or not args.organ_t2_supervision_strategy or
        not args.numerical_debug or args.test_evaluation or args.max_train_batches is not None or
        args.organ_task or args.t2_from or args.zs_clean_bn_writer or args.zs_gd_loss or
        args.zs_adversarial_perturbation):
        raise ValueError('T3 probe forbids test access, changed horizon, or unapproved controls')


def check_paired_restore(model, replay, saved):
    """One targeted exact restoration check; no hashes or target recomputation."""
    for key, value in model.state_dict().items():
        if not torch.equal(value.cpu(), saved['model'][key]):
            raise ValueError(f'model restore mismatch: {key}')
    state = saved['continual']
    for field in ('examples', 'feature_targets', 'sparse_labels'):
        if not all(torch.equal(x, y) for x, y in zip(getattr(replay, field), state[field])):
            raise ValueError(f'replay restore mismatch: {field}')
    for field in ('task_ids', 'class_counts'):
        if getattr(replay, field) != state[field].tolist():
            raise ValueError(f'replay restore mismatch: {field}')
    for field in ('num_seen_examples', 'replay_batches', 'replay_draw_counts'):
        if getattr(replay, field) != state[field]:
            raise ValueError(f'replay counter mismatch: {field}')
    if Counter(replay.task_ids) != {0: 76, 1: 52} or replay.num_seen_examples != 8801:
        raise ValueError('source replay does not match declared T2 checkpoint')
    print('PASS: source model, buffers, replay tensors and counters strictly restored', flush=True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def observation(model):
    state = rng_state()
    modes = [(m, m.training) for m in model.modules()]
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    try:
        yield
    finally:
        restore_rng(state)
        for module, mode in modes:
            module.training = mode
        if any(not torch.equal(v, buffers[k]) for k, v in model.named_buffers()):
            raise RuntimeError('shared implementation error: observation mutated model buffers')


def state_delta(current, old):
    max_delta = 0.
    squared = 0.
    for k, before in old.items():
        difference = current[k].detach().cpu().double() - before.double()
        max_delta = max(max_delta, float(difference.abs().max()))
        squared += float(difference.square().sum())
    return dict(max_abs=max_delta, l2=squared ** .5)


def backbone_bn(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith('backbone.') and k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def gradient_probe(model, losses):
    parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    if not parameters:
        return {'available': False, 'reason': 'backbone_frozen'}
    vectors, norms = {}, {}
    for name, loss in losses.items():
        if loss is None:
            norms[name] = None
            continue
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
                                  for p, g in zip(parameters, gradients)])
        norms[name] = float(torch.linalg.vector_norm(vectors[name], dtype=torch.float64))
    result = dict(available=True, norms=norms)
    for left, right in (('current', 'T2_replay'), ('feature', 'T2_replay')):
        result[left + '_' + right + '_cosine'] = (None if not norms[left] or not norms[right] else
            float((vectors[left] / norms[left]).dot(vectors[right] / norms[right])))
    return result


class RetentionProbe:
    def __init__(self, args, model, replay, tasks, device, stages):
        from runner_core import H5Slices, _loader
        self.args, self.model, self.replay, self.device = args, model, replay, device
        self.name, self.output = args.organ_t3_probe, args.output
        self.limit = 10 if self.name == 'frozen' else 178
        self.fixed_bn = self.name in ('frozen', 'R2', 'R3')
        self.backbone_lr = .003 if self.name in ('R1', 'R3') else .03
        self.streams = {name: np.random.RandomState(seed).get_state()
                        for name, seed in (('sample', 4401), ('reservoir', 4402))}
        self.old_heads = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if k.startswith(('heads.0.', 'heads.1.'))}
        self.initial_backbone = (copy.deepcopy(model.backbone.state_dict()) if self.name == 'frozen' else None)
        self.bn_start = backbone_bn(model)
        self.validation = []
        self.evaluation_loaders = []
        for task in tasks[:3]:
            dataset = H5Slices(args.data_root / task.folder / task.filename, 'val')
            self.evaluation_loaders.append((_loader(dataset, args.batch_size, False, 0, 4403), dataset.ends))
        self.started = time.time()
        self.before_head = self.evaluate('restored_T2', count=2)
        difference = max(abs(self.before_head[t]['benchmark_mean'] - BASELINE[t]) for t in BASELINE)
        if difference > TOLERANCE:
            write_json(self.output / 'GATE_FAILED.json', dict(gate='restored_validation', max_delta=difference))
            raise RuntimeError('restored validation does not match declared source; stop matrix')
        self.source = dict(checkpoint=str(args.t3_one_epoch_from / 's02_state.pt'),
            selected_T2_epoch=stages[1]['best_validation']['epoch'] + 1,
            actual_training_commit='65ab7ef8783a687ac9787f5419563f5491422b4e',
            replay=replay.summary(), restoration_max_dice_delta=difference)
        write_json(self.output / 'source_restore.json', self.source)

    def evaluate(self, node, count=3):
        from runner_core import evaluate
        scores = {}
        with observation(self.model):
            for i, (loader, ends) in enumerate(self.evaluation_loaders[:count]):
                generator_state = loader.generator.get_state()
                try:
                    score = evaluate(self.model, loader, ends, self.device, i, (1,),
                                     self.output / 'private' / str(node) / f'T{i+1}')
                finally:
                    loader.generator.set_state(generator_state)
                score.pop('per_patient')
                scores[f'T{i+1}'] = score
        print(json.dumps(dict(validation_node=node, scores=scores)), flush=True)
        return scores

    def check_old_heads(self):
        delta = state_delta(self.model.state_dict(), self.old_heads)
        if delta['max_abs'] != 0:
            raise RuntimeError('shared implementation error: old head state changed')
        if any(p.requires_grad for i in ('0', '1') for p in self.model.heads[i].parameters()):
            raise RuntimeError('shared implementation error: old head trainable')
        if any(m.training for i in ('0', '1') for m in self.model.heads[i].modules()):
            raise RuntimeError('shared implementation error: old head training mode')
        return delta

    def after_head(self):
        self.model.freeze_backbone_bn = self.fixed_bn
        if self.name == 'frozen':
            self.model.backbone.requires_grad_(False)
        self.model.train()
        self.check_old_heads()
        # Guard every BN invocation, covering saliency, clean, global, replay and capture.
        self.bn_forward_calls = 0
        def check_mode(module, inputs):
            self.bn_forward_calls += 1
            if self.fixed_bn and module.training:
                raise RuntimeError('shared implementation error: fixed backbone BN reopened')
        self.bn_hooks = [m.register_forward_pre_hook(check_mode) for m in self.model.backbone.modules()
                         if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        self.baseline = self.evaluate(0)
        difference = max(abs(self.baseline[t]['benchmark_mean'] - self.before_head[t]['benchmark_mean'])
                         for t in BASELINE)
        if difference > TOLERANCE:
            raise RuntimeError('head initialization changed old validation; stop matrix')
        self.validation.append(dict(step=0, scores=self.baseline))
        write_json(self.output / 'zero_step_gate.json', dict(status='passed', max_dice_delta=difference,
            old_heads=self.check_old_heads(), backbone_bn=state_delta(self.model.state_dict(), self.bn_start)))
        torch.save(self.model.heads['2'].state_dict(), self.output / 'initial_T3_head.pt')

    def parameter_groups(self):
        result = []
        params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        if params:
            result.append(dict(params=params, lr=self.backbone_lr, base_lr=self.backbone_lr, name='backbone'))
        result.append(dict(params=list(self.model.heads['2'].parameters()), lr=.03, base_lr=.03, name='T3_head'))
        return result

    def start(self, optimizer, loader, max_iterations, strategy):
        if len(loader) != 178 or len(loader.dataset) != 711 or max_iterations != 10680:
            raise RuntimeError('shared data/horizon mismatch: stop all probes')
        self.loader, self.horizon = loader, max_iterations
        self.manifest = dict(branch=self.name, status='running', max_updates=self.limit,
            executed_updates=0, batches_per_epoch=len(loader), lr_horizon=max_iterations,
            strategy=strategy, groups=[{k: g[k] for k in ('name', 'lr', 'base_lr')} for g in optimizer.param_groups],
            backbone_bn='fixed_T2_running_statistics' if self.fixed_bn else 'original_train_behavior',
            backbone_bn_affine_trainable=self.name != 'frozen', old_heads_frozen=True,
            source=self.source, validation_only=True, test_data_accessed=False, nodes=NODES,
            spatial=dict(weight=.01, first_epoch_one_based=30, active_this_probe=False),
            seeds=dict(transition=44, data_workers=44, replay_sample=4401, reservoir=4402, validation=4403),
            rng_policy='same head/data seed; isolated NumPy sample/reservoir streams; original algorithms',
            rng_limits='new standardized branches, not historical RNG continuation; CUDA grid_sample backward nondeterministic',
            argv=sys.argv, long_training_allowed=False)
        write_json(self.output / 'probe_manifest.json', self.manifest)
        write_json(self.output / 'validation_curve.json', self.validation)
        base = json.loads((self.output / 'manifest.json').read_text())
        base['task_strategies']['T3'] = strategy
        base['organ_t3_probe'] = self.name
        base['probe_max_updates'] = self.limit
        write_json(self.output / 'manifest.json', base)
        print(json.dumps(dict(actual_effective_policy=self.manifest)), flush=True)

    @contextmanager
    def random_stream(self, name):
        outer = np.random.get_state()
        np.random.set_state(self.streams[name])
        try:
            yield
        finally:
            self.streams[name] = np.random.get_state()
            np.random.set_state(outer)

    def before_backward(self, step, pce, glob, spatial, feature, rpce, rglobal, terms, replay):
        scalar = lambda x: float(x.detach())
        self.step_log = dict(step=step, losses=dict(
            current_pce=dict(raw=scalar(pce), weighted=scalar(pce)),
            current_global=dict(raw=scalar(glob), weighted=scalar(glob)),
            current_spatial=dict(raw=scalar(spatial), weighted=.01*scalar(spatial)),
            feature_mse=dict(raw=scalar(feature)/.05, weighted=scalar(feature)),
            replay_pce=dict(raw=scalar(rpce), weighted=.5*scalar(rpce)),
            replay_global=dict(raw=scalar(rglobal), weighted=.5*scalar(rglobal))))
        self.step_log['replay_tasks'] = {str(t): dict(
            samples=int(replay[3].eq(t).sum()),
            foreground_pixels=int(replay[2][replay[3].eq(t)].gt(0).sum()),
            pce_raw=scalar(v[0]), global_raw=scalar(v[1]),
            contribution_weight=.5/len(terms)) for t, v in terms.items()}
        if step in GRAD_NODES:
            t2 = None if 1 not in terms else .5 * (terms[1][0] + terms[1][1]) / len(terms)
            self.step_log['gradient_probe'] = gradient_probe(self.model,
                dict(current=pce+glob, feature=feature, T2_replay=t2))

    def after_step(self, step, optimizer, audit, grad_norm):
        old_delta = self.check_old_heads()
        bn_delta = state_delta(self.model.state_dict(), self.bn_start)
        if self.fixed_bn and bn_delta['max_abs'] != 0:
            raise RuntimeError('shared implementation error: frozen backbone BN buffers changed')
        updates = {}
        current = self.model.state_dict()
        for group_index, group in enumerate(optimizer.param_groups):
            prefix = 'backbone.' if group['name'] == 'backbone' else 'heads.2.'
            keys = [k for k, p in self.model.named_parameters() if k.startswith(prefix)]
            old = {k: audit.pre_step['model'][k] for k in keys}
            delta = state_delta(current, old)
            denom = sum(float(x.double().square().sum()) for x in old.values()) ** .5
            updates[group['name']] = dict(**delta, relative_l2=delta['l2']/denom if denom else None,
                lr_used=audit.before_step['learning_rates'][group_index],
                next_lr=group['lr'])
        self.step_log.update(status='finite', preclip_norm=grad_norm,
            clip_factor=min(1., 5./(grad_norm+1e-6)), updates=updates,
            backbone_bn_delta=bn_delta, old_heads_delta=old_delta,
            buffer_task_counts=dict(Counter(self.replay.task_ids)), num_seen_examples=self.replay.num_seen_examples)
        with (self.output / 'steps.jsonl').open('a') as f:
            f.write(json.dumps(self.step_log, sort_keys=True, allow_nan=False)+'\n')
        if step in NODES or step == self.limit:
            scores = self.evaluate(step)
            for task in BASELINE:
                baseline = self.baseline[task]['benchmark_mean']
                scores[task]['absolute_change'] = scores[task]['benchmark_mean']-baseline
                scores[task]['retention_ratio'] = scores[task]['benchmark_mean']/baseline
            self.validation.append(dict(step=step, scores=scores))
            write_json(self.output / 'validation_curve.json', self.validation)
        if step == self.limit:
            if self.name == 'frozen':
                for key, value in self.model.backbone.state_dict().items():
                    if not torch.equal(value, self.initial_backbone[key]):
                        raise RuntimeError('fully frozen backbone state changed; stop matrix')
                if max(abs(scores[t]['benchmark_mean']-self.baseline[t]['benchmark_mean']) for t in BASELINE)>TOLERANCE:
                    raise RuntimeError('fully frozen validation changed; stop matrix')
            rng = rng_state()
            torch.save(dict(model=self.model.state_dict(), continual=self.replay.state_dict(),
                optimizer=optimizer.state_dict(), stage=2, method='zs-derpp', step=step,
                task_strategy=self.manifest['strategy'], rng=rng, numpy_streams=self.streams,
                train_loader_generator=self.loader.generator.get_state(),
                recovery=dict(source=self.source, manifest=self.manifest, next_epoch=1 if step==178 else 0,
                    batch_offset=step % 178, worker_state_limit='mid-epoch worker/prefetch RNG unavailable for frozen 10-step control; full-epoch endpoints restart workers at next epoch seed')),
                self.output / 'endpoint_state.pt')
            self.manifest.update(status='completed', executed_updates=step,
                elapsed_seconds=time.time()-self.started, old_heads_delta=old_delta,
                backbone_bn_delta=bn_delta, bn_forward_calls=self.bn_forward_calls,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
            write_json(self.output / 'probe_manifest.json', self.manifest)
            write_json(self.output / 'completion.json', dict(status='completed', updates=step,
                endpoint=scores, old_heads_delta=old_delta, backbone_bn_delta=bn_delta,
                frozen_gate_passed=self.name=='frozen'))
            print(json.dumps(dict(completed=self.name, updates=step)), flush=True)
            # Deliberate hard stop before the ordinary runner's best selection or later epochs.
            raise SystemExit(0)
