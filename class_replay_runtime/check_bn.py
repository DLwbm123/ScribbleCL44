"""Bounded, paired real-input diagnosis of replay BatchNorm (no test selection)."""
import gc
import json
import random
from types import SimpleNamespace
import numpy as np
import torch
from setproctitle import setproctitle
import cl_methods as cm
import runner_core as rc
from run import ROOT

setproctitle('job')
torch.set_num_threads(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0')
cfg = json.loads((ROOT / 'paths.json').read_text())
rc.H5Slices.cache_arrays = True
task = rc.TASKS['class'][0]
from pathlib import Path
dataset = rc.H5Slices(Path(cfg['data']) / task.folder / task.filename, 'train',
    Path(cfg['sparse']) / 'class/T1_v2_s2_seed42.npz', augment=True, return_index=True)
old_features = cm.stable_features

def batch_features(model, images):
    layers = [m for m in model.backbone.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    settings = [(m.training, m.track_running_stats) for m in layers]
    try:
        for m in layers:
            m.training, m.track_running_stats = True, False
        return model.backbone(images)
    finally:
        for m, (training, tracking) in zip(layers, settings):
            m.training, m.track_running_stats = training, tracking

results = {}
for name, feature_fn in [('running_stats', old_features), ('batch_stats_no_update', batch_features)]:
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    cm.stable_features = feature_fn
    model = rc.ClassModel().to(device).train()
    buffer = cm.DarkExperienceReplayPlus(64, 4, .5, 0)
    buffer.freeze_replay_bn = True
    optimizer = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9, weight_decay=.0001)
    rows = []
    for index, batch in enumerate(rc._loader(dataset, 4, True, 0, 42)):
        if index == 32:
            break
        image, label = batch[0].to(device), batch[1].to(device)
        target = rc.native_target(label, 4)
        optimizer.zero_grad(set_to_none=True)
        out, global_loss, _, _ = rc.zs_cutout_invariance(model, image, target, None,
            SimpleNamespace(zs_adversarial_perturbation=False), device)
        replay_loss, _ = buffer.feature_penalty(model, device)
        loss = rc.pce_loss(out['pred_masks'], target) + .1 * global_loss + replay_loss
        loss.backward()
        norm = torch.stack([p.grad.detach().norm() for p in model.parameters() if p.grad is not None]).norm()
        row = dict(step=index + 1, loss=float(loss.detach()), replay=float(replay_loss.detach()), grad_norm=float(norm))
        rows.append(row)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            break
        optimizer.step()
        with torch.no_grad():
            before = {k: v.clone() for k, v in model.backbone.named_buffers()}
            features = feature_fn(model, image)
            assert all(torch.equal(before[k], v) for k, v in model.backbone.named_buffers())
            buffer.add_data(image, features, label, torch.zeros(4,device=device,dtype=torch.long), 4, batch[2])
        if index % 8 == 0:
            print(name, json.dumps(row), flush=True)
    results[name] = rows
    del model, buffer, optimizer, features, before, loss, replay_loss, out, norm, global_loss
    gc.collect(); torch.cuda.empty_cache()
dataset.close()
(ROOT / 'checks/bn_diagnosis.json').write_text(json.dumps(results, indent=2) + '\n')
print('BN_DIAGNOSIS_DONE', flush=True)
