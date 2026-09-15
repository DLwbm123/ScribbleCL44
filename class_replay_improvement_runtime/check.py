"""Small checks for partial labels, frozen BN/heads, and refreshed feature targets."""
import math
from types import SimpleNamespace
from setproctitle import setproctitle
setproctitle("job")
import torch
from torch import nn
import runner_core as r
from cl_methods import DarkExperienceReplayPlus

torch.set_num_threads(1)
p = torch.tensor([.1, .2, .3, .4]).view(1, 4, 1, 1).requires_grad_()
y = torch.zeros((1, 1, 1), dtype=torch.long)
loss = r.partial_background_pce(p, y, (3,))
assert math.isclose(loss.item(), -math.log(.6), rel_tol=1e-6)
loss.backward()
assert p.grad[0, 1, 0, 0] < 0 and p.grad[0, 3, 0, 0] == 0
assert r.partial_background_pce(p, y.fill_(-100), (3,)).item() == 0
try:
    r.partial_background_pce(p, y.fill_(2), (3,))
    raise AssertionError("invalid class was accepted")
except ValueError:
    pass
r._native_backbone = lambda: nn.Sequential(nn.Conv2d(1, 64, 1), nn.BatchNorm2d(64))
m = r.ClassModel()
m.improved_replay = m.lock_old_heads = True
m.activate_stage(1)
m.train()
assert all(not v.training for v in m.modules() if isinstance(v, nn.BatchNorm2d))
assert all(not v.requires_grad for v in m.blocks[0].parameters())
assert all(v.requires_grad for v in m.blocks[1].parameters())
buffers = {k: v.clone() for k, v in m.named_buffers()}
x = torch.randn(2, 1, 8, 8)
m(x).sum().backward()
assert all(torch.equal(v, buffers[k]) for k, v in m.named_buffers())
b = DarkExperienceReplayPlus(2, 2, .5, 0)
b.add_data(x, torch.full((2, 64, 8, 8), 100.), torch.zeros((2, 8, 8), dtype=torch.long),
           torch.zeros(2, dtype=torch.long), 4)
r.refresh_feature_targets(m, b, torch.device("cpu"))
penalty, _ = b.feature_penalty(m, torch.device("cpu"))
assert penalty.item() < 1e-10
assert m.training and all(not v.training for v in m.modules() if isinstance(v, nn.BatchNorm2d))
print("IMPROVEMENT_CHECK_PASSED", flush=True)

