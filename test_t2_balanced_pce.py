"""Run with the existing training source on PYTHONPATH; no GPU required."""
import torch
from types import SimpleNamespace
from diagnose_t2_transition import balance_probe, class_balanced_pce

logits = torch.zeros(1, 2, 1, 4, requires_grad=True)
labels = torch.tensor([[[0, 0, 0, 1]]])
loss = class_balanced_pce(logits.log_softmax(1), labels)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
assert torch.allclose(logits.grad[0, 0, 0, :3].abs().sum(), logits.grad[0, 0, 0, 3].abs())
for labels in [torch.full((1, 1, 4), -100), torch.zeros(1, 1, 4, dtype=torch.long)]:
    logits = torch.zeros(1, 2, 1, 4, requires_grad=True)
    loss = class_balanced_pce(logits.log_softmax(1), labels)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    if bool(labels.eq(-100).all()):
        assert float(loss) == 0 and bool(logits.grad.eq(0).all())
try:
    class_balanced_pce(torch.zeros(1, 2, 1, 4), torch.full((1, 1, 4), 2))
except ValueError:
    pass
else:
    raise AssertionError('Invalid labels were not rejected')
model = SimpleNamespace(backbone=torch.nn.Linear(2, 2, bias=False))
with torch.no_grad():
    model.backbone.weight.fill_(.3)
value = model.backbone(torch.ones(1, 2))
current, feature, supervision = value.square().mean(), (value-1).square().mean(), (value+1).square().mean()
total = current + feature + supervision
expected = torch.autograd.grad(total, model.backbone.weight, retain_graph=True)[0]
records = []
audit = SimpleNamespace(note=lambda *args, **kwargs: records.append(kwargs))
balance_probe(audit, model, current, feature, supervision, torch.zeros(1, 1, 1, dtype=torch.long), None)
assert model.backbone.weight.grad is None
total.backward()
assert torch.allclose(expected, model.backbone.weight.grad)
assert abs(records[0]['current_feature_cosine'] + 1) < 1e-6
assert abs(records[0]['current_replay_supervision_cosine'] - 1) < 1e-6
print('PASS: balanced PCE edge cases; probe directions and unchanged backward gradient')
