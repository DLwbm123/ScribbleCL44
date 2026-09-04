#!/usr/bin/env python3
"""CPU smoke test for canonical DER++ state, coverage, and BatchNorm safety."""
from __future__ import annotations

import json

import numpy as np
import torch

from cl_methods import DarkExperienceReplayPlus, stable_backbone_features


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(1, 3, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward_logits(self, image: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        return self.head(self.backbone(image))


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    model = Tiny().train()
    memory = DarkExperienceReplayPlus(buffer_size=8, minibatch_size=4, alpha=0.5, beta=0.5)
    examples = torch.randn(12, 1, 8, 8)
    labels = torch.full((12, 8, 8), -100, dtype=torch.int64)
    labels[:, 2:4, 2:4] = 1
    labels[:, 5:6, 5:7] = 0
    task_ids = torch.tensor([0] * 6 + [1] * 6, dtype=torch.int64)
    targets = stable_backbone_features(model, examples, no_grad=True)
    memory.add_data(examples, targets, labels, task_ids, 2)
    bn = model.backbone[1]
    before = bn.running_mean.detach().clone(), bn.running_var.detach().clone()
    feature_loss, replay = memory.feature_penalty(model, torch.device("cpu"))
    after = bn.running_mean.detach().clone(), bn.running_var.detach().clone()
    assert replay is not None
    replay_examples, _, replay_labels, replay_tasks, replay_classes = replay
    probabilities = model.forward_logits(replay_examples).softmax(dim=1)
    known = replay_labels.ne(-100)
    pce_loss = -probabilities.gather(1, replay_labels.long().clamp_min(0).unsqueeze(1)).squeeze(1)[known].log().mean()
    total = feature_loss + memory.beta * pce_loss
    total.backward()
    restored = DarkExperienceReplayPlus(buffer_size=8, minibatch_size=4, alpha=0.5, beta=0.5)
    restored.load_state_dict(memory.state_dict())
    coverage = restored.coverage()
    finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    bn_unchanged = bool(torch.equal(before[0], after[0]) and torch.equal(before[1], after[1]))
    passed = (
        len(memory) == 8
        and memory.num_seen_examples == 12
        and feature_loss < 1e-10
        and float(pce_loss) > 0
        and finite
        and bn_unchanged
        and sum(coverage["stored_examples_by_task"].values()) == 8
        and sum(coverage["replay_examples_by_task"].values()) == 4
        and coverage["replay_batches"] == 1
        and replay_tasks.shape == (4,)
        and replay_classes.shape == (4,)
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "stored_examples": len(memory),
        "seen_examples": memory.num_seen_examples,
        "feature_loss_exact_target": float(feature_loss.detach()),
        "replay_pce_loss": float(pce_loss.detach()),
        "batchnorm_unchanged_by_feature_replay": bn_unchanged,
        "coverage": coverage,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
