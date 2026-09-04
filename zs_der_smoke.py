#!/usr/bin/env python3
"""Method-level gates for CL_Benchmark-compatible DER feature replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
runtime = ROOT / "q1d7f_class_organ"
if runtime.is_dir():
    sys.path.insert(0, str(runtime))

try:
    from cl_methods import DarkExperienceReplay
except ImportError:
    from zs_cl_methods import DarkExperienceReplay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    np.random.seed(42)
    torch.manual_seed(42)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Conv2d(1, 3, kernel_size=3, padding=1)

    model = Tiny().to(device)
    memory = DarkExperienceReplay(buffer_size=5, minibatch_size=3, alpha=0.5)
    examples = torch.randn(12, 1, 8, 8, device=device)
    with torch.no_grad():
        targets = model.backbone(examples)
    memory.add_data(examples, targets)
    exact = float(memory.penalty(model, device).detach())
    with torch.no_grad():
        model.backbone.weight.add_(0.01)
    positive = memory.penalty(model, device)
    positive.backward()
    finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    state = memory.state_dict()
    passed = (
        len(memory) == 5
        and memory.num_seen_examples == 12
        and exact < 1e-10
        and float(positive.detach()) > 0
        and finite
        and state["examples"].device.type == "cpu"
        and state["feature_targets"].shape == (5, 3, 8, 8)
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "stored_examples": len(memory),
        "seen_examples": memory.num_seen_examples,
        "exact_target_penalty": exact,
        "perturbed_penalty": float(positive.detach()),
        "finite_gradients": finite,
        "state_bytes": memory.nbytes(),
        "target_shape": list(state["feature_targets"].shape),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
