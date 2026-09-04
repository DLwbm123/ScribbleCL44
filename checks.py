#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from runner_core import (
    ClassModel,
    H5Slices,
    OrganModel,
    TASKS,
    native_target,
    pce_loss,
    zs_cutout_invariance,
    zs_forward,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("class", "organ"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    tasks = TASKS[args.scenario]
    model = (ClassModel() if args.scenario == "class" else OrganModel()).to(device)
    shapes = []
    with torch.no_grad():
        for stage, _ in enumerate(tasks):
            model.activate_stage(stage)
            shapes.append(list(model(torch.randn(1, 1, 256, 256, device=device), stage if args.scenario == "organ" else None).shape))
    model.activate_stage(0)
    dataset = H5Slices(
        args.data_root / tasks[0].folder / tasks[0].filename,
        "train",
        args.sparse_root / args.scenario / "T1_v2_s2_seed42.npz",
    )
    samples = [dataset[index] for index in range(4)]
    image = torch.stack([sample[0] for sample in samples]).to(device)
    label = torch.stack([sample[1] for sample in samples]).to(device)
    target = native_target(label, model.output_channels(0))
    options = SimpleNamespace(zs_adversarial_perturbation=False)
    model.train()
    outputs, global_loss, gd_loss, _ = zs_cutout_invariance(
        model,
        image,
        target,
        0 if args.scenario == "organ" else None,
        options,
        device,
    )
    partial_ce = pce_loss(outputs["pred_masks"], target)
    loss = partial_ce + global_loss + gd_loss
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    expected_shapes = (
        [[1, 4, 244, 244], [1, 6, 244, 244], [1, 8, 244, 244]]
        if args.scenario == "class"
        else [[1, 2, 244, 244]] * 4
    )
    result = {
        "status": "PASS" if shapes == expected_shapes and finite_gradients and bool(torch.isfinite(loss)) else "FAIL",
        "scenario": args.scenario,
        "stage_shapes": shapes,
        "expected_shapes": expected_shapes,
        "pce": float(partial_ce.detach()),
        "global": float(global_loss.detach()),
        "gd": float(gd_loss.detach()),
        "finite_gradients": finite_gradients,
        "known_labels": np.unique(label.cpu().numpy()).tolist(),
    }
    dataset.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
