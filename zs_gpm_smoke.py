#!/usr/bin/env python3
"""Static correctness gates for the ScribbleCL GPM implementation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
source_runtime = ROOT / "q1d7f_class_organ"
if source_runtime.is_dir():
    sys.path.insert(0, str(source_runtime))

try:
    from cl_methods import GradientProjectionMemory, sample_conv_patches
except ImportError:
    from zs_cl_methods import GradientProjectionMemory, sample_conv_patches


def patch_parity(device: torch.device) -> float:
    convolution = torch.nn.Conv2d(2, 3, kernel_size=3, stride=2, padding=1, bias=False).to(device)
    image = torch.randn(2, 2, 11, 13, device=device)
    count = 40
    seed = 1729
    sampled = sample_conv_patches(
        image,
        convolution,
        count,
        torch.Generator(device=device).manual_seed(seed),
    )
    unfolded = F.unfold(
        image,
        kernel_size=convolution.kernel_size,
        dilation=convolution.dilation,
        padding=convolution.padding,
        stride=convolution.stride,
    ).permute(0, 2, 1).reshape(-1, sampled.shape[0]).T
    locations = torch.randperm(
        unfolded.shape[1],
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )[:count]
    return float((sampled - unfolded[:, locations]).abs().max())


def synthetic_projection_gate(device: torch.device) -> dict:
    memory = GradientProjectionMemory(threshold=0.9)
    representation = torch.randn(27, 96)
    representation -= representation.mean(dim=1, keepdim=True)
    basis, summary = memory._basis_for_threshold(representation, None, 0.9)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Conv2d(3, 5, 3, bias=False))

    model = Tiny().to(device)
    layer_name = next(iter(memory.convolution_layers(model)))
    memory.bases[layer_name] = basis
    layer = memory.convolution_layers(model)[layer_name]
    layer.weight.grad = torch.randn_like(layer.weight)
    projection = memory.project_gradients(model)
    residual = layer.weight.grad.view(layer.out_channels, -1) @ basis.to(device)
    return {
        "summary": summary,
        "projection": projection,
        "orthogonality_max": float(residual.abs().max()),
    }


def model_gate(scenario: str, device: torch.device) -> dict:
    try:
        from runner_core import ClassModel, DomainModel, OrganModel, native_target, pce_loss, zs_forward
    except ModuleNotFoundError:
        from zs_setting_runner import ClassModel, DomainModel, OrganModel, native_target, pce_loss, zs_forward
    model = {"class": ClassModel, "organ": OrganModel, "domain": DomainModel}[scenario]().to(device)
    model.activate_stage(0)
    task_id = 0 if scenario == "organ" else None
    image = torch.randn(1, 1, 256, 256, device=device)
    label = torch.full((1, 256, 256), -100, dtype=torch.long, device=device)
    label[:, 24:56, 24:56] = 0
    label[:, 96:128, 96:128] = 1
    if scenario == "class":
        label[:, 144:160, 144:160] = 2
        label[:, 176:192, 176:192] = 3
    loss = pce_loss(
        zs_forward(model, image, task_id)["pred_masks"],
        native_target(label, model.output_channels(0)),
    )
    loss.backward()
    layers = GradientProjectionMemory.convolution_layers(model)
    covered = [name for name, module in layers.items() if module.weight.grad is not None]
    model.zero_grad(set_to_none=True)
    memory = GradientProjectionMemory(
        threshold=0.9,
        threshold_step=0.001,
        examples=2,
        max_patches_per_layer=16,
        max_matrix_elements=20_000,
    )
    loader = DataLoader(
        TensorDataset(image.detach().cpu().repeat(2, 1, 1, 1), label.detach().cpu().repeat(2, 1, 1)),
        batch_size=1,
        shuffle=False,
    )
    update = memory.update_from_loader(model, loader, device, 0, task_id, 42)
    second_loss = pce_loss(
        zs_forward(model, torch.randn_like(image), task_id)["pred_masks"],
        native_target(label, model.output_channels(0)),
    )
    second_loss.backward()
    projection = memory.project_gradients(model)
    orthogonality = 0.0
    for name, basis in memory.bases.items():
        gradient = layers[name].weight.grad
        if gradient is not None and basis.shape[1]:
            value = gradient.view(gradient.shape[0], -1) @ basis.to(device)
            orthogonality = max(orthogonality, float(value.abs().max()))
    return {
        "convolution_layers": len(layers),
        "covered_gradients": len(covered),
        "finite_loss": bool(torch.isfinite(loss)),
        "update_layers": len(update["layers"]),
        "minimum_retained_energy": min(row["retained_energy"] for row in update["layers"]),
        "projected_layers": projection["layers"],
        "orthogonality_max": orthogonality,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("class", "organ", "domain"), default="domain")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    parity = patch_parity(device)
    synthetic = synthetic_projection_gate(device)
    model = None if args.skip_model else model_gate(args.scenario, device)
    passed = (
        parity < 1e-6
        and synthetic["summary"]["retained_energy"] >= 0.9 - 5e-5
        and synthetic["orthogonality_max"] < 1e-5
        and synthetic["projection"]["gradient_norm_after"]
        <= synthetic["projection"]["gradient_norm_before"] + 1e-6
        and (
            model is None
            or model["finite_loss"]
            and model["covered_gradients"] == model["convolution_layers"]
            and model["update_layers"] == model["convolution_layers"]
            and model["projected_layers"] == model["convolution_layers"]
            and model["minimum_retained_energy"] >= 0.9 - 5e-5
            and model["orthogonality_max"] < 1e-4
        )
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "scenario": args.scenario,
        "patch_parity_max": parity,
        "synthetic": synthetic,
        "model": model,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
