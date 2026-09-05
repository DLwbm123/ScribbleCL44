#!/usr/bin/env python3
"""Read-only scalar diagnosis for one Organ-CL DER++ failure snapshot.

The report deliberately contains no images, labels, patient identifiers, or
model tensors.  It is for a failed post-step snapshot, not a training resume.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cl_methods import freeze_batchnorm_stats
from runner_core import OrganModel


def _json_safe(value: Any) -> Any:
    """Keep inherited failure metadata serializable without copying tensors."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _finite_stats(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    source = value.detach()
    floating = source.is_floating_point()
    finite = torch.isfinite(source) if floating else torch.ones_like(source, dtype=torch.bool)
    result: dict[str, Any] = {
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "finite": bool(finite.all()),
        "nonfinite": int((~finite).sum().item()),
    }
    if not source.numel() or not floating or not bool(finite.any()):
        return result
    valid = source[finite].double()
    scale = valid.abs().max()
    result.update(
        minimum=float(valid.min()),
        maximum=float(valid.max()),
        mean=float(valid.mean()),
        absmax=float(scale),
        # Dividing first makes this safe even for finite FP32 values near max.
        fp64_rms=float(scale * (valid.div(scale).square().mean().sqrt()) if scale else 0.0),
    )
    return result


def _channel_stats(value: torch.Tensor) -> dict[str, Any]:
    if value.ndim < 2 or not value.is_floating_point():
        return {"available": False}
    source = value.detach().double().transpose(0, 1).reshape(value.shape[1], -1)
    finite = torch.isfinite(source)
    per_channel = []
    for channel, mask in zip(source, finite):
        valid = channel[mask]
        per_channel.append(None if not valid.numel() else float(valid.abs().max()))
    present = torch.tensor([item for item in per_channel if item is not None], dtype=torch.float64)
    return {
        "available": bool(present.numel()),
        "channels": int(value.shape[1]),
        "finite_channels": int(present.numel()),
        "absmax_min": None if not present.numel() else float(present.min()),
        "absmax_median": None if not present.numel() else float(present.median()),
        "absmax_max": None if not present.numel() else float(present.max()),
    }


def _tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": _finite_stats(value)}
    if isinstance(value, dict):
        return {str(key): _tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {"container": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__}


def _batchnorm_details(layer: nn.modules.batchnorm._BatchNorm, input_value: torch.Tensor) -> dict[str, Any]:
    running_variance = layer.running_var.detach() if layer.running_var is not None else None
    running_mean = layer.running_mean.detach() if layer.running_mean is not None else None
    weight = layer.weight.detach() if layer.weight is not None else None
    if running_variance is None or weight is None:
        gain = None
        variance_legal = None
    else:
        denominator = (running_variance.double() + float(layer.eps)).sqrt()
        gain = weight.double().abs() / denominator
        variance_legal = bool(torch.isfinite(running_variance).all() and (running_variance >= 0).all())
    return {
        "training": bool(layer.training),
        "track_running_stats": bool(layer.track_running_stats),
        "eps": float(layer.eps),
        "momentum": None if layer.momentum is None else float(layer.momentum),
        "num_batches_tracked": None if layer.num_batches_tracked is None else int(layer.num_batches_tracked.item()),
        "running_mean": _finite_stats(running_mean),
        "running_variance": _finite_stats(running_variance),
        "weight": _finite_stats(weight),
        "bias": _finite_stats(layer.bias.detach() if layer.bias is not None else None),
        "variance_legal": variance_legal,
        "effective_abs_gain": _finite_stats(gain),
        "input_channel_absmax": _channel_stats(input_value),
    }


class _Trace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.first_bad: dict[str, Any] | None = None
        self.first_bad_input: torch.Tensor | None = None

    def record(self, name: str, direction: str, value: Any, module: nn.Module, input_value: torch.Tensor | None) -> None:
        tensor = value[0] if isinstance(value, tuple) and value else value
        if not isinstance(tensor, torch.Tensor):
            return
        item = {"module": name, "kind": type(module).__name__, "direction": direction, "tensor": _finite_stats(tensor)}
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and input_value is not None:
            item["batchnorm"] = _batchnorm_details(module, input_value)
        self.events.append(item)
        if self.first_bad is None and not item["tensor"]["finite"]:
            self.first_bad = item
            if input_value is not None:
                # One in-memory reference for a local FP64 confirmation; never serialized.
                self.first_bad_input = input_value.detach().to("cpu", dtype=torch.float64).clone()


def _trace_forward(
    model: OrganModel, image: torch.Tensor, *, running_stats: bool,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    traced = _Trace()
    handles = []
    leaf_types = (nn.Conv2d, nn.ReLU, nn.modules.batchnorm._BatchNorm, nn.MaxPool2d, nn.Upsample, nn.ConstantPad2d)
    for name, module in model.backbone.named_modules():
        if not name:
            continue
        logical_block = name in {"Pad", "Conv1", "Conv2", "Conv3", "Conv4", "Conv5", "Up4", "Up_conv4", "Up3", "Up_conv3", "Up2", "Up_conv2", "Up1", "Up_conv1"}
        if not (logical_block or isinstance(module, leaf_types)):
            continue
        handles.append(module.register_forward_pre_hook(
            lambda layer, values, module_name=name: traced.record(module_name, "input", values[0], layer, values[0])
        ))
        handles.append(module.register_forward_hook(
            lambda layer, values, output, module_name=name: traced.record(module_name, "output", output, layer, values[0] if values else None)
        ))
    try:
        context = freeze_batchnorm_stats(model.backbone) if running_stats else nullcontext()
        with torch.no_grad(), context:
            model.backbone(image)
    finally:
        for handle in handles:
            handle.remove()
    return {"events": traced.events, "first_nonfinite": traced.first_bad}, traced.first_bad_input


def _bn_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.backbone.named_buffers()
        if "running_" in name or "num_batches_tracked" in name
    }


def _buffers_unchanged(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return before.keys() == after.keys() and all(torch.equal(before[name], after[name]) for name in before)


def _stable_features(model: OrganModel, image: torch.Tensor) -> torch.Tensor:
    with freeze_batchnorm_stats(model.backbone), torch.no_grad():
        return model.backbone(image)


def _batch_stats_features(model: OrganModel, image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    probe = copy.deepcopy(model).train()
    before = _bn_buffers(probe)
    for layer in probe.backbone.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.train(True)
            layer.track_running_stats = False
    with torch.no_grad():
        output = probe.backbone(image)
    return output, _buffers_unchanged(before, _bn_buffers(probe))


def _fp64_reference(first_bad: dict[str, Any] | None, input_value: torch.Tensor | None, model: OrganModel, device: torch.device) -> dict[str, Any] | None:
    if first_bad is None or input_value is None:
        return None
    module = dict(model.backbone.named_modules()).get(first_bad["module"])
    if module is None:
        return {"available": False, "reason": "module_not_found"}
    try:
        reference = copy.deepcopy(module).to(device=device, dtype=torch.float64).eval()
        with torch.no_grad():
            output = reference(input_value.to(device))
        return {"available": True, "module": first_bad["module"], "output": _finite_stats(output)}
    except Exception as error:  # Diagnostic only: preserve the actual first-bad evidence.
        return {"available": False, "module": first_bad["module"], "error": f"{type(error).__name__}: {error}"}


def _stage_from_state(state: dict[str, Any], snapshot_metadata: dict[str, Any]) -> tuple[int, str]:
    saved = snapshot_metadata.get("stage_zero_based")
    if isinstance(saved, int) and saved >= 0:
        return saved, "snapshot_metadata"
    seen = [int(name.split(".", 2)[1]) for name in state if name.startswith("heads.") and name.split(".", 2)[1].isdigit()]
    return (max(seen) if seen else 0), "head_keys_inferred"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:4")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnosis requested but unavailable")
    payload = torch.load(args.snapshot, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict) or not isinstance(payload.get("batch"), dict):
        raise ValueError("snapshot is not a NumericalAudit failure snapshot")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    stage, stage_source = _stage_from_state(payload["model"], metadata)
    model = OrganModel().to(device)
    for index in range(1, stage + 1):
        model.activate_stage(index)
    model.load_state_dict(payload["model"], strict=True)
    model.activate_stage(stage)
    model.train()

    batch = payload["batch"]
    raw = batch.get("replay_image")
    augmented = batch.get("image")
    if not isinstance(raw, torch.Tensor) or not isinstance(augmented, torch.Tensor):
        raise ValueError("snapshot lacks raw replay_image or augmented image")
    raw, augmented = raw.to(device), augmented.to(device)
    inventory = {key: _tree(value) for key, value in payload.items() if key != "derpp"}
    inventory["derpp"] = _tree(payload.get("derpp"))
    # The buffer can dominate host RAM; every item needed below is now extracted.
    del payload

    train_trace, _ = _trace_forward(copy.deepcopy(model), raw, running_stats=False)
    trace, first_bad_input = _trace_forward(copy.deepcopy(model), raw, running_stats=True)
    running_raw = _stable_features(copy.deepcopy(model), raw)
    running_augmented = _stable_features(copy.deepcopy(model), augmented)
    batch_raw, batch_buffers_unchanged = _batch_stats_features(model, raw)
    report = {
        "schema": "organ_derpp_poststep_diagnosis_v1",
        "privacy": "scalar_metadata_only; no tensors, labels, images, paths, or identifiers",
        "snapshot": {"basename": args.snapshot.name, "metadata": metadata, "stage": stage, "stage_source": stage_source},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(), "device": str(device)},
        "snapshot_inventory": inventory,
        "reconstruction_limits": {
            "exact_historical_mode_available": False,
            "reason": "failure snapshot has no complete module-mode/pre-step/replay-draw capture",
            "reconstruction": "state_dict loaded into OrganModel; train mode and inferred active head used only for probes",
        },
        "inputs": {"raw_replay": _finite_stats(raw), "augmented_current": _finite_stats(augmented)},
        "backbone_trace_train_stat_raw_replay": train_trace,
        "backbone_trace_running_stat_raw_replay": trace,
        "feature_probes": {
            "running_stat_raw": _finite_stats(running_raw),
            "running_stat_augmented": _finite_stats(running_augmented),
            "batch_stat_raw_no_write": _finite_stats(batch_raw),
            "batch_stat_buffers_unchanged": batch_buffers_unchanged,
        },
        "fp64_first_bad_operation": _fp64_reference(trace["first_nonfinite"], first_bad_input, model, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
