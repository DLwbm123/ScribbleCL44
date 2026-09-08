"""Continual-learning regularizers used by the three ZS projects."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from collections import Counter, OrderedDict
from contextlib import contextmanager
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numerical_safety import NumericalAudit, require_finite


@contextmanager
def freeze_batchnorm_stats(module: nn.Module):
    """Keep replay forwards differentiable without mutating BN running state."""
    layers = [
        layer for layer in module.modules()
        if isinstance(layer, nn.modules.batchnorm._BatchNorm)
    ]
    states = [layer.training for layer in layers]
    for layer in layers:
        layer.eval()
    try:
        yield
    finally:
        for layer, state in zip(layers, states):
            layer.train(state)


def stable_backbone_features(
    model: nn.Module,
    images: torch.Tensor,
    *,
    no_grad: bool = False,
) -> torch.Tensor:
    """Run the shared backbone without updating its BatchNorm statistics."""
    with freeze_batchnorm_stats(model.backbone):
        if no_grad:
            with torch.no_grad():
                return model.backbone(images).detach()
        return model.backbone(images)


def selected_parameters(model) -> dict[str, torch.nn.Parameter]:
    source = (
        model.importance_named_parameters()
        if hasattr(model, "importance_named_parameters")
        else model.named_parameters()
    )
    return {name: parameter for name, parameter in source if parameter.requires_grad}


@dataclass
class FisherSummary:
    batches: int
    known_pixels: int
    parameter_count: int
    nonzero: int
    minimum: float
    maximum: float
    mean: float


def estimate_sparse_fisher(
    model,
    loader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_batches: int = 50,
) -> tuple[dict[str, torch.Tensor], dict]:
    parameters = selected_parameters(model)
    if not parameters:
        raise RuntimeError("empty Fisher parameter scope")
    fisher = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
    was_training = model.training
    model.eval()
    batches = known_pixels = 0
    for image, sparse in loader:
        if batches >= max_batches:
            break
        image, sparse = image.to(device), sparse.to(device)
        known = sparse.ne(-100)
        if not bool(known.any()):
            continue
        model.zero_grad(set_to_none=True)
        loss = loss_fn(image, sparse)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Fisher loss")
        loss.backward()
        for name, parameter in parameters.items():
            if parameter.grad is not None:
                fisher[name].add_(parameter.grad.detach().square())
        batches += 1
        known_pixels += int(known.sum())
    if not batches:
        raise RuntimeError("no valid sparse Fisher batches")
    for value in fisher.values():
        value.div_(batches)
    model.train(was_training)
    flat = torch.cat([value.flatten() for value in fisher.values()])
    if not bool(torch.isfinite(flat).all()) or bool((flat < 0).any()):
        raise FloatingPointError("invalid Fisher values")
    summary = FisherSummary(
        batches=batches,
        known_pixels=known_pixels,
        parameter_count=flat.numel(),
        nonzero=int((flat > 0).sum()),
        minimum=float(flat.min()),
        maximum=float(flat.max()),
        mean=float(flat.mean()),
    )
    return fisher, asdict(summary)


class OnlineEWC:
    def __init__(self, lambda_: float = 1.0, gamma: float = 0.1) -> None:
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}

    def consolidate(self, model, current: dict[str, torch.Tensor]) -> None:
        parameters = selected_parameters(model)
        if set(parameters) != set(current):
            raise ValueError("Fisher and active importance scopes differ")
        updated = {}
        for name in parameters:
            value = current[name].detach()
            if name in self.fisher:
                value = self.gamma * self.fisher[name] + value
            updated[name] = value.clone()
        self.fisher = updated
        self.anchor = {name: parameter.detach().clone() for name, parameter in parameters.items()}

    def penalty(self, model) -> torch.Tensor:
        parameters = dict(model.named_parameters())
        if not self.anchor:
            return next(model.parameters()).sum() * 0.0
        terms = []
        for name, anchor in self.anchor.items():
            if name not in parameters:
                raise KeyError(f"missing EWC parameter: {name}")
            terms.append((self.fisher[name] * (parameters[name] - anchor).square()).sum())
        return self.lambda_ * torch.stack(terms).sum()

    def state_dict(self) -> dict:
        return {
            "lambda": self.lambda_,
            "gamma": self.gamma,
            "fisher": self.fisher,
            "anchor": self.anchor,
        }

    def nbytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (*self.fisher.values(), *self.anchor.values())
        )


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def sample_conv_patches(
    activation: torch.Tensor,
    module: torch.nn.Conv2d,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample input patches in the same flattened order as a Conv2d kernel."""
    if module.groups != 1:
        raise ValueError("GPM currently requires ungrouped Conv2d layers")
    kernel_h, kernel_w = _pair(module.kernel_size)
    stride_h, stride_w = _pair(module.stride)
    padding_h, padding_w = _pair(module.padding)
    dilation_h, dilation_w = _pair(module.dilation)
    batch, channels, height, width = activation.shape
    output_h = math.floor(
        (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1
    )
    output_w = math.floor(
        (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1
    )
    total = batch * output_h * output_w
    count = min(int(count), total)
    if count < 1:
        raise ValueError("cannot sample an empty Conv2d representation")
    locations = torch.randperm(
        total,
        generator=generator,
        device=activation.device,
    )[:count]
    batch_index = locations // (output_h * output_w)
    spatial = locations % (output_h * output_w)
    output_y, output_x = spatial // output_w, spatial % output_w
    padded = F.pad(activation, (padding_w, padding_w, padding_h, padding_h))
    channel_last = padded.permute(0, 2, 3, 1)
    pieces = []
    for kernel_y in range(kernel_h):
        input_y = output_y * stride_h + kernel_y * dilation_h
        for kernel_x in range(kernel_w):
            input_x = output_x * stride_w + kernel_x * dilation_w
            pieces.append(channel_last[batch_index, input_y, input_x])
    # [sample, channel, kernel_h * kernel_w] -> [flattened kernel input, sample]
    return torch.stack(pieces, dim=2).reshape(count, channels * kernel_h * kernel_w).T


class GradientProjectionMemory:
    """Gradient Projection Memory for convolutional segmentation backbones.

    This follows the CL_Benchmark GPM task-boundary procedure: build centered
    convolution-input representation matrices after each task, retain a PCA
    subspace up to the task-specific energy threshold, and remove gradient
    components in that subspace while learning later tasks. Projection uses the
    low-rank basis directly, which is algebraically identical to multiplying by
    the explicit ``U @ U.T`` matrix used by CL_Benchmark.
    """

    def __init__(
        self,
        threshold: float = 0.97,
        threshold_step: float = 0.001,
        examples: int = 16,
        max_patches_per_layer: int = 4096,
        max_matrix_elements: int = 4_000_000,
    ) -> None:
        self.threshold = float(threshold)
        self.threshold_step = float(threshold_step)
        self.examples = int(examples)
        self.max_patches_per_layer = int(max_patches_per_layer)
        self.max_matrix_elements = int(max_matrix_elements)
        self.bases: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._device_bases: dict[tuple[str, torch.device, torch.dtype], torch.Tensor] = {}

    @staticmethod
    def convolution_layers(model) -> OrderedDict[str, torch.nn.Conv2d]:
        if not hasattr(model, "backbone"):
            raise ValueError("GPM requires a model with an explicit backbone")
        layers: OrderedDict[str, torch.nn.Conv2d] = OrderedDict()
        for name, module in model.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                layers[f"backbone.{name}"] = module
        if not layers:
            raise ValueError("GPM found no backbone Conv2d layers")
        return layers

    def task_threshold(self, stage: int) -> float:
        return min(self.threshold + int(stage) * self.threshold_step, 1.0 - 1e-7)

    @staticmethod
    def _basis_for_threshold(
        centered: torch.Tensor,
        previous: torch.Tensor | None,
        threshold: float,
    ) -> tuple[torch.Tensor, dict]:
        centered = centered.float()
        total_energy = centered.square().sum()
        if not bool(torch.isfinite(total_energy)) or float(total_energy) <= 0:
            raise FloatingPointError("invalid GPM representation energy")
        if previous is None:
            previous = centered.new_zeros((centered.shape[0], 0))
        else:
            previous = previous.to(centered)
            if previous.shape[0] != centered.shape[0]:
                raise ValueError("GPM basis and representation dimensions differ")
        residual = centered
        if previous.shape[1]:
            residual = residual - previous @ (previous.T @ residual)
        residual_energy = residual.square().sum()
        captured_before = max(0.0, min(1.0, 1.0 - float(residual_energy / total_energy)))
        add_rank = 0
        basis = previous
        if captured_before + 1e-7 < threshold and float(residual_energy) > 0:
            left, singular, _ = torch.linalg.svd(residual, full_matrices=False)
            ratios = singular.square() / total_energy
            required = threshold - captured_before
            cumulative = torch.cumsum(ratios, dim=0)
            add_rank = min(
                int(torch.searchsorted(cumulative, cumulative.new_tensor(required)).item()) + 1,
                left.shape[1],
                centered.shape[0] - previous.shape[1],
            )
            candidate = left[:, :add_rank]
            joined = torch.cat((previous, candidate), dim=1)
            basis = torch.linalg.qr(joined, mode="reduced").Q[:, : joined.shape[1]]
        captured_after = float((basis.T @ centered).square().sum() / total_energy) if basis.shape[1] else 0.0
        if captured_after + 5e-5 < threshold:
            raise RuntimeError(
                f"GPM energy gate failed: retained={captured_after:.6f}, threshold={threshold:.6f}"
            )
        return basis.cpu(), {
            "input_dimension": int(centered.shape[0]),
            "patches": int(centered.shape[1]),
            "rank_before": int(previous.shape[1]),
            "rank_added": int(add_rank),
            "rank_after": int(basis.shape[1]),
            "retained_energy": captured_after,
        }

    def update_from_loader(
        self,
        model,
        loader,
        device: torch.device,
        stage: int,
        task_id: int | None,
        seed: int,
    ) -> dict:
        layers = self.convolution_layers(model)
        expected_batches = max(1, math.ceil(self.examples / max(int(loader.batch_size or 1), 1)))
        targets: dict[str, int] = {}
        chunks: dict[str, list[torch.Tensor]] = {name: [] for name in layers}
        generators: dict[torch.device, torch.Generator] = {}

        def generator_for(value: torch.Tensor) -> torch.Generator:
            key = value.device
            if key not in generators:
                generators[key] = torch.Generator(device=key).manual_seed(int(seed))
            return generators[key]

        hooks = []
        for name, module in layers.items():
            input_dimension = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            target = min(
                self.max_patches_per_layer,
                max(1, self.max_matrix_elements // input_dimension),
            )
            targets[name] = target
            per_forward = math.ceil(target / expected_batches)

            def capture(current, inputs, layer_name=name, sample_count=per_forward):
                value = inputs[0].detach()
                patches = sample_conv_patches(
                    value,
                    current,
                    sample_count,
                    generator_for(value),
                )
                chunks[layer_name].append(patches.cpu())

            hooks.append(module.register_forward_pre_hook(capture))

        was_training = model.training
        model.eval()
        seen = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    image = batch[0]
                    remaining = self.examples - seen
                    if remaining <= 0:
                        break
                    image = image[:remaining].to(device)
                    model(image, task_id)
                    seen += int(image.shape[0])
        finally:
            for hook in hooks:
                hook.remove()
            model.train(was_training)
        if seen < 1:
            raise RuntimeError("GPM received no representation examples")

        threshold = self.task_threshold(stage)
        summaries = []
        for name in layers:
            if not chunks[name]:
                raise RuntimeError(f"GPM layer was not executed: {name}")
            matrix = torch.cat(chunks[name], dim=1)[:, : targets[name]]
            centered = matrix - matrix.mean(dim=1, keepdim=True)
            basis, summary = self._basis_for_threshold(
                centered,
                self.bases.get(name),
                threshold,
            )
            self.bases[name] = basis
            summaries.append({"layer": name, **summary})
        self._device_bases.clear()
        return {
            "stage": int(stage),
            "examples": seen,
            "threshold": threshold,
            "layers": summaries,
            "state_bytes": self.nbytes(),
        }

    @torch.no_grad()
    def project_gradients(self, model) -> dict:
        layers = self.convolution_layers(model)
        projected = 0
        before_sq = after_sq = 0.0
        for name, basis_cpu in self.bases.items():
            if name not in layers:
                raise KeyError(f"missing GPM layer: {name}")
            gradient = layers[name].weight.grad
            if gradient is None or basis_cpu.shape[1] == 0:
                continue
            key = (name, gradient.device, gradient.dtype)
            basis = self._device_bases.get(key)
            if basis is None:
                basis = basis_cpu.to(device=gradient.device, dtype=gradient.dtype)
                self._device_bases[key] = basis
            flat = gradient.view(gradient.shape[0], -1)
            before_sq += float(flat.square().sum())
            flat.sub_((flat @ basis) @ basis.T)
            after_sq += float(flat.square().sum())
            projected += 1
        return {
            "layers": projected,
            "gradient_norm_before": math.sqrt(before_sq),
            "gradient_norm_after": math.sqrt(after_sq),
        }

    def state_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "threshold_step": self.threshold_step,
            "examples": self.examples,
            "max_patches_per_layer": self.max_patches_per_layer,
            "max_matrix_elements": self.max_matrix_elements,
            "bases": self.bases,
        }

    def nbytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.bases.values())


def reservoir_index(num_seen_examples: int, buffer_size: int) -> int:
    """CL_Benchmark reservoir sampling, returning -1 when an item is dropped."""
    if num_seen_examples < buffer_size:
        return int(num_seen_examples)
    index = int(np.random.randint(0, num_seen_examples + 1))
    return index if index < buffer_size else -1


class DarkExperienceReplay:
    """Feature-level DER matching ``CL_Benchmark/models/der.py``.

    CL_Benchmark stores the shared U-Net ``features()`` tensor as the dark
    target, not final class logits. That distinction is preserved here so DER
    remains valid when Class-CL expands its output space and Organ-CL switches
    task-specific heads. Images and targets are kept on CPU between steps;
    this changes storage placement only, not the reservoir or MSE objective.
    """

    def __init__(self, buffer_size: int = 32, minibatch_size: int = 8, alpha: float = 0.5) -> None:
        self.buffer_size = int(buffer_size)
        self.minibatch_size = int(minibatch_size)
        self.alpha = float(alpha)
        if min(self.buffer_size, self.minibatch_size) < 1 or self.alpha < 0:
            raise ValueError("invalid DER controls")
        self.num_seen_examples = 0
        self.examples: list[torch.Tensor] = []
        self.feature_targets: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.examples)

    def add_data(self, examples: torch.Tensor, feature_targets: torch.Tensor) -> None:
        if examples.shape[0] != feature_targets.shape[0]:
            raise ValueError("DER examples and feature targets have different batch sizes")
        for example, target in zip(examples.detach(), feature_targets.detach()):
            index = reservoir_index(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index < 0:
                continue
            example_cpu = example.to(device="cpu", dtype=torch.float32).clone()
            target_cpu = target.to(device="cpu", dtype=torch.float32).clone()
            if index == len(self.examples):
                self.examples.append(example_cpu)
                self.feature_targets.append(target_cpu)
            else:
                self.examples[index] = example_cpu
                self.feature_targets[index] = target_cpu

    def sample(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.examples:
            raise RuntimeError("cannot sample an empty DER buffer")
        count = min(self.minibatch_size, len(self.examples))
        indices = np.random.choice(len(self.examples), size=count, replace=False).tolist()
        examples = torch.stack([self.examples[index] for index in indices]).to(device)
        targets = torch.stack([self.feature_targets[index] for index in indices]).to(device)
        return examples, targets

    def penalty(self, model, device: torch.device) -> torch.Tensor:
        if not self.examples:
            return next(model.parameters()).sum() * 0.0
        examples, targets = self.sample(device)
        features = stable_backbone_features(model, examples)
        if features.shape != targets.shape:
            raise ValueError("DER replay feature shape mismatch")
        return self.alpha * F.mse_loss(features, targets)

    def state_dict(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "minibatch_size": self.minibatch_size,
            "alpha": self.alpha,
            "num_seen_examples": self.num_seen_examples,
            "examples": None if not self.examples else torch.stack(self.examples),
            "feature_targets": (
                None if not self.feature_targets else torch.stack(self.feature_targets)
            ),
        }

    def summary(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "minibatch_size": self.minibatch_size,
            "alpha": self.alpha,
            "num_seen_examples": self.num_seen_examples,
            "stored_examples": len(self.examples),
            "state_bytes": self.nbytes(),
        }

    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.examples, *self.feature_targets)
        )


class DarkExperienceReplayPlus:
    """DER++ buffer with sparse labels and task/head metadata.

    The original CL_Benchmark DER++ adds a replayed supervised loss to the
    feature MSE. Here the replayed supervised term is computed from the
    historical sparse labels; the runner additionally applies the ZS global
    consistency loss to the same replay batch. Keeping labels and task IDs in
    the buffer is required for Class-CL and Organ-CL, where replay examples may
    belong to different output heads/class spaces.
    """

    def __init__(
        self,
        buffer_size: int = 32,
        minibatch_size: int = 8,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        self.buffer_size = int(buffer_size)
        self.minibatch_size = int(minibatch_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        if min(self.buffer_size, self.minibatch_size) < 1 or min(self.alpha, self.beta) < 0:
            raise ValueError("invalid DER++ controls")
        self.num_seen_examples = 0
        self.examples: list[torch.Tensor] = []
        self.feature_targets: list[torch.Tensor] = []
        self.sparse_labels: list[torch.Tensor] = []
        self.task_ids: list[int] = []
        self.class_counts: list[int] = []
        self.replay_batches = 0
        self.replay_draw_counts: Counter[int] = Counter()

    def __len__(self) -> int:
        return len(self.examples)

    def add_data(
        self,
        examples: torch.Tensor,
        feature_targets: torch.Tensor,
        sparse_labels: torch.Tensor,
        task_ids: torch.Tensor,
        class_counts: int,
    ) -> None:
        batch = examples.shape[0]
        if any(value.shape[0] != batch for value in (feature_targets, sparse_labels, task_ids)):
            raise ValueError("DER++ buffer fields have different batch sizes")
        require_finite("buffer_capture/examples", examples)
        require_finite("buffer_capture/feature_targets", feature_targets)
        if bool((sparse_labels.ne(-100) & sparse_labels.lt(0)).any()):
            raise ValueError("DER++ sparse labels must be -100 or non-negative")
        for example, target, label, task_id in zip(
            examples.detach(), feature_targets.detach(), sparse_labels.detach(), task_ids.detach()
        ):
            index = reservoir_index(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index < 0:
                continue
            values = (
                example.to(device="cpu", dtype=torch.float32).clone(),
                target.to(device="cpu", dtype=torch.float32).clone(),
                label.to(device="cpu", dtype=torch.int16).clone(),
                int(task_id),
                int(class_counts),
            )
            if index == len(self.examples):
                self.examples.append(values[0])
                self.feature_targets.append(values[1])
                self.sparse_labels.append(values[2])
                self.task_ids.append(values[3])
                self.class_counts.append(values[4])
            else:
                self.examples[index] = values[0]
                self.feature_targets[index] = values[1]
                self.sparse_labels[index] = values[2]
                self.task_ids[index] = values[3]
                self.class_counts[index] = values[4]

    def sample(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        if not self.examples:
            raise RuntimeError("cannot sample an empty DER++ buffer")
        count = min(self.minibatch_size, len(self.examples))
        indices = np.random.choice(len(self.examples), size=count, replace=False).tolist()
        examples = torch.stack([self.examples[index] for index in indices]).to(device)
        targets = torch.stack([self.feature_targets[index] for index in indices]).to(device)
        labels = torch.stack([self.sparse_labels[index] for index in indices]).to(device)
        require_finite("replay_feature/examples", examples)
        require_finite("replay_feature/stored_targets", targets)
        task_ids = torch.tensor([self.task_ids[index] for index in indices], device=device)
        class_counts = torch.tensor([self.class_counts[index] for index in indices], device=device)
        self.replay_batches += 1
        self.replay_draw_counts.update(int(value) for value in task_ids.tolist())
        return examples, targets, labels, task_ids, class_counts

    def feature_penalty(
        self,
        model,
        device: torch.device,
        audit: NumericalAudit | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        if not self.examples:
            zero = next(model.parameters()).sum() * 0.0
            return zero, None
        replay = self.sample(device)
        examples, targets = replay[:2]
        features = stable_backbone_features(model, examples)
        if features.shape != targets.shape:
            raise ValueError("DER++ replay feature shape mismatch")
        if audit is not None:
            audit.check("replay_feature", "features", features)
            audit.check("replay_feature", "stored_targets", targets)
        mismatch = F.mse_loss(features, targets)
        if audit is not None:
            audit.check("replay_feature", "unweighted_mse", mismatch)
        penalty = self.alpha * mismatch
        if audit is not None:
            audit.check("replay_feature", "weighted_mse", penalty)
        return penalty, replay

    def coverage(self) -> dict:
        stored = Counter(self.task_ids)
        labels = {
            task_id: [label for label, value in zip(self.sparse_labels, self.task_ids) if value == task_id]
            for task_id in sorted(stored)
        }
        label_pixels = {}
        for task_id, values in labels.items():
            joined = torch.stack(values)
            known = joined.ne(-100)
            label_pixels[str(task_id)] = {
                "known": int(known.sum().item()),
                "ignore": int((~known).sum().item()),
                "background": int(joined.eq(0).sum().item()),
                "foreground": int(joined.gt(0).sum().item()),
            }
        return {
            "stored_examples_by_task": {str(key): int(value) for key, value in sorted(stored.items())},
            "label_pixels_by_task": label_pixels,
            "replay_batches": int(self.replay_batches),
            "replay_examples_by_task": {
                str(key): int(value) for key, value in sorted(self.replay_draw_counts.items())
            },
        }

    def state_dict(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "minibatch_size": self.minibatch_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "num_seen_examples": self.num_seen_examples,
            "examples": None if not self.examples else torch.stack(self.examples),
            "feature_targets": None if not self.feature_targets else torch.stack(self.feature_targets),
            "sparse_labels": None if not self.sparse_labels else torch.stack(self.sparse_labels),
            "task_ids": None if not self.task_ids else torch.tensor(self.task_ids, dtype=torch.int64),
            "class_counts": None if not self.class_counts else torch.tensor(self.class_counts, dtype=torch.int64),
            "replay_batches": self.replay_batches,
            "replay_draw_counts": dict(self.replay_draw_counts),
            "coverage": self.coverage(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["buffer_size"]) != self.buffer_size:
            raise ValueError("DER++ buffer-size mismatch")
        if int(state["minibatch_size"]) != self.minibatch_size:
            raise ValueError("DER++ minibatch-size mismatch")
        if float(state["alpha"]) != self.alpha or float(state["beta"]) != self.beta:
            raise ValueError("DER++ coefficient mismatch")
        examples = state["examples"]
        targets = state["feature_targets"]
        labels = state["sparse_labels"]
        task_ids = state["task_ids"]
        class_counts = state["class_counts"]
        if examples is None:
            if any(value is not None for value in (targets, labels, task_ids, class_counts)):
                raise ValueError("incomplete empty DER++ state")
            self.examples = []
            self.feature_targets = []
            self.sparse_labels = []
            self.task_ids = []
            self.class_counts = []
        else:
            count = int(examples.shape[0])
            if any(value is None or int(value.shape[0]) != count for value in (targets, labels, task_ids, class_counts)):
                raise ValueError("inconsistent DER++ state")
            self.examples = [value.to(device="cpu", dtype=torch.float32).clone() for value in examples]
            self.feature_targets = [value.to(device="cpu", dtype=torch.float32).clone() for value in targets]
            self.sparse_labels = [value.to(device="cpu", dtype=torch.int16).clone() for value in labels]
            self.task_ids = [int(value) for value in task_ids.tolist()]
            self.class_counts = [int(value) for value in class_counts.tolist()]
        self.num_seen_examples = int(state["num_seen_examples"])
        self.replay_batches = int(state.get("replay_batches", 0))
        self.replay_draw_counts = Counter(
            {int(key): int(value) for key, value in state.get("replay_draw_counts", {}).items()}
        )

    def summary(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "minibatch_size": self.minibatch_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "num_seen_examples": self.num_seen_examples,
            "stored_examples": len(self.examples),
            "state_bytes": self.nbytes(),
            "coverage": self.coverage(),
        }

    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.examples, *self.feature_targets, *self.sparse_labels)
        ) + 8 * (len(self.task_ids) + len(self.class_counts))


def mib_sparse_loss(
    student_logits: torch.Tensor,
    sparse: torch.Tensor,
    old_class_count: int,
) -> torch.Tensor:
    """MiB unbiased sparse classification with old classes folded into background."""
    probabilities = student_logits.softmax(dim=1)
    known = sparse.ne(-100)
    if not bool(known.any()):
        raise ValueError("MiB batch contains no scribble pixels")
    pixel = torch.zeros_like(sparse, dtype=probabilities.dtype)
    foreground = known & sparse.ge(old_class_count)
    gathered = probabilities.gather(1, sparse.clamp_min(0).unsqueeze(1)).squeeze(1)
    pixel[foreground] = -gathered[foreground].clamp_min(1e-12).log()
    background = known & sparse.lt(old_class_count)
    old_background = probabilities[:, :old_class_count].sum(dim=1)
    pixel[background] = -old_background[background].clamp_min(1e-12).log()
    return pixel[known].mean()


def mib_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """MiB unbiased KD: student background includes background plus new classes."""
    old_class_count = teacher_logits.shape[1]
    student = student_logits.softmax(dim=1)
    teacher = teacher_logits.softmax(dim=1).detach()
    new_background = student[:, :1] + student[:, old_class_count:].sum(dim=1, keepdim=True)
    aligned = torch.cat((new_background, student[:, 1:old_class_count]), dim=1).clamp_min(1e-12)
    return -(teacher * aligned.log()).sum(dim=1).mean() / old_class_count
