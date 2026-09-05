"""Small, fail-fast numerical guards for the ZS continual-learning runner."""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F


class NumericalFailure(FloatingPointError):
    """A finite guard with compact, JSON-safe evidence for the first failure."""

    def __init__(self, name: str, details: dict[str, Any]) -> None:
        super().__init__(f"{name}: non-finite value")
        self.name = name
        self.details = details


def tensor_summary(value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    """Return metadata only: no patient pixels or model tensors leave the server."""
    if isinstance(value, torch.Tensor):
        source = value.detach()
        finite = torch.isfinite(source) if source.is_floating_point() else torch.ones_like(source, dtype=torch.bool)
        item: dict[str, Any] = {
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "finite": bool(finite.all()),
            "nonfinite": int((~finite).sum().item()),
        }
        if source.numel() and source.is_floating_point() and bool(finite.any()):
            valid = source[finite]
            item.update(min=float(valid.min()), max=float(valid.max()), mean=float(valid.double().mean()))
        return item
    source = np.asarray(value)
    finite = np.isfinite(source) if np.issubdtype(source.dtype, np.floating) else np.ones(source.shape, dtype=bool)
    item = {
        "shape": list(source.shape), "dtype": str(source.dtype),
        "finite": bool(finite.all()), "nonfinite": int((~finite).sum()),
    }
    if source.size and np.issubdtype(source.dtype, np.number) and finite.any():
        valid = source[finite]
        item.update(min=float(valid.min()), max=float(valid.max()), mean=float(valid.astype(np.float64).mean()))
    return item


def _cpu_copy(value: Any) -> Any:
    """Clone a debug-only failure capture so later optimizer steps cannot alias it."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return copy.deepcopy(value)


@dataclass
class NumericalAudit:
    """Collect only scalar diagnostics; tensors stay in private failure snapshots."""

    output: Path
    enabled: bool = False
    stage: int | None = None
    epoch: int | None = None
    iteration: int | None = None
    task_id: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    batch: dict[str, torch.Tensor] = field(default_factory=dict)
    pending: list[tuple[str, str, torch.Tensor | np.ndarray]] = field(default_factory=list)
    pre_step: dict[str, Any] | None = None
    before_step: dict[str, Any] | None = None

    def begin(
        self,
        *,
        stage: int,
        epoch: int,
        iteration: int,
        task_id: int | None,
        image: torch.Tensor,
        label: torch.Tensor,
        replay_image: torch.Tensor | None = None,
        replay_label: torch.Tensor | None = None,
        transform_trace: torch.Tensor | None = None,
    ) -> None:
        self.stage, self.epoch, self.iteration, self.task_id = stage, epoch, iteration, task_id
        self.events.clear()
        self.pending.clear()
        self.pre_step = None
        self.before_step = None
        self.batch = {"image": image.detach(), "label": label.detach()}
        if replay_image is not None:
            self.batch["replay_image"] = replay_image.detach()
        if replay_label is not None:
            self.batch["replay_label"] = replay_label.detach()
        if transform_trace is not None:
            self.batch["transform_trace"] = transform_trace.detach()
        self.check("current_global", "input", image)
        self.check_labels("current_global", "sparse_label", label)
        if replay_image is not None:
            self.check("buffer_capture", "raw_replay_input", replay_image)
        if replay_label is not None:
            self.check_labels("buffer_capture", "raw_replay_label", replay_label)

    def capture_pre_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        derpp_summary: dict[str, Any] | None,
    ) -> None:
        """Hold one debug-only, non-aliased candidate pre-step state in memory."""
        if not self.enabled:
            return
        self.pre_step = {
            "model": _cpu_copy(model.state_dict()),
            "optimizer": _cpu_copy(optimizer.state_dict()),
            "active_stage": getattr(model, "active_stage", None),
            "requires_grad": {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()},
            "module_training": {name: bool(module.training) for name, module in model.named_modules()},
            "derpp_summary": _cpu_copy(derpp_summary),
            "rng": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        self.before_step = None

    def attach_replay(self, replay: tuple[torch.Tensor, ...] | None) -> None:
        if self.enabled and self.pre_step is not None and replay is not None:
            names = ("examples", "feature_targets", "sparse_labels", "task_ids", "class_counts")
            self.pre_step["actual_replay"] = {
                name: _cpu_copy(value) for name, value in zip(names, replay)
            }

    def capture_before_step(self, *, losses: dict[str, torch.Tensor], gradient_norm: float) -> None:
        if self.enabled and self.pre_step is not None:
            self.before_step = {
                "losses": {name: float(value.detach().double()) for name, value in losses.items()},
                "gradient_norm": float(gradient_norm),
                "learning_rates": [float(group["lr"]) for group in self.pre_step["optimizer"]["param_groups"]],
            }

    def record_success(self) -> None:
        """Append scalar-only evidence for a debug step that completed finite."""
        if not self.enabled:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        entry = {
            **self.metadata(),
            "status": "post_step_finite",
            "before_step": self.before_step,
            "events": self.events,
        }
        with (self.output / "numerical_trace.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")

    def metadata(self) -> dict[str, Any]:
        return {
            "stage_zero_based": self.stage,
            "epoch_zero_based": self.epoch,
            "iteration_one_based": self.iteration,
            "task_id": self.task_id,
        }

    def check(self, branch: str, name: str, value: torch.Tensor | np.ndarray) -> None:
        if self.enabled:
            self.pending.append((branch, name, value))
            return
        elif isinstance(value, torch.Tensor):
            finite = bool(torch.isfinite(value).all()) if value.is_floating_point() else True
            summary = {"shape": list(value.shape), "dtype": str(value.dtype), "finite": finite}
        else:
            array = np.asarray(value)
            finite = bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.floating) else True
            summary = {"shape": list(array.shape), "dtype": str(array.dtype), "finite": finite}
        if not finite:
            raise NumericalFailure(
                f"{branch}/{name}",
                {**self.metadata(), "events": self.events, "summary": summary},
            )

    def flush(self) -> None:
        """Check every deferred tensor in one device synchronization per device."""
        if not self.enabled or not self.pending:
            return
        groups: dict[str, list[tuple[str, str, torch.Tensor]]] = {}
        arrays: list[tuple[str, str, np.ndarray]] = []
        for branch, name, value in self.pending:
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    groups.setdefault(str(value.device), []).append((branch, name, value))
                else:
                    self.events.append({"branch": branch, "name": name, "shape": list(value.shape), "dtype": str(value.dtype), "finite": True})
            else:
                arrays.append((branch, name, np.asarray(value)))
        tensors_finite = all(
            bool(torch.stack([torch.isfinite(value).all() for _, _, value in group]).all())
            for group in groups.values()
        )
        arrays_finite = all(
            not np.issubdtype(value.dtype, np.floating) or bool(np.isfinite(value).all())
            for _, _, value in arrays
        )
        if tensors_finite and arrays_finite:
            for group in groups.values():
                self.events.extend(
                    {"branch": branch, "name": name, "shape": list(value.shape), "dtype": str(value.dtype), "finite": True}
                    for branch, name, value in group
                )
            self.events.extend(
                {"branch": branch, "name": name, "shape": list(value.shape), "dtype": str(value.dtype), "finite": True}
                for branch, name, value in arrays
            )
            self.pending.clear()
            return
        for branch, name, value in self.pending:
            summary = tensor_summary(value)
            self.events.append({"branch": branch, "name": name, **summary})
            if not summary["finite"]:
                self.pending.clear()
                raise NumericalFailure(
                    f"{branch}/{name}",
                    {**self.metadata(), "events": self.events, "summary": summary},
                )
        raise NumericalFailure("numerical_audit/unknown", {**self.metadata(), "events": self.events})

    def check_labels(self, branch: str, name: str, labels: torch.Tensor, classes: int | None = None) -> None:
        item = tensor_summary(labels)
        known = labels.ne(-100)
        item.update(
            known_pixels=int(known.sum().item()),
            background_pixels=int(labels.eq(0).sum().item()),
            foreground_pixels=int(labels.gt(0).sum().item()),
        )
        valid = labels.eq(-100) | labels.ge(0)
        if classes is not None:
            valid = valid & (labels.eq(-100) | labels.lt(classes))
        if self.enabled:
            self.events.append({"branch": branch, "name": name, **item})
        if not bool(valid.all()):
            raise NumericalFailure(f"{branch}/{name}_range", {**self.metadata(), "events": self.events})

    def note(self, branch: str, name: str, **values: Any) -> None:
        if self.enabled:
            self.events.append({"branch": branch, "name": name, **values})

    def check_gradients(self, model: torch.nn.Module) -> float:
        gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
        finite = not gradients or bool(torch.stack([torch.isfinite(value).all() for _, value in gradients]).all())
        if not finite:
            for name, value in gradients:
                self.check("total_backward", f"gradient/{name}", value)
            raise NumericalFailure("total_backward/gradient", {**self.metadata(), "events": self.events})
        if not self.enabled:
            return 0.0
        # FP32 squaring can overflow for entirely finite gradients.
        scale = max((value.detach().abs().max().double() for _, value in gradients), default=torch.zeros(()))
        norm = 0.0 if not bool(scale) else float(scale * torch.stack([
            value.detach().double().div(scale).square().sum() for _, value in gradients
        ]).sum().sqrt())
        self.note("total_backward", "global_grad_norm", value=norm)
        if not np.isfinite(norm):
            raise NumericalFailure("total_backward/global_grad_norm", {**self.metadata(), "events": self.events})
        return norm

    def check_model_state(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        values = [(f"parameter/{name}", parameter) for name, parameter in model.named_parameters()]
        values.extend((f"buffer/{name}", value) for name, value in model.named_buffers() if value.is_floating_point())
        values.extend(
            (f"optimizer/{index}/{name}", value)
            for index, state in enumerate(optimizer.state.values())
            for name, value in state.items()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        )
        finite = not values or bool(torch.stack([torch.isfinite(value).all() for _, value in values]).all())
        if not finite:
            for name, value in values:
                self.check("optimizer_step", name, value)
            raise NumericalFailure("optimizer_step/state", {**self.metadata(), "events": self.events})
        if self.enabled:
            self.note("optimizer_step", "finite_state_tensors", count=len(values))

    def save_failure(
        self,
        failure: BaseException,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        derpp_state: dict[str, Any] | None,
    ) -> None:
        if not self.enabled or (self.output / "FIRST_NONFINITE.json").exists():
            return
        self.output.mkdir(parents=True, exist_ok=True)
        self.events.extend(
            {"branch": branch, "name": name, "shape": list(value.shape), "dtype": str(value.dtype), "deferred": True}
            for branch, name, value in self.pending
        )
        self.pending.clear()
        details = {**self.metadata(), "events": self.events}
        if isinstance(failure, NumericalFailure):
            details.update(failure.details)
        details.update(exception=type(failure).__name__, message=str(failure))
        (self.output / "FIRST_NONFINITE.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "derpp": derpp_state,
                "batch": {name: value.detach().cpu() for name, value in self.batch.items()},
                "capture": _cpu_copy({"pre_step": self.pre_step, "before_step": self.before_step}),
                "rng": {
                    "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                "metadata": details,
            },
            self.output / "FIRST_NONFINITE.pt",
        )


def require_finite(name: str, value: torch.Tensor | np.ndarray) -> None:
    summary = tensor_summary(value)
    if not summary["finite"]:
        raise NumericalFailure(name, summary)


def rms_saliency(gradient: torch.Tensor) -> torch.Tensor:
    """Overflow-safe RMS across channels. Single-channel RMS is exactly abs.

    This is a detached saliency calculation, not a replacement for the
    differentiable training loss. Exact zeros are not artificially inflated.
    """
    if gradient.ndim != 4 or not gradient.is_floating_point():
        raise ValueError("gradient must be floating [B,C,H,W]")
    g = gradient.detach().float()
    require_finite("input_gradient", g)
    if g.shape[1] == 1:
        return g[:, 0].abs()
    scale = g.abs().amax(dim=1, keepdim=True)
    denom = torch.where(scale > 0, scale, torch.ones_like(scale))
    result = (g / denom).square().mean(dim=1).sqrt() * scale[:, 0]
    require_finite("rms_saliency", result)
    return result


def normalized_unary(saliency: torch.Tensor, block_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Normalize nonnegative saliency per image; use a uniform zero-map prior.

    A sample-wide rescaling BEFORE pooling cancels mathematically in the final
    normalization and avoids overflow/underflow for very large/small maps.
    Non-finite inputs are errors, never silently replaced by zero.
    """
    if saliency.ndim != 3 or not saliency.is_floating_point():
        raise ValueError("saliency must be floating [B,H,W]")
    if block_size < 1 or saliency.shape[-2] % block_size or saliency.shape[-1] % block_size:
        raise ValueError("positive block_size must divide both spatial dimensions")
    s = saliency.detach().float()
    require_finite("saliency_before_normalization", s)
    if bool((s < 0).any()):
        raise ValueError("saliency must be nonnegative")
    peak = s.amax(dim=(-2, -1), keepdim=True)
    degenerate = peak.eq(0)
    scaled = s / torch.where(degenerate, torch.ones_like(peak), peak)
    pooled = F.avg_pool2d(scaled.unsqueeze(1), block_size).squeeze(1)
    mass = pooled.sum(dim=(-2, -1), keepdim=True)
    zero_mass = mass.eq(0)
    denominator = torch.where(zero_mass, torch.ones_like(mass), mass)
    uniform = torch.full_like(pooled, 1.0 / (pooled.shape[-2] * pooled.shape[-1]))
    unary = torch.where(zero_mass, uniform, pooled / denominator)
    require_finite("normalized_unary", unary)
    return unary, {
        "zero_saliency_samples": int(degenerate.sum().item()),
        "zero_saliency_sample_indices": torch.where(degenerate.flatten())[0].cpu().tolist(),
        "normalization_policy": "per_image_max_rescale_then_sum;uniform_for_exact_zero",
    }


def checked_int32_cost(name: str, cost: np.ndarray) -> np.ndarray:
    """Validate floating costs BEFORE cast; retain the existing truncation rule."""
    value = np.asarray(cost, dtype=np.float64)
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name}: non-finite graph-cut cost before int32 cast")
    limits = np.iinfo(np.int32)
    if value.size and (value.min() < limits.min or value.max() > limits.max):
        raise OverflowError(f"{name}: graph-cut cost outside int32 range")
    return np.ascontiguousarray(value.astype(np.int32))


def validate_graph_labels(labels: np.ndarray, classes: int) -> None:
    values = np.asarray(labels)
    if classes < 2 or not np.isfinite(values).all():
        raise FloatingPointError("invalid graph-cut label values")
    if not np.equal(values, np.floor(values)).all() or (values < 0).any() or (values >= classes).any():
        raise ValueError("graph-cut labels must be integer IDs in [0, classes)")


def log_probability_resize(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Stable log(interpolate(softmax(logits))) for bilinear align_corners=False.

    Unlike interpolating logits then calling CE, this preserves the runner's
    original probability-space interpolation convention (up to rounding).
    It is a separately testable OPTIONAL PCE integration helper.
    """
    require_finite("native_logits", logits)
    if logits.ndim != 4 or min(size) < 1:
        raise ValueError("expected logits [B,C,H,W] and a positive output size")
    z = logits.float()
    logp = F.log_softmax(z, dim=1)
    require_finite("native_log_probabilities", logp)
    ih, iw = logp.shape[-2:]
    oh, ow = size
    if (ih, iw) == (oh, ow):
        return logp
    y = ((torch.arange(oh, device=z.device, dtype=z.dtype) + .5) * (ih / oh) - .5).clamp_min(0)
    x = ((torch.arange(ow, device=z.device, dtype=z.dtype) + .5) * (iw / ow) - .5).clamp_min(0)
    y0 = y.floor().long().clamp_max(ih - 1)
    x0 = x.floor().long().clamp_max(iw - 1)
    y1, x1 = (y0 + 1).clamp_max(ih - 1), (x0 + 1).clamp_max(iw - 1)
    dy, dx = y - y0, x - x0
    terms = []
    for yi, wy in ((y0, 1 - dy), (y1, dy)):
        for xi, wx in ((x0, 1 - dx), (x1, dx)):
            weight = wy[:, None] * wx[None, :]
            # Weights have no autograd dependency. log(0)=-inf excludes a term.
            terms.append(logp[:, :, yi[:, None], xi[None, :]] + weight.log()[None, None])
    resized = torch.logsumexp(torch.stack(terms, dim=0), dim=0)
    # The original runner also renormalizes after interpolation.
    resized = resized - torch.logsumexp(resized, dim=1, keepdim=True)
    require_finite("resized_log_probabilities", resized)
    return resized


def sparse_pce_from_log_probs(logp: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    require_finite("log_probabilities_for_pce", logp)
    if labels.dtype != torch.long or labels.shape != (logp.shape[0], *logp.shape[-2:]):
        raise ValueError("labels must be long [B,H,W] aligned to predictions")
    known = labels.ne(-100)
    if bool((known & ((labels < 0) | (labels >= logp.shape[1]))).any()):
        raise ValueError("invalid sparse label")
    if not bool(known.any()):
        # Avoid summing huge negative finite values before multiplication by 0.
        return (logp * 0).sum()
    gathered = logp.gather(1, labels.clamp_min(0).unsqueeze(1)).squeeze(1)
    return -gathered[known].mean()
