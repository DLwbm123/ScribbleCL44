#!/usr/bin/env python3
"""ZScribbleSeg continual runner for the Class-CL and Organ-CL projects."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from cutout import Cutout, rotate_back, rotate_invariant
from mixup import mixup_process
from models import build_model
from spatial_function import ModelWeightGatedCRF
from cl_methods import (
    DarkExperienceReplay,
    DarkExperienceReplayPlus,
    GradientProjectionMemory,
    OnlineEWC,
    estimate_sparse_fisher,
    mib_distillation_loss,
    mib_sparse_loss,
)


IGNORE_INDEX = -100


@dataclass(frozen=True)
class Task:
    code: str
    folder: str
    filename: str
    classes: tuple[int, ...]
    label_shift: int = 0


TASKS = {
    "class": (
        Task("T1", "MMWHS", "myo_lv_la.h5", (1, 2, 3), 0),
        Task("T2", "MMWHS", "ra_rv.h5", (4, 5), 3),
        Task("T3", "MMWHS", "ao_pa.h5", (6, 7), 5),
    ),
    "organ": (
        Task("T1", "Task_incre", "UtahI.h5", (1,)),
        Task("T2", "Task_incre", "UCL.h5", (1,)),
        Task("T3", "Task_incre", "Lits.h5", (1,)),
        Task("T4", "Task_incre", "brain.h5", (1,)),
    ),
    "domain": (
        Task("A", "Domain_Prostate", "BIDMC.h5", (1,)),
        Task("B", "Domain_Prostate", "HK.h5", (1,)),
        Task("C", "Domain_Prostate", "ISBI.h5", (1,)),
        Task("D", "Domain_Prostate", "UCL.h5", (1,)),
        Task("E", "Domain_Prostate", "ISBI_1.5.h5", (1,)),
        Task("F", "Domain_Prostate", "I2CVB.h5", (1,)),
    ),
}


class H5Slices(Dataset):
    def __init__(
        self,
        path: Path,
        split: str,
        sparse_path: Path | None = None,
        label_shift: int = 0,
        augment: bool = False,
    ) -> None:
        self.path = str(path)
        self.split = split
        self.label_shift = int(label_shift)
        self.augment = augment
        self._handle = None
        with h5py.File(self.path, "r") as handle:
            self.length = int(handle[f"{split}_images"].shape[2])
            self.ends = None if split == "train" else np.asarray(handle[f"patient_info_{split}"], dtype=np.int64)
        self.sparse = None
        if split == "train":
            if sparse_path is None:
                raise ValueError("training requires a sparse annotation archive")
            archive = np.load(sparse_path, allow_pickle=False)
            if set(archive.files) != {"annotations"}:
                raise ValueError("sparse archive contract violation")
            self.sparse = np.asarray(archive["annotations"], dtype=np.int16)
            if self.sparse.shape != (self.length, 256, 256):
                raise ValueError("sparse annotation shape mismatch")

    def __len__(self) -> int:
        return self.length

    def _open(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        handle = self._open()
        image = np.asarray(handle[f"{self.split}_images"][:, :, index], dtype=np.float32)
        if self.sparse is not None:
            label = self.sparse[index].astype(np.int64, copy=True)
        else:
            label = np.asarray(handle[f"{self.split}_labels"][:, :, index]).astype(np.int64)
            if self.label_shift:
                label = np.where(label > 0, label + self.label_shift, 0)
        if self.augment:
            if random.random() > 0.5:
                turns, axis = np.random.randint(0, 4), np.random.randint(0, 2)
                image = np.flip(np.rot90(image, turns), axis).copy()
                label = np.flip(np.rot90(label, turns), axis).copy()
            elif random.random() > 0.5:
                angle = np.random.randint(-20, 20)
                image = ndimage.rotate(image, angle, order=0, reshape=False)
                label = ndimage.rotate(label, angle, order=0, reshape=False)
        return torch.from_numpy(image[None].copy()), torch.from_numpy(label.copy()).long()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.sparse = None


def _match_spatial(skip: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    height, width = reference.shape[-2:]
    top = (skip.shape[-2] - height) // 2
    left = (skip.shape[-1] - width) // 2
    return skip[:, :, top:top + height, left:left + width]


class ZSBackbone(nn.Module):
    """The native ZScribbleSeg U-Net up to its final 64-channel feature map."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        for name in (
            "Pad", "Maxpool1", "Maxpool2", "Maxpool3", "Maxpool4",
            "Conv1", "Conv2", "Conv3", "Conv4", "Conv5",
            "Up4", "Up_conv4", "Up3", "Up_conv3",
            "Up2", "Up_conv2", "Up1", "Up_conv1",
        ):
            setattr(self, name, getattr(source, name))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.Conv1(self.Pad(image))
        e2 = self.Conv2(self.Maxpool1(e1))
        e3 = self.Conv3(self.Maxpool2(e2))
        e4 = self.Conv4(self.Maxpool3(e3))
        e5 = self.Conv5(self.Maxpool4(e4))
        d4 = self.Up4(e5)
        d4 = self.Up_conv4(torch.cat((d4, _match_spatial(e4, d4)), dim=1))
        d3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((d3, _match_spatial(e3, d3)), dim=1))
        d2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((d2, _match_spatial(e2, d2)), dim=1))
        d1 = self.Up1(d2)
        return self.Up_conv1(torch.cat((d1, _match_spatial(e1, d1)), dim=1))


class OutputHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(64, channels, 1, bias=True)
        self.norm = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(features))


def _native_backbone() -> ZSBackbone:
    options = SimpleNamespace(
        device="cpu",
        tasks={"MR": {"lab_values": [0, 1], "out_channels": 2}},
        out_channels=2,
        frozen_weights=None,
        in_channels=1,
        multiDice_loss_coef=0.0,
        CrossEntropy_loss_coef=1.0,
        Rv=1.0,
        Lv=1.0,
        Myo=1.0,
        Avg=1.0,
    )
    source, _, _, _ = build_model(options)
    return ZSBackbone(source.UNet)


class ClassModel(nn.Module):
    block_sizes = (3, 2, 2)

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _native_backbone()
        self.background = OutputHead(1)
        self.blocks = nn.ModuleList(OutputHead(size) for size in self.block_sizes)
        self.active_stage = 0

    def activate_stage(self, stage: int) -> None:
        self.active_stage = int(stage)

    def forward_logits(self, image: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        stage = self.active_stage if task_id is None else int(task_id)
        features = self.backbone(image)
        logits = [self.background(features)]
        logits.extend(self.blocks[index](features) for index in range(stage + 1))
        return torch.cat(logits, dim=1)

    def forward(self, image: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        return torch.softmax(self.forward_logits(image, task_id), dim=1)

    def output_channels(self, stage: int) -> int:
        return 1 + sum(self.block_sizes[:stage + 1])

    def importance_named_parameters(self):
        for name, parameter in self.backbone.named_parameters():
            yield f"backbone.{name}", parameter
        for name, parameter in self.background.named_parameters():
            yield f"background.{name}", parameter
        for index in range(self.active_stage + 1):
            for name, parameter in self.blocks[index].named_parameters():
                yield f"blocks.{index}.{name}", parameter


class OrganModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _native_backbone()
        self.heads = nn.ModuleDict({"0": OutputHead(2)})
        self.active_stage = 0

    def activate_stage(self, stage: int) -> None:
        stage = int(stage)
        key = str(stage)
        self.active_stage = stage
        if key not in self.heads:
            reference = next(self.backbone.parameters())
            self.heads[key] = OutputHead(2).to(reference.device)
        for index, head in self.heads.items():
            enabled = int(index) == stage
            for parameter in head.parameters():
                parameter.requires_grad_(enabled)
            if not enabled:
                head.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for index, head in self.heads.items():
                if int(index) != self.active_stage:
                    head.eval()
        return self

    def forward_logits(self, image: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        stage = self.active_stage if task_id is None else int(task_id)
        return self.heads[str(stage)](self.backbone(image))

    def forward(self, image: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        return torch.softmax(self.forward_logits(image, task_id), dim=1)

    @staticmethod
    def output_channels(stage: int) -> int:
        return 2

    def importance_named_parameters(self):
        for name, parameter in self.backbone.named_parameters():
            yield f"backbone.{name}", parameter


class DomainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _native_backbone()
        self.head = OutputHead(2)
        self.active_stage = 0

    def activate_stage(self, stage: int) -> None:
        self.active_stage = int(stage)

    def forward_logits(self, image: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        return self.head(self.backbone(image))

    def forward(self, image: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        return torch.softmax(self.forward_logits(image, task_id), dim=1)

    @staticmethod
    def output_channels(stage: int) -> int:
        return 2

    def importance_named_parameters(self):
        return self.named_parameters()


def native_target(labels: torch.Tensor, classes: int) -> torch.Tensor:
    invalid = labels.ne(IGNORE_INDEX) & (labels.lt(0) | labels.ge(classes))
    if bool(invalid.any()):
        raise ValueError("sparse labels are outside the active class space")
    value = torch.zeros((labels.shape[0], classes, *labels.shape[1:]), device=labels.device)
    for label in range(classes):
        value[:, label] = labels.eq(label)
    return value


def pce_loss(probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log(probabilities + 1e-12)).sum() / target.sum().clamp_min(1)


def sparse_pce_loss(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """PCE on a sparse integer label map, ignoring unlabeled pixels."""
    known = labels.ne(IGNORE_INDEX)
    if not bool(known.any()):
        return probabilities.sum() * 0.0
    gathered = probabilities.gather(1, labels.clamp_min(0).unsqueeze(1)).squeeze(1)
    return -gathered[known].clamp_min(1e-12).log().mean()


def zs_forward(model: nn.Module, image: torch.Tensor, task_id: int | None) -> dict[str, torch.Tensor]:
    probabilities = model(image, task_id)
    if probabilities.shape[-2:] != image.shape[-2:]:
        probabilities = F.interpolate(probabilities, size=image.shape[-2:], mode="bilinear", align_corners=False)
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return {"pred_masks": probabilities}


def derpp_replay_losses(
    model: nn.Module,
    replay: tuple[torch.Tensor, ...],
    scenario: str,
    args,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sparse PCE and ZS global consistency on DER++ replay samples."""
    examples, _, labels, task_ids, class_counts = replay
    pce_terms = []
    global_terms = []
    for task_value in torch.unique(task_ids, sorted=True).tolist():
        selection = task_ids.eq(int(task_value))
        classes = int(class_counts[selection][0].item())
        target = native_target(labels[selection], classes)
        task_id = int(task_value) if scenario in {"class", "organ"} else None
        outputs, global_loss, _, _ = zs_cutout_invariance(
            model, examples[selection], target, task_id, args, device,
        )
        pce_terms.append(pce_loss(outputs["pred_masks"], target))
        global_terms.append(global_loss)
    return torch.stack(pce_terms).mean(), torch.stack(global_terms).mean()


def logits_forward(model: nn.Module, image: torch.Tensor, task_id: int | None) -> torch.Tensor:
    logits = model.forward_logits(image, task_id)
    if logits.shape[-2:] != image.shape[-2:]:
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def zs_cutout_invariance(
    model: nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    task_id: int | None,
    args,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, bool]:
    use_adversarial = False
    working_image = image
    if args.zs_adversarial_perturbation:
        use_adversarial = bool(np.random.binomial(1, 0.1) or np.random.binomial(1, 0.1))
        if use_adversarial:
            working_image = image + torch.zeros_like(image).uniform_(10.0 / 255.0, 10.0 / 255.0)
    unary_image = working_image.detach().requires_grad_(True)
    unary = torch.sqrt(torch.mean(torch.autograd.grad(
        pce_loss(zs_forward(model, unary_image, task_id)["pred_masks"], target), unary_image,
    )[0].square(), dim=1))
    mix_args = SimpleNamespace(
        mixup_alpha=0.5,
        in_batch=False,
        mean=torch.tensor([0.5], device=device).reshape(1, 1, 1, 1),
        std=torch.tensor([0.5], device=device).reshape(1, 1, 1, 1),
        box=False,
        graph=True,
        beta=1.2,
        gamma=0.5,
        eta=0.2,
        neigh_size=4,
        # GraphCut discretizes the mixing mask, not the segmentation classes.
        # The native implementation defines priors only for 2, 3, or 4 levels.
        n_labels=min(4, target.shape[1]),
        transport=False,
        t_eps=0.8,
        t_size=4,
        device=str(device),
    )
    outputs = zs_forward(model, working_image, task_id)
    mixed_image, mixed_target, indices, mask = mixup_process(
        working_image, target, args=mix_args, grad=unary,
    )
    cut_image, cut_target, cut_mask = Cutout(mixed_image, mixed_target, device)
    cut_image, cut_target, angles = rotate_invariant(cut_image, cut_target)
    cut_outputs = zs_forward(model, cut_image, task_id)
    _, rotated_outputs, cut_target = rotate_back(
        cut_image, cut_outputs["pred_masks"], cut_target, angles,
    )
    cut_probability = rotated_outputs["pred_masks"]
    shuffled = outputs["pred_masks"][torch.as_tensor(indices, device=device)]
    mixed_output = (outputs["pred_masks"] * mask + shuffled * (1 - mask)) * cut_mask
    invariant = 1 - F.cosine_similarity(cut_probability, mixed_output, dim=1).mean()
    annotated_cut = cut_target.sum(dim=1, keepdim=True)
    gd = -(cut_target * torch.log(cut_probability + 1e-12)).sum(dim=1, keepdim=True)
    return outputs, invariant, (gd * annotated_cut).mean(), use_adversarial


def zs_em_mixture_ratios(probabilities: torch.Tensor, target: torch.Tensor) -> dict[int, float]:
    annotated = target.sum(dim=1).bool()
    active = [label for label in range(target.shape[1]) if bool(target[:, label].any())]
    if not active:
        return {label: 0.0 for label in range(target.shape[1])}
    evidence = []
    for label in active:
        p = probabilities[:, label][annotated].detach().clamp_min(1e-12)
        g = target[:, label][annotated].detach().mean().clamp_min(1e-12)
        evidence.append((label, p, g))
    ratios = torch.ones(len(evidence), device=probabilities.device, dtype=probabilities.dtype)
    for _ in range(100):
        numerators = [ratio * p / g for ratio, (_, p, g) in zip(ratios, evidence)]
        denominator = torch.stack(numerators).sum(dim=0).clamp_min(1e-12)
        updated = torch.stack([(value / denominator).mean() for value in numerators])
        if torch.max(torch.abs(updated - ratios)).item() <= 1e-3:
            ratios = updated
            break
        ratios = updated
    result = {label: 0.0 for label in range(target.shape[1])}
    result.update({label: float(value) for (label, _, _), value in zip(evidence, ratios)})
    return result


def zs_spatial_prior_loss(
    probabilities: torch.Tensor,
    image: torch.Tensor,
    target: torch.Tensor,
    ratios: dict[int, float],
) -> tuple[torch.Tensor, float]:
    spatial = ModelWeightGatedCRF()(
        probabilities,
        [{"weight": 1, "xy": 6, "rgb": 0.1}],
        8,
        image,
        image.shape[-2],
        image.shape[-1],
    )
    unannotated = target.sum(dim=1, keepdim=True).eq(0)
    pseudo_negative = torch.zeros_like(probabilities)
    for label, ratio in ratios.items():
        if ratio <= 0:
            continue
        candidates = spatial[:, label][unannotated[:, 0]]
        if candidates.numel() == 0:
            continue
        rank = max(int(candidates.numel() * (1.0 - min(ratio, 1.0))) - 1, 0)
        threshold = torch.sort(candidates.flatten()).values[rank]
        pseudo_negative[:, label][spatial[:, label] < threshold] = 1
    pseudo_negative *= unannotated
    any_negative = pseudo_negative.sum(dim=1).clamp(max=1.0)
    allowed = (probabilities * (1.0 - pseudo_negative)).sum(dim=1)
    loss = -(any_negative * torch.log(allowed + 1e-12)).mean()
    return loss, float(pseudo_negative.mean().detach())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    ends: np.ndarray,
    device: torch.device,
    task_id: int | None,
    classes: tuple[int, ...],
) -> dict:
    was_training = model.training
    model.eval()
    predictions, targets = [], []
    for image, label in loader:
        predictions.append(zs_forward(model, image.to(device), task_id)["pred_masks"].argmax(1).cpu().numpy())
        targets.append(label.numpy())
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    starts = [0] + [int(value) + 1 for value in ends[:-1]]
    stops = [int(value) + 1 for value in ends]
    per_patient = []
    for start, stop in zip(starts, stops):
        per_class = []
        for label in classes:
            predicted = prediction[start:stop] == label
            expected = target[start:stop] == label
            per_class.append(float((2 * np.logical_and(predicted, expected).sum() + 1e-5) /
                                   (predicted.sum() + expected.sum() + 1e-5)))
        per_patient.append(per_class)
    values = np.asarray(per_patient, dtype=float)
    result = {
        "benchmark_mean": float(values.mean()),
        "per_class": values.mean(axis=0).tolist(),
        "per_patient": values.tolist(),
        "prediction_fg_fraction": float((prediction > 0).mean()),
    }
    if was_training:
        model.train()
    return result


def _worker_init(seed: int, worker_id: int) -> None:
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=partial(_worker_init, seed),
    )


def _evaluate_task(model, scenario: str, task: Task, stage: int, root: Path, split: str,
                   batch_size: int, device: torch.device) -> dict:
    dataset = H5Slices(root / task.folder / task.filename, split, label_shift=task.label_shift if scenario == "class" else 0)
    task_id = stage if scenario == "organ" else None
    result = evaluate(model, _loader(dataset, batch_size, False, 0, 0), dataset.ends, device, task_id, task.classes)
    dataset.close()
    return result


def _evaluate_joint(model, tasks: tuple[Task, ...], root: Path, split: str,
                    batch_size: int, device: torch.device) -> dict:
    per_task = {
        task.code: _evaluate_task(model, "domain", task, stage, root, split, batch_size, device)
        for stage, task in enumerate(tasks)
    }
    return {
        "benchmark_mean": float(np.mean([score["benchmark_mean"] for score in per_task.values()])),
        "per_task": per_task,
    }


def _write_matrix(path: Path, matrix: np.ndarray, tasks: tuple[Task, ...],
                  row_labels: tuple[str, ...] | None = None) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stage", *[task.code for task in tasks]])
        for index, row in enumerate(matrix):
            label = index + 1 if row_labels is None else row_labels[index]
            writer.writerow([label, *["" if np.isnan(value) else f"{value:.10f}" for value in row]])


def domain_matrix_metrics(
    matrix: np.ndarray,
    random_scores: list[float],
    independent_scores: list[float] | None = None,
) -> dict:
    value = np.asarray(matrix, dtype=np.float64)
    completed = np.flatnonzero(np.isfinite(np.diag(value)))
    if not len(completed):
        raise ValueError("empty Domain-CL performance matrix")
    last = int(completed[-1])
    diagonal = np.diag(value)[:last + 1]
    final = value[last]
    if last > 0 and np.any(diagonal[:last] <= 0):
        raise ValueError("Domain BWTR requires positive acquisition Dice")
    result = {
        "A-Dice": float(np.mean(final)) if last == len(value) - 1 else None,
        "BWTR": 0.0 if last == 0 else float(np.mean(
            (final[:last] - diagonal[:last]) / diagonal[:last]
        )),
    }
    random = np.asarray(random_scores, dtype=np.float64)
    if random.shape != (len(value),):
        raise ValueError("Domain E-FWT random baseline mismatch")
    forward = [
        value[stage, future] - random[future]
        for stage in range(last + 1)
        for future in range(stage + 1, len(value))
        if np.isfinite(value[stage, future])
    ]
    result["E-FWT"] = None if not forward else float(np.mean(forward))
    if independent_scores is None or last == 0:
        result["RMA"] = None
    else:
        reference = np.asarray(independent_scores, dtype=np.float64)
        if reference.shape != (len(value),) or np.any(reference[1:last + 1] <= 0):
            raise ValueError("Domain RMA independent reference mismatch")
        result["RMA"] = float(np.mean(diagonal[1:] / reference[1:last + 1]))
    return result


def _run_independent_domain_references(args, tasks: tuple[Task, ...], device: torch.device) -> None:
    if args.max_task is not None:
        raise ValueError("independent references require the complete A-to-F task list")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "scenario": "domain",
        "mode": "independent_pce_references",
        "main_entry": "main.py",
        "backbone": "ZScribbleSeg_UNet",
        "seed": args.seed,
        "epochs_per_task": args.epochs_per_task,
        "task_order": [task.code for task in tasks],
        "history_images": False,
        "replay": False,
        "data_root": "<external_data>",
        "sparse_root": "<external_data>",
        "status": "running",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    scores, records = [], []
    train_log = args.output / "train.jsonl"
    for index, task in enumerate(tasks):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        model = DomainModel().to(device)
        train = H5Slices(
            args.data_root / task.folder / task.filename,
            "train",
            _sparse_path(args.sparse_root, "domain", task, args.seed),
            augment=True,
        )
        val = H5Slices(args.data_root / task.folder / task.filename, "val")
        train_loader = _loader(train, args.batch_size, True, args.workers, args.seed)
        val_loader = _loader(val, args.batch_size, False, 0, args.seed)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4,
        )
        batches_per_epoch = len(train_loader)
        max_iterations = batches_per_epoch * args.epochs_per_task
        iteration = 0
        best = {"benchmark_mean": -1.0, "epoch": None, "iteration": None}
        best_path = args.output / f"s{index + 1:02d}_best.pt"
        with train_log.open("a") as stream:
            for epoch in range(args.epochs_per_task):
                model.train()
                losses = []
                for image, label in train_loader:
                    image, label = image.to(device), label.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    probability = zs_forward(model, image, None)["pred_masks"]
                    loss = pce_loss(probability, native_target(label, 2))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite independent-reference loss")
                    loss.backward()
                    optimizer.step()
                    iteration += 1
                    learning_rate = args.lr * max(0.0, 1.0 - iteration / max_iterations) ** 0.9
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    losses.append(float(loss.detach()))
                    if iteration % args.validate_every == 0:
                        validation = evaluate(model, val_loader, val.ends, device, None, task.classes)
                        if validation["benchmark_mean"] > best["benchmark_mean"]:
                            best = {**validation, "epoch": epoch, "iteration": iteration}
                            torch.save(model.state_dict(), best_path)
                stream.write(json.dumps({
                    "task_index": index,
                    "task": task.code,
                    "epoch": epoch,
                    "iteration": iteration,
                    "loss": float(np.mean(losses)),
                }, sort_keys=True) + "\n")
                stream.flush()
        validation = evaluate(model, val_loader, val.ends, device, None, task.classes)
        if validation["benchmark_mean"] > best["benchmark_mean"]:
            best = {**validation, "epoch": args.epochs_per_task - 1, "iteration": iteration}
            torch.save(model.state_dict(), best_path)
        model.load_state_dict(torch.load(best_path, map_location=device))
        torch.save(model.state_dict(), args.output / f"s{index + 1:02d}.pt")
        test = _evaluate_task(model, "domain", task, index, args.data_root, "test", args.batch_size, device)
        scores.append(test["benchmark_mean"])
        records.append({"task": task.code, "score": test["benchmark_mean"], "best_validation": best})
        (args.output / "independent_scores.json").write_text(json.dumps({
            "scenario": "domain",
            "seed": args.seed,
            "scores": scores,
            "records": records,
            "complete": len(scores) == len(tasks),
        }, indent=2, sort_keys=True) + "\n")
        train.close()
        val.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest["status"] = "complete"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"scores": scores, "records": records}, indent=2, sort_keys=True))


METHODS = {
    "class": (
        "pce-sequential", "zs-sequential", "pce-ewc", "pce-gpm", "pce-der",
        "zs-mib", "zs-gpm", "zs-der", "zs-derpp", "zs-derpp-mib",
    ),
    "organ": (
        "pce-sequential", "zs-sequential", "pce-ewc", "zs-ewc", "pce-gpm", "zs-gpm",
        "pce-der", "zs-der", "zs-derpp",
    ),
    "domain": (
        "pce-sequential", "zs-sequential", "pce-ewc", "zs-ewc", "pce-gpm", "zs-gpm",
        "pce-der", "zs-der", "zs-derpp", "zs-joint",
    ),
}


def _build_model(scenario: str) -> ClassModel | OrganModel | DomainModel:
    if scenario == "class":
        return ClassModel()
    if scenario == "organ":
        return OrganModel()
    return DomainModel()


def _sparse_path(root: Path, scenario: str, task: Task, seed: int) -> Path:
    base = root / scenario if scenario in {"class", "organ"} else root
    return base / f"{task.code}_v2_s2_seed{seed}.npz"


def main(project_scenario: str) -> None:
    if project_scenario not in TASKS:
        raise ValueError(f"unknown scenario: {project_scenario}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-per-task", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-task", type=int)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--method", choices=METHODS[project_scenario], default="pce-sequential")
    parser.add_argument("--pce-loss-weight", type=float, default=1.0)
    parser.add_argument("--zs-global-weight", type=float)
    parser.add_argument("--zs-gd-loss", action="store_true")
    parser.add_argument("--zs-adversarial-perturbation", action="store_true")
    parser.add_argument("--zs-spatial-loss-weight", type=float, default=0.0)
    parser.add_argument("--zs-spatial-warmup-epochs", type=int, default=60)
    parser.add_argument("--ewc-lambda", type=float, default=1.0)
    parser.add_argument("--ewc-gamma", type=float, default=0.1)
    parser.add_argument("--fisher-batches", type=int, default=50)
    parser.add_argument("--gpm-threshold", type=float, default=0.97)
    parser.add_argument("--gpm-threshold-step", type=float, default=0.001)
    parser.add_argument("--gpm-examples", type=int, default=16)
    parser.add_argument("--gpm-max-patches-per-layer", type=int, default=4096)
    parser.add_argument("--gpm-max-matrix-elements", type=int, default=4_000_000)
    parser.add_argument("--der-buffer-size", type=int, default=32)
    parser.add_argument("--der-minibatch-size", type=int, default=8)
    parser.add_argument("--der-alpha", type=float, default=0.5)
    parser.add_argument("--der-beta", type=float, default=0.5)
    parser.add_argument("--mib-kd-weight", type=float, default=10.0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--independent-reference", action="store_true")
    parser.add_argument("--independent-scores", type=Path)
    args = parser.parse_args()
    use_zs = args.method.startswith("zs-")
    use_ewc = args.method.endswith("-ewc")
    use_gpm = args.method.endswith("-gpm")
    use_der = args.method.endswith("-der")
    use_derpp = args.method in {"zs-derpp", "zs-derpp-mib"}
    use_mib = args.method in {"zs-mib", "zs-derpp-mib"}
    use_joint = args.method == "zs-joint"
    if args.zs_global_weight is None:
        args.zs_global_weight = 1.0 if use_zs else 0.0
    if not use_zs and (
        args.zs_global_weight != 0.0
        or args.zs_gd_loss
        or args.zs_adversarial_perturbation
        or args.zs_spatial_loss_weight != 0.0
    ):
        parser.error("PCE methods cannot enable ZS-only components")
    if args.zs_adversarial_perturbation and not (args.zs_global_weight or args.zs_gd_loss):
        parser.error("adversarial perturbation requires global consistency or gd loss")
    if args.epochs_per_task < 1 or args.batch_size < 1 or args.validate_every < 1:
        parser.error("epochs, batch size, and validation interval must be positive")
    if args.fisher_batches < 1 or args.ewc_lambda < 0 or not 0 <= args.ewc_gamma <= 1:
        parser.error("invalid EWC controls")
    if not 0 < args.gpm_threshold < 1:
        parser.error("--gpm-threshold must be between zero and one")
    if args.gpm_threshold_step < 0:
        parser.error("--gpm-threshold-step must be non-negative")
    if min(
        args.gpm_examples,
        args.gpm_max_patches_per_layer,
        args.gpm_max_matrix_elements,
    ) < 1:
        parser.error("GPM sampling controls must be positive")
    if min(args.der_buffer_size, args.der_minibatch_size) < 1 or min(args.der_alpha, args.der_beta) < 0:
        parser.error("invalid DER controls")
    if args.mib_kd_weight < 0:
        parser.error("MiB KD weight must be non-negative")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("--max-train-batches must be positive")
    if args.independent_reference and (
        project_scenario != "domain" or args.method != "pce-sequential"
    ):
        parser.error("independent references are Domain-only PCE from-scratch runs")
    tasks = TASKS[project_scenario]
    if use_joint and args.max_task is not None:
        parser.error("zs-joint always trains on all Domain-CL tasks")
    if args.max_task is not None and not 1 <= args.max_task <= len(tasks):
        parser.error("--max-task is one-based and outside the task sequence")
    last_stage = len(tasks) - 1 if args.max_task is None else args.max_task - 1
    for task in tasks[:last_stage + 1]:
        data_path = args.data_root / task.folder / task.filename
        sparse_path = _sparse_path(args.sparse_root, project_scenario, task, args.seed)
        if not data_path.is_file() or not sparse_path.is_file():
            raise FileNotFoundError(f"missing task input for {task.code}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if args.independent_reference:
        _run_independent_domain_references(args, tasks, device)
        return
    model = _build_model(project_scenario)
    model.to(device)
    ewc = OnlineEWC(args.ewc_lambda, args.ewc_gamma) if use_ewc else None
    gpm = (
        GradientProjectionMemory(
            threshold=args.gpm_threshold,
            threshold_step=args.gpm_threshold_step,
            examples=args.gpm_examples,
            max_patches_per_layer=args.gpm_max_patches_per_layer,
            max_matrix_elements=args.gpm_max_matrix_elements,
        )
        if use_gpm
        else None
    )
    der = (
        DarkExperienceReplay(
            buffer_size=args.der_buffer_size,
            minibatch_size=args.der_minibatch_size,
            alpha=args.der_alpha,
        )
        if use_der
        else None
    )
    derpp = (
        DarkExperienceReplayPlus(
            buffer_size=args.der_buffer_size,
            minibatch_size=args.der_minibatch_size,
            alpha=args.der_alpha,
            beta=args.der_beta,
        )
        if use_derpp
        else None
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "scenario": project_scenario,
        "method": args.method,
        "main_entry": "main.py",
        "backbone": "ZScribbleSeg_UNet",
        "seed": args.seed,
        "epochs_per_task": args.epochs_per_task,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "workers": args.workers,
        "validate_every": args.validate_every,
        "task_count": last_stage + 1,
        "task_order": [task.code for task in tasks[:last_stage + 1]],
        "training_mode": "joint" if use_joint else "continual",
        "test_for_selection": False,
        "history_images": use_der or use_derpp,
        "replay": use_der or use_derpp,
        "ignore_index": IGNORE_INDEX,
        "data_root": "<external_data>",
        "sparse_root": "<external_data>",
        "pce_loss_weight": args.pce_loss_weight,
        "zs_global_weight": args.zs_global_weight,
        "zs_gd_loss": args.zs_gd_loss,
        "zs_adversarial_perturbation": args.zs_adversarial_perturbation,
        "zs_spatial_loss_weight": args.zs_spatial_loss_weight,
        "zs_spatial_warmup_epochs": args.zs_spatial_warmup_epochs,
        "ewc_lambda": args.ewc_lambda if use_ewc else None,
        "ewc_gamma": args.ewc_gamma if use_ewc else None,
        "fisher_batches": args.fisher_batches if use_ewc else None,
        "gpm_threshold": args.gpm_threshold if use_gpm else None,
        "gpm_threshold_step": args.gpm_threshold_step if use_gpm else None,
        "gpm_examples": args.gpm_examples if use_gpm else None,
        "gpm_max_patches_per_layer": args.gpm_max_patches_per_layer if use_gpm else None,
        "gpm_max_matrix_elements": args.gpm_max_matrix_elements if use_gpm else None,
        "der_buffer_size": args.der_buffer_size if (use_der or use_derpp) else None,
        "der_minibatch_size": args.der_minibatch_size if (use_der or use_derpp) else None,
        "der_alpha": args.der_alpha if (use_der or use_derpp) else None,
        "der_target": "backbone_features" if use_der else None,
        "der_beta": args.der_beta if use_derpp else None,
        "derpp_target": "backbone_features_plus_sparse_pce_global" if use_derpp else None,
        "mib_kd_weight": args.mib_kd_weight if use_mib else None,
        "max_train_batches": args.max_train_batches,
        "independent_scores": None if args.independent_scores is None else args.independent_scores.name,
        "status": "running",
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    matrix = np.full((1 if use_joint else len(tasks), len(tasks)), np.nan)
    stage_rows = []
    fisher_rows = []
    gpm_rows = []
    train_log = args.output / "train.jsonl"
    random_scores = None
    if project_scenario == "domain" and not use_joint:
        random_scores = [
            _evaluate_task(
                model, project_scenario, task, index, args.data_root,
                "test", args.batch_size, device,
            )["benchmark_mean"]
            for index, task in enumerate(tasks)
        ]
        (args.output / "random_scores.json").write_text(json.dumps({
            "seed": args.seed,
            "scores": random_scores,
            "source": "same-seed untrained model before task A",
        }, indent=2, sort_keys=True) + "\n")

    stage_plan = ((0, tasks[-1]),) if use_joint else enumerate(tasks[:last_stage + 1])
    for stage, task in stage_plan:
        teacher = None
        old_class_count = None
        if use_mib and stage > 0:
            teacher = copy.deepcopy(model).to(device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            old_class_count = model.output_channels(stage - 1)
        model.activate_stage(stage)
        model.train()
        training_tasks = tasks if use_joint else (task,)
        train_parts = [
            H5Slices(
                args.data_root / training_task.folder / training_task.filename,
                "train",
                _sparse_path(args.sparse_root, project_scenario, training_task, args.seed),
                augment=True,
            )
            for training_task in training_tasks
        ]
        train = ConcatDataset(train_parts) if use_joint else train_parts[0]
        val = None if use_joint else H5Slices(
            args.data_root / task.folder / task.filename,
            "val",
            label_shift=task.label_shift if project_scenario == "class" else 0,
        )
        train_loader = _loader(train, args.batch_size, True, args.workers, args.seed + stage)
        val_loader = None if val is None else _loader(val, args.batch_size, False, 0, args.seed)
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            momentum=0.9,
            # For GPM, weight decay is folded into gradients before projection.
            # Letting SGD add it afterward would move convolutional kernels back
            # into protected directions and violate the projection constraint.
            weight_decay=0.0 if use_gpm else 1e-4,
        )
        batches_per_epoch = len(train_loader)
        if args.max_train_batches is not None:
            batches_per_epoch = min(batches_per_epoch, args.max_train_batches)
        max_iterations = batches_per_epoch * args.epochs_per_task
        iteration = 0
        best = {"benchmark_mean": -1.0, "epoch": None, "iteration": None}
        best_path = args.output / f"s{stage + 1:02d}_best.pt"
        task_id = stage if project_scenario == "organ" else None
        classes = model.output_channels(stage)

        def validate_current() -> dict:
            if use_joint:
                return _evaluate_joint(model, tasks, args.data_root, "val", args.batch_size, device)
            return evaluate(model, val_loader, val.ends, device, task_id, task.classes)

        with train_log.open("a") as stream:
            for epoch in range(args.epochs_per_task):
                model.train()
                totals = {
                    "loss": [], "pce": [], "global": [], "gd": [], "spatial": [],
                    "ewc": [], "mib_kd": [], "gpm_gradient_ratio": [], "der": [],
                    "derpp_feature": [], "derpp_pce": [], "derpp_global": [],
                }
                adversarial_batches = 0
                ratios = None
                for batch_index, (image, label) in enumerate(train_loader):
                    if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                        break
                    image, label = image.to(device), label.to(device)
                    target = native_target(label, classes)
                    optimizer.zero_grad(set_to_none=True)
                    use_spatial = args.zs_spatial_loss_weight > 0 and epoch > args.zs_spatial_warmup_epochs
                    if use_spatial:
                        model.eval()
                        with torch.no_grad():
                            ratios = zs_em_mixture_ratios(zs_forward(model, image, task_id)["pred_masks"], target)
                        model.train()
                    if use_zs and (args.zs_global_weight or args.zs_gd_loss):
                        outputs, global_loss, gd_loss, used_adversarial = zs_cutout_invariance(
                            model, image, target, task_id, args, device,
                        )
                    else:
                        outputs = zs_forward(model, image, task_id)
                        global_loss = gd_loss = image.new_zeros(())
                        used_adversarial = False
                    if use_mib and stage > 0:
                        student_logits = logits_forward(model, image, task_id)
                        partial_ce = mib_sparse_loss(student_logits, label, old_class_count)
                        with torch.no_grad():
                            teacher_logits = logits_forward(teacher, image, None)
                        mib_kd = mib_distillation_loss(student_logits, teacher_logits)
                    else:
                        partial_ce = pce_loss(outputs["pred_masks"], target)
                        mib_kd = image.new_zeros(())
                    loss = args.pce_loss_weight * partial_ce + args.zs_global_weight * global_loss
                    if use_mib:
                        loss = loss + args.mib_kd_weight * mib_kd
                    ewc_penalty = ewc.penalty(model) if use_ewc else image.new_zeros(())
                    loss = loss + ewc_penalty
                    der_penalty = der.penalty(model, device) if use_der else image.new_zeros(())
                    loss = loss + der_penalty
                    if use_derpp:
                        derpp_feature, replay = derpp.feature_penalty(model, device)
                        if replay is None:
                            derpp_pce = derpp_global = image.new_zeros(())
                        else:
                            derpp_pce, derpp_global = derpp_replay_losses(
                                model, replay, project_scenario, args, device,
                            )
                        loss = loss + derpp_feature + args.der_beta * (
                            derpp_pce + args.zs_global_weight * derpp_global
                        )
                    else:
                        derpp_feature = derpp_pce = derpp_global = image.new_zeros(())
                    if args.zs_gd_loss:
                        loss = loss + gd_loss
                    if use_spatial:
                        spatial_loss, spatial_fraction = zs_spatial_prior_loss(
                            outputs["pred_masks"], image, target, ratios,
                        )
                        loss = loss + args.zs_spatial_loss_weight * spatial_loss
                    else:
                        spatial_loss, spatial_fraction = image.new_zeros(()), 0.0
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training loss")
                    loss.backward()
                    if use_gpm:
                        with torch.no_grad():
                            for parameter in model.parameters():
                                if parameter.grad is not None:
                                    parameter.grad.add_(parameter, alpha=1e-4)
                    if use_gpm and stage > 0:
                        projection = gpm.project_gradients(model)
                        if projection["layers"] < 1:
                            raise RuntimeError("GPM did not project any convolutional gradients")
                        denominator = max(projection["gradient_norm_before"], 1e-12)
                        totals["gpm_gradient_ratio"].append(
                            projection["gradient_norm_after"] / denominator
                        )
                    optimizer.step()
                    if use_der:
                        with torch.no_grad():
                            der.add_data(image, model.backbone(image))
                    if use_derpp:
                        with torch.no_grad():
                            replay_task_ids = torch.full(
                                (image.shape[0],), stage, device=image.device, dtype=torch.int64,
                            )
                            derpp.add_data(
                                image,
                                model.backbone(image),
                                label,
                                replay_task_ids,
                                classes,
                            )
                    iteration += 1
                    learning_rate = args.lr * max(0.0, 1.0 - iteration / max_iterations) ** 0.9
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    totals["loss"].append(float(loss.detach()))
                    totals["pce"].append(float(partial_ce.detach()))
                    totals["global"].append(float(global_loss.detach()))
                    totals["gd"].append(float(gd_loss.detach()))
                    totals["spatial"].append(float(spatial_loss.detach()))
                    totals["ewc"].append(float(ewc_penalty.detach()))
                    totals["mib_kd"].append(float(mib_kd.detach()))
                    totals["der"].append(float(der_penalty.detach()))
                    totals["derpp_feature"].append(float(derpp_feature.detach()))
                    totals["derpp_pce"].append(float(derpp_pce.detach()))
                    totals["derpp_global"].append(float(derpp_global.detach()))
                    adversarial_batches += int(used_adversarial)
                    if iteration % args.validate_every == 0:
                        validation = validate_current()
                        stream.write(json.dumps({
                            "stage": stage,
                            "epoch": epoch,
                            "iteration": iteration,
                            "validation": validation,
                        }, sort_keys=True) + "\n")
                        stream.flush()
                        if validation["benchmark_mean"] > best["benchmark_mean"]:
                            best = {**validation, "epoch": epoch, "iteration": iteration}
                            torch.save(model.state_dict(), best_path)
                row = {
                    "stage": stage,
                    "task": "joint" if use_joint else task.code,
                    "epoch": epoch,
                    "iteration": iteration,
                    "loss": float(np.mean(totals["loss"])),
                    "pce_loss": float(np.mean(totals["pce"])),
                    "zs_global_loss": float(np.mean(totals["global"])),
                    "zs_gd_loss": float(np.mean(totals["gd"])),
                    "zs_spatial_loss": float(np.mean(totals["spatial"])),
                    "ewc_penalty": float(np.mean(totals["ewc"])),
                    "mib_kd_loss": float(np.mean(totals["mib_kd"])),
                    "gpm_gradient_ratio": (
                        None
                        if not totals["gpm_gradient_ratio"]
                        else float(np.mean(totals["gpm_gradient_ratio"]))
                    ),
                    "der_penalty": float(np.mean(totals["der"])),
                    "der_stored_examples": None if der is None else len(der),
                    "derpp_feature_loss": float(np.mean(totals["derpp_feature"])),
                    "derpp_pce_loss": float(np.mean(totals["derpp_pce"])),
                    "derpp_global_loss": float(np.mean(totals["derpp_global"])),
                    "derpp_stored_examples": None if derpp is None else len(derpp),
                    "zs_em_mixture_ratios": ratios,
                    "zs_spatial_pseudo_negative_fraction": spatial_fraction,
                    "adversarial_batches": adversarial_batches,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
        final_validation = validate_current()
        if final_validation["benchmark_mean"] > best["benchmark_mean"]:
            best = {**final_validation, "epoch": args.epochs_per_task - 1, "iteration": iteration}
            torch.save(model.state_dict(), best_path)
        model.load_state_dict(torch.load(best_path, map_location=device))
        torch.save(model.state_dict(), args.output / f"s{stage + 1:02d}.pt")
        fisher_summary = None
        if use_ewc:
            fisher_data = H5Slices(
                args.data_root / task.folder / task.filename,
                "train",
                _sparse_path(args.sparse_root, project_scenario, task, args.seed),
                augment=False,
            )
            fisher_loader = _loader(fisher_data, args.batch_size, False, 0, args.seed + stage)

            def fisher_loss_fn(fisher_image: torch.Tensor, fisher_label: torch.Tensor) -> torch.Tensor:
                fisher_target = native_target(fisher_label, classes)
                probability = zs_forward(model, fisher_image, task_id)["pred_masks"]
                return pce_loss(probability, fisher_target)

            fisher, fisher_summary = estimate_sparse_fisher(
                model, fisher_loader, device, fisher_loss_fn, args.fisher_batches,
            )
            ewc.consolidate(model, fisher)
            fisher_summary = {"stage": stage, "task": task.code, **fisher_summary}
            fisher_rows.append(fisher_summary)
            (args.output / "fisher.json").write_text(
                json.dumps(fisher_rows, indent=2, sort_keys=True) + "\n"
            )
            fisher_data.close()
        gpm_summary = None
        if use_gpm:
            representation_data = H5Slices(
                args.data_root / task.folder / task.filename,
                "train",
                _sparse_path(args.sparse_root, project_scenario, task, args.seed),
                augment=False,
            )
            representation_loader = _loader(
                representation_data,
                min(args.batch_size, args.gpm_examples),
                True,
                0,
                args.seed + stage,
            )
            gpm_summary = gpm.update_from_loader(
                model,
                representation_loader,
                device,
                stage,
                task_id,
                args.seed + stage,
            )
            gpm_summary["task"] = task.code
            gpm_rows.append(gpm_summary)
            (args.output / "gpm.json").write_text(
                json.dumps(gpm_rows, indent=2, sort_keys=True) + "\n"
            )
            representation_data.close()
        torch.save(
            {
                "model": model.state_dict(),
                "continual": (
                    ewc.state_dict()
                    if ewc is not None
                    else gpm.state_dict()
                    if gpm is not None
                    else der.state_dict()
                    if der is not None
                    else derpp.state_dict()
                    if derpp is not None
                    else None
                ),
                "stage": stage,
                "method": args.method,
            },
            args.output / f"s{stage + 1:02d}_state.pt",
        )
        seen_validation = {} if use_joint else {
            seen_task.code: _evaluate_task(
                model, project_scenario, seen_task, seen_stage, args.data_root,
                "val", args.batch_size, device,
            )["benchmark_mean"]
            for seen_stage, seen_task in enumerate(tasks[:stage + 1])
        }
        evaluated = {}
        evaluation_tasks = tasks if project_scenario == "domain" else tasks[:stage + 1]
        for evaluated_stage, evaluated_task in enumerate(evaluation_tasks):
            score = _evaluate_task(
                model,
                project_scenario,
                evaluated_task,
                evaluated_stage,
                args.data_root,
                "test",
                args.batch_size,
                device,
            )
            matrix[stage, evaluated_stage] = score["benchmark_mean"]
            evaluated[evaluated_task.code] = score
        stage_row = {
            "stage": stage,
            "task": "joint" if use_joint else task.code,
            "train_samples": len(train),
            "best_validation": best,
            "seen_validation": seen_validation,
            "seen_validation_mean": (
                None if use_joint else float(np.mean(tuple(seen_validation.values())))
            ),
            "evaluated": evaluated,
            "fisher": fisher_summary,
            "gpm": gpm_summary,
            "der": None if der is None else der.summary(),
            "derpp": None if derpp is None else derpp.summary(),
        }
        stage_rows.append(stage_row)
        (args.output / "stages.json").write_text(json.dumps(stage_rows, indent=2, sort_keys=True) + "\n")
        row_labels = ("joint",) if use_joint else None
        _write_matrix(args.output / "matrix.csv", matrix, tasks, row_labels)
        _write_matrix(args.output / "performance_matrix.csv", matrix, tasks, row_labels)
        for train_part in train_parts:
            train_part.close()
        if val is not None:
            val.close()

    serializable_matrix = [
        [None if np.isnan(value) else float(value) for value in row]
        for row in matrix
    ]
    final_matrix_row = 0 if use_joint else last_stage
    final_values = matrix[final_matrix_row] if use_joint else matrix[final_matrix_row, :last_stage + 1]
    summary = {
        "method": args.method,
        "completed_stages": 1 if use_joint else last_stage + 1,
        "joint_training": use_joint,
        "final_seen_mean": float(np.nanmean(final_values)),
        "final_seen_validation_mean": stage_rows[-1]["seen_validation_mean"],
        "matrix": serializable_matrix,
        "stage_rows": stage_rows,
        "history_images": use_der or use_derpp,
        "replay": use_der or use_derpp,
        "ewc_state_bytes": None if ewc is None else ewc.nbytes(),
        "gpm_state_bytes": None if gpm is None else gpm.nbytes(),
        "der_state_bytes": None if der is None else der.nbytes(),
        "der_buffer": None if der is None else der.summary(),
        "derpp_state_bytes": None if derpp is None else derpp.nbytes(),
        "derpp_buffer": None if derpp is None else derpp.summary(),
    }
    if project_scenario == "domain":
        if use_joint:
            summary.update({"A-Dice": summary["final_seen_mean"], "BWTR": None, "E-FWT": None, "RMA": None})
            summary["rma_reference"] = None
        else:
            independent = None
            if args.independent_scores is not None and args.independent_scores.is_file():
                reference_payload = json.loads(args.independent_scores.read_text())
                if reference_payload.get("complete") and len(reference_payload.get("scores", ())) == len(tasks):
                    independent = reference_payload["scores"]
            summary.update(domain_matrix_metrics(matrix, random_scores, independent))
            summary["rma_reference"] = (
                None if args.independent_scores is None else args.independent_scores.name
            )
    if project_scenario == "class" and last_stage == len(tasks) - 1:
        whole = H5Slices(args.data_root / "MMWHS" / "whole_heart_test.h5", "test")
        summary["whole_class_dice"] = evaluate(
            model,
            _loader(whole, args.batch_size, False, 0, args.seed),
            whole.ends,
            device,
            None,
            tuple(range(1, 8)),
        )
        whole.close()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
