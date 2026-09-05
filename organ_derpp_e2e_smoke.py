#!/usr/bin/env python3
"""Two-stage Organ-CL DER++ contract test through main.py --setting-run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

from runner_core import IGNORE_INDEX


def write_task(path: Path, sparse_path: Path, offset: float) -> None:
    train = np.zeros((256, 256, 4), dtype=np.float32)
    train[48:208, 48:208, :] = offset
    labels = np.zeros((256, 256, 2), dtype=np.int16)
    labels[96:160, 96:160, :] = 1
    sparse = np.full((4, 256, 256), IGNORE_INDEX, dtype=np.int16)
    sparse[:, 112:144, 112:144] = 1
    sparse[:, 80:96, 80:96] = 0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_images", data=train)
        handle.create_dataset("train_labels", data=np.zeros_like(train, dtype=np.int16))
        handle.create_dataset("val_images", data=np.repeat(train[:, :, :1], 2, axis=2))
        handle.create_dataset("val_labels", data=labels)
        handle.create_dataset("patient_info_val", data=np.asarray([1], dtype=np.int64))
    np.savez_compressed(sparse_path, annotations=sparse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="organ_derpp_smoke_", dir=args.work_root) as temporary:
        root = Path(temporary)
        data = root / "data" / "Task_incre"
        sparse = root / "sparse" / "organ"
        data.mkdir(parents=True)
        sparse.mkdir(parents=True)
        write_task(data / "UtahI.h5", sparse / "T1_v2_s2_seed42.npz", 0.25)
        write_task(data / "UCL.h5", sparse / "T2_v2_s2_seed42.npz", 0.75)
        output = root / "output"
        command = [
            sys.executable,
            "main.py",
            "--setting-run",
            "--data-root",
            str(root / "data"),
            "--sparse-root",
            str(root / "sparse"),
            "--output",
            str(output),
            "--device",
            args.device,
            "--seed",
            "42",
            "--epochs-per-task",
            "1",
            "--batch-size",
            "4",
            "--workers",
            "0",
            "--validate-every",
            "1",
            "--max-train-batches",
            "1",
            "--max-task",
            "2",
            "--method",
            "zs-derpp",
            "--numerical-debug",
            "--zs-global-weight",
            "0.1",
            "--der-buffer-size",
            "4",
            "--der-minibatch-size",
            "4",
        ]
        subprocess.run(command, check=True, cwd=Path(__file__).parent, env=os.environ.copy())
        subprocess.run(
            [
                sys.executable,
                "zs_audit_pipeline_output.py",
                "--output",
                str(output),
                "--method",
                "zs-derpp",
                "--scenario",
                "organ",
                "--expected-stages",
                "2",
            ],
            check=True,
            cwd=Path(__file__).parent,
            env=os.environ.copy(),
        )
        summary = json.loads((output / "summary.json").read_text())
        state = torch.load(output / "s02_state.pt", map_location="cpu")
        best = torch.load(output / "s02_best_state.pt", map_location="cpu")
        continual = state["continual"]
        test_hidden = all(value is None for row in summary["matrix"] for value in row)
        validation_seen = all(
            summary["validation_matrix"][1][index] is not None for index in range(2)
        )
        buffer_matches_best = (
            torch.equal(continual["examples"], best["derpp"]["examples"])
            and torch.equal(continual["task_ids"], best["derpp"]["task_ids"])
        )
        no_numerical_failure = not (output / "FIRST_NONFINITE.json").exists()
        passed = (
            summary["final_seen_mean"] is None
            and test_hidden
            and validation_seen
            and continual["coverage"]["replay_examples_by_task"]
            and buffer_matches_best
            and no_numerical_failure
        )
        result = {
            "status": "PASS" if passed else "FAIL",
            "test_hidden": test_hidden,
            "validation_seen": validation_seen,
            "buffer_matches_best": buffer_matches_best,
            "no_numerical_failure": no_numerical_failure,
            "coverage": continual["coverage"],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
