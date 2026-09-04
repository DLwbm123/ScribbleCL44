#!/usr/bin/env python3
"""Small deterministic checks for Organ-CL data and dispatcher invariants."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

import runner_core
from runner_core import H5Slices, IGNORE_INDEX


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        h5_path = root / "task.h5"
        sparse_path = root / "sparse.npz"
        image = np.arange(256 * 256, dtype=np.float32).reshape(256, 256, 1)
        sparse = np.full((1, 256, 256), IGNORE_INDEX, dtype=np.int16)
        sparse[0, 128, 128] = 1
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("train_images", data=image)
            handle.create_dataset("train_labels", data=np.zeros_like(image, dtype=np.int16))
        np.savez_compressed(sparse_path, annotations=sparse)
        dataset = H5Slices(h5_path, "train", sparse_path, augment=True, replay_source=True)
        with patch.object(runner_core.random, "random", side_effect=(0.0, 0.9)), patch.object(
            runner_core.np.random, "randint", return_value=17
        ):
            augmented_image, augmented_label, raw_image, raw_label = dataset[0]
        dataset.close()
    dispatcher = Path(__file__).with_name("runner.py").read_text()
    runner = Path(__file__).with_name("runner_core.py").read_text()
    passed = (
        torch.equal(raw_image, torch.from_numpy(image.transpose(2, 0, 1)))
        and torch.equal(raw_label, torch.from_numpy(sparse[0]).long())
        and int(augmented_label[0, 0]) == IGNORE_INDEX
        and not torch.equal(augmented_image, raw_image)
        and 'run("organ")' in dispatcher
        and "--test-evaluation" in runner
        and "replay_source=use_derpp" in runner
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "rotation_border_label": int(augmented_label[0, 0]),
        "raw_replay_source": bool(torch.equal(raw_label, torch.from_numpy(sparse[0]).long())),
        "organ_dispatcher": 'run("organ")' in dispatcher,
        "test_evaluation_opt_in": "--test-evaluation" in runner,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
