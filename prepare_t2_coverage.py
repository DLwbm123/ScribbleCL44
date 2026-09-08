"""Create a nested T2 foreground-only coverage contrast; never edit source masks."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt


def expand(sparse, dense, fraction, seed=42):
    assert sparse.shape == dense.shape and 0 < fraction <= 1
    assert np.isin(sparse, [-100, 0, 1]).all()
    assert np.all(dense[sparse == 1] == 1) and np.all(dense[sparse == 0] == 0)
    result = sparse.copy()
    rng = np.random.default_rng(seed)
    for annotation, truth in zip(result, dense):
        foreground = annotation == 1
        desired = max(int(foreground.sum()), int(np.ceil(fraction * (truth == 1).sum())))
        needed = desired - int(foreground.sum())
        if not needed:
            continue
        if not foreground.any():
            raise ValueError("foreground slice has no original scribble to expand")
        candidates = np.flatnonzero((annotation == -100) & (truth == 1))
        distances = distance_transform_edt(~foreground).ravel()[candidates]
        order = np.lexsort((rng.random(len(candidates)), distances))
        annotation.ravel()[candidates[order[:needed]]] = 1
    assert np.array_equal(result == 0, sparse == 0)
    assert np.array_equal(result[sparse != -100], sparse[sparse != -100])
    assert np.all(dense[result == 1] == 1)
    return result


def stats(labels, dense):
    fg = int((labels == 1).sum())
    bg = int((labels == 0).sum())
    total_fg = int((dense == 1).sum())
    return dict(foreground_pixels=fg, background_pixels=bg, dense_foreground_pixels=total_fg,
                foreground_coverage=fg / total_fg, all_pixel_coverage=(fg + bg) / labels.size,
                train_slices=len(labels), total_pixels=labels.size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sparse-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    original = args.sparse_root / "organ"
    baseline = np.load(original / "T2_v2_s2_seed42.npz", allow_pickle=False)["annotations"]
    with h5py.File(args.data_root / "Task_incre/UCL.h5", "r") as h:
        dense = np.asarray(h["train_labels"]).transpose(2, 0, 1).astype(np.int64)
    expanded = expand(baseline, dense, .4)
    args.output.mkdir(parents=True, exist_ok=False)
    for variant, labels in [("fg20", baseline), ("fg40", expanded)]:
        folder = args.output / variant / "organ"
        folder.mkdir(parents=True)
        for task in ["T1", "T3", "T4"]:
            for suffix in ["npz", "json"]:
                name = f"{task}_v2_s2_seed42.{suffix}"
                (folder / name).symlink_to((original / name).resolve())
        np.savez_compressed(folder / "T2_v2_s2_seed42.npz", annotations=labels)
        metadata = dict(protocol="organ_T2_nested_foreground_20260908", variant=variant, seed=42,
                        source_protocol="v2_S2_area_scaled", background_unchanged=True,
                        original_labels_preserved=True, split_used="train",
                        expansion="nearest original foreground scribble within integer dense foreground; seeded distance ties",
                        stats=stats(labels, dense))
        (folder / "T2_v2_s2_seed42.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = dict(baseline=stats(baseline, dense), expanded=stats(expanded, dense),
                  added_foreground_pixels=int((expanded == 1).sum() - (baseline == 1).sum()),
                  background_unchanged=True, original_labels_preserved=True)
    (args.output / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
