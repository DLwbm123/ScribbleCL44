"""Recovered Organ v2 area-scaled WSL recipe from the project thesis experiment snapshot.

Only used generation functions retained; original pixel construction is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

IGNORE_INDEX = -100


@dataclass
class AnnotationStats:
    foreground_pixels: int
    background_pixels: int
    unknown_pixels: int
    dense_foreground_pixels: int
    zero_foreground_slices: int
    minimum_valid_pixels: int
    maximum_valid_pixels: int


def _background(mask: np.ndarray, seed: int, index: int) -> np.ndarray:
    safe = ndi.distance_transform_edt(~mask) >= 5
    h, w = mask.shape; out = np.zeros_like(mask, dtype=bool)
    rng = np.random.default_rng(seed * 1_000_003 + index); span = 28
    x = int(rng.integers(8, max(9, w - span - 8))); y = int(rng.integers(8, max(9, h - span - 8)))
    out[4:7, x:x+span] = True; out[h-7:h-4, w-x-span:w-x] = True
    out[y:y+span, 4:7] = True; out[h-y-span:h-y, w-7:w-4] = True
    return out & safe


def _expand(seed_mask: np.ndarray, allowed: np.ndarray, multiplier: float) -> np.ndarray:
    """Deterministically grow a stroke only within its semantically valid area."""
    if multiplier <= 1:
        return seed_mask & allowed
    target = min(int(allowed.sum()), int(round(seed_mask.sum() * multiplier)))
    if target <= int(seed_mask.sum()):
        return seed_mask & allowed
    distance = ndi.distance_transform_edt(~seed_mask)
    candidates = np.flatnonzero(allowed)
    order = np.lexsort((candidates, distance.ravel()[candidates]))
    chosen = candidates[order[:target]]
    output = np.zeros(allowed.shape, dtype=bool, order="C")
    # ``allowed`` originates from transposed H5 slices and may be
    # non-contiguous.  Both the target and its flat indexing must be C-order;
    # otherwise a later boolean assignment can move background candidates.
    output.flat[chosen] = True
    return output


def _grow_to_count(seed_mask: np.ndarray, allowed: np.ndarray, target: int) -> np.ndarray:
    """Grow a mask to an absolute area without leaving ``allowed``."""
    seed_mask = np.asarray(seed_mask & allowed, dtype=bool)
    target = min(int(allowed.sum()), max(0, int(target)))
    if target == 0:
        return np.zeros(allowed.shape, dtype=bool)
    if int(seed_mask.sum()) >= target:
        # The WSL seed is normally much thinner than the requested area.  If a
        # tiny target is requested, retain a deterministic spatial subset.
        choices = np.flatnonzero(np.ascontiguousarray(seed_mask).reshape(-1))
        output = np.zeros(allowed.shape, dtype=bool)
        output.flat[choices[np.linspace(0, len(choices) - 1, target, dtype=int)]] = True
        return output
    distance = ndi.distance_transform_edt(~seed_mask)
    candidates = np.flatnonzero(np.ascontiguousarray(allowed).reshape(-1))
    order = np.lexsort((candidates, np.ascontiguousarray(distance).reshape(-1)[candidates]))
    output = np.zeros(allowed.shape, dtype=bool)
    output.flat[candidates[order[:target]]] = True
    return output


def _wsl_skeleton(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Erode then skeletonize, following the WSL4MIS 2-D scribble recipe."""
    if int(mask.sum()) == 0:
        return np.zeros(mask.shape, dtype=bool)
    work = mask
    if int(mask.sum()) > 900:
        iterations = int(rng.integers(4, 11))
        work = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(2, 2), iterations=iterations)
        if not work.any():
            work = mask
    return np.asarray(skeletonize(work, method="lee"), dtype=bool)


def _two_largest_components(mask: np.ndarray) -> list[np.ndarray]:
    labels, count = ndi.label(mask, structure=ndi.generate_binary_structure(2, 2))
    if count == 0:
        return []
    sizes = ndi.sum(mask, labels, range(1, count + 1))
    ordered = np.argsort(sizes)[::-1]
    chosen = [int(ordered[0])]
    if len(ordered) > 1 and sizes[ordered[1]] * 10 > sizes[ordered[0]]:
        chosen.append(int(ordered[1]))
    return [labels == item + 1 for item in chosen if sizes[item] > 15]


def _cut_branch(skeleton: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Keep one random centreline branch per major WSL-style component."""
    output = np.zeros(skeleton.shape, dtype=bool)
    height, width = skeleton.shape
    neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for component in _two_largest_components(skeleton):
        points = np.argwhere(component)
        if len(points) == 0:
            continue
        degrees = []
        for y, x in points:
            degrees.append(sum(0 <= y + dy < height and 0 <= x + dx < width and component[y + dy, x + dx] for dy, dx in neighbours))
        endpoints = points[np.asarray(degrees) == 1]
        start = endpoints[int(rng.integers(len(endpoints)))] if len(endpoints) else points[int(rng.integers(len(points)))]
        y, x = int(start[0]), int(start[1]); previous = None
        for _ in range(int(component.sum())):
            output[y, x] = True
            next_points = [(y + dy, x + dx) for dy, dx in neighbours if 0 <= y + dy < height and 0 <= x + dx < width and component[y + dy, x + dx] and (y + dy, x + dx) != previous and not output[y + dy, x + dx]]
            if not next_points:
                break
            previous = (y, x); y, x = next_points[int(rng.integers(len(next_points)))]
    return ndi.binary_dilation(output, structure=ndi.generate_binary_structure(2, 2), iterations=1)


def _wsl_foreground_seed(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    skeleton = _cut_branch(_wsl_skeleton(mask, rng), rng)
    if int(mask.sum()) > 1000 and skeleton.any():
        shifted = ndi.shift(skeleton.astype(np.uint8), shift=tuple(rng.integers(-6, 7, size=2)), order=0, mode="constant", cval=0) > 0
        rotated = ndi.rotate(shifted.astype(np.uint8), angle=float(rng.uniform(-15, 15)), reshape=False, order=0, mode="constant", cval=0) > 0
        skeleton = rotated & mask
    return skeleton & mask


def generate_wsl_style_from_baseline(labels: np.ndarray, annotations: np.ndarray, shift: int, seed: int, foreground_area_multiplier: float, background_area_multiplier: float) -> tuple[np.ndarray, AnnotationStats]:
    """WSL4MIS-style random skeleton scribbles with the existing area contract.

    Foreground uses eroded skeletons, a random branch cut, translation and
    rotation before clipping to the dense class.  Background uses the eroded
    background skeleton without branch cutting, so it forms anatomy-dependent
    paths instead of fixed border strokes.  The old v2 annotation fixes the
    requested foreground/background pixel budgets.
    """
    if labels.shape != annotations.shape:
        raise ValueError("labels and annotations must have identical shapes")
    result = np.full(labels.shape, IGNORE_INDEX, dtype=np.int16)
    for index, dense in enumerate(labels):
        foreground = dense > 0
        for local_class in sorted(set(np.unique(dense).tolist()) - {0}):
            value = int(local_class) + int(shift)
            rng = np.random.default_rng(seed * 1_000_003 + index * 97 + int(local_class))
            target = int(round(int((annotations[index] == value).sum()) * foreground_area_multiplier))
            result[index][_grow_to_count(_wsl_foreground_seed(dense == local_class, rng), dense == local_class, target)] = value
        rng = np.random.default_rng(seed * 1_000_003 + index * 97 + 89)
        target = int(round(int((annotations[index] == 0).sum()) * background_area_multiplier))
        background_seed = _wsl_skeleton(~foreground, rng)
        result[index][_grow_to_count(background_seed, (~foreground) & (result[index] == IGNORE_INDEX), target)] = 0
    valid = result != IGNORE_INDEX
    stats = AnnotationStats(int((result > 0).sum()), int((result == 0).sum()), int((result == IGNORE_INDEX).sum()), int((labels > 0).sum()), int((~(result > 0).reshape(len(result), -1).any(1)).sum()), int(valid.reshape(len(result), -1).sum(1).min()), int(valid.reshape(len(result), -1).sum(1).max()))
    return result, stats


def generate(labels: np.ndarray, shift: int, seed: int = 42, foreground_area_multiplier: float = 1.0, background_area_multiplier: float = 1.0) -> tuple[np.ndarray, AnnotationStats]:
    if foreground_area_multiplier < 1 or background_area_multiplier < 1:
        raise ValueError("area multipliers must be >= 1")
    result = np.full(labels.shape, IGNORE_INDEX, dtype=np.int16)
    for index, dense in enumerate(labels):
        foreground = dense > 0
        for local_class in sorted(set(np.unique(dense).tolist()) - {0}):
            line = ndi.binary_dilation(skeletonize(dense == local_class), iterations=1) & (dense == local_class)
            line = _expand(line, dense == local_class, foreground_area_multiplier)
            result[index][line] = int(local_class) + int(shift)
        base_bg = _background(foreground, seed, index) & (result[index] == IGNORE_INDEX)
        bg = _expand(base_bg, (~foreground) & (result[index] == IGNORE_INDEX), background_area_multiplier)
        result[index][bg] = 0
    valid = result != IGNORE_INDEX
    stats = AnnotationStats(int((result > 0).sum()), int((result == 0).sum()), int((result == IGNORE_INDEX).sum()), int((labels > 0).sum()), int((~(result > 0).reshape(len(result), -1).any(1)).sum()), int(valid.reshape(len(result), -1).sum(1).min()), int(valid.reshape(len(result), -1).sum(1).max()))
    return result, stats

