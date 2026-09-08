"""Compatibility shim for the modern ``gco`` wheel used by ZS PuzzleMix."""
from __future__ import annotations

import numpy as np
import gco
from numerical_safety import validate_graph_labels


def cut_grid_graph(unary_cost: np.ndarray, pairwise_cost: np.ndarray,
                   vertical_cost: np.ndarray, horizontal_cost: np.ndarray,
                   algorithm: str = "swap") -> np.ndarray:
    """Match the legacy ``gco.cut_grid_graph`` contract used by ZScribbleSeg."""
    unary_cost = np.asarray(unary_cost)
    pairwise_cost = np.asarray(pairwise_cost)
    vertical_cost = np.asarray(vertical_cost)
    horizontal_cost = np.asarray(horizontal_cost)
    if unary_cost.ndim != 3:
        raise ValueError("unary graph-cut costs must be [height,width,classes]")
    height, width, classes = unary_cost.shape
    expected = ((classes, classes), (height - 1, width), (height, width - 1))
    if (pairwise_cost.shape, vertical_cost.shape, horizontal_cost.shape) != expected:
        raise ValueError("graph-cut cost shapes do not match the unary grid")
    for name, value in (("unary", unary_cost), ("pairwise", pairwise_cost), ("vertical", vertical_cost), ("horizontal", horizontal_cost)):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{name}: non-finite graph-cut cost")
    graph = gco.GCOGridGraph(width, height, classes)
    graph.set_data_cost(np.ascontiguousarray(unary_cost.reshape(-1, classes), dtype=np.float64))
    vertical = np.zeros((height, width), dtype=np.float64)
    horizontal = np.zeros((height, width), dtype=np.float64)
    vertical[:-1, :] = vertical_cost
    horizontal[:, :-1] = horizontal_cost
    graph.set_smooth_cost(
        np.ascontiguousarray(pairwise_cost, dtype=np.float64),
        np.ascontiguousarray(vertical.ravel(), dtype=np.float64),
        np.ascontiguousarray(horizontal.ravel(), dtype=np.float64),
    )
    if algorithm == "swap":
        graph.swap(-1)
    else:
        graph.expansion(-1)
    labels = np.asarray(graph.label, dtype=np.int32).reshape(height, width)
    validate_graph_labels(labels, classes)
    return labels
