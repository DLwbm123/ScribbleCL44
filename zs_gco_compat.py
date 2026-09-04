"""Compatibility shim for the modern ``gco`` wheel used by ZS PuzzleMix."""
from __future__ import annotations

import numpy as np
import gco


def cut_grid_graph(unary_cost: np.ndarray, pairwise_cost: np.ndarray,
                   vertical_cost: np.ndarray, horizontal_cost: np.ndarray,
                   algorithm: str = "swap") -> np.ndarray:
    """Match the legacy ``gco.cut_grid_graph`` contract used by ZScribbleSeg."""
    height, width, classes = unary_cost.shape
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
    return np.asarray(graph.label, dtype=np.int32).reshape(height, width)
