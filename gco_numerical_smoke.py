#!/usr/bin/env python3
"""Small real-GCO contract test, including exhaustive 2x2 energy checking."""
from __future__ import annotations

import itertools
import json

import numpy as np

from zs_gco_compat import cut_grid_graph


def energy(labels, unary, pairwise, vertical, horizontal):
    total = sum(unary[row, column, labels[row, column]] for row in range(2) for column in range(2))
    total += sum(pairwise[labels[row, column], labels[row + 1, column]] * vertical[row, column] for row in range(1) for column in range(2))
    total += sum(pairwise[labels[row, column], labels[row, column + 1]] * horizontal[row, column] for row in range(2) for column in range(1))
    return float(total)


def main() -> None:
    unary = np.array([[[0, 4], [1, 0]], [[3, 0], [0, 2]]], dtype=np.int32)
    pairwise = np.array([[0, 1], [1, 0]], dtype=np.int32)
    vertical = np.array([[2, 1]], dtype=np.int32)
    horizontal = np.array([[1], [2]], dtype=np.int32)
    labels = cut_grid_graph(unary, pairwise, vertical, horizontal)
    actual = energy(labels, unary, pairwise, vertical, horizontal)
    optimum = min(energy(np.array(values).reshape(2, 2), unary, pairwise, vertical, horizontal) for values in itertools.product(range(2), repeat=4))
    rejected_nonfinite = False
    try:
        cut_grid_graph(np.full(unary.shape, np.nan), pairwise, vertical, horizontal)
    except FloatingPointError:
        rejected_nonfinite = True
    result = {
        "status": "PASS" if actual == optimum and rejected_nonfinite else "FAIL",
        "labels": labels.tolist(),
        "energy": actual,
        "exhaustive_optimum": optimum,
        "nonfinite_cost_rejected": rejected_nonfinite,
    }
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
