#!/usr/bin/env python3
"""Evaluate one stable continual-learning checkpoint without mutating training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--scenario", choices=("class", "organ"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code))
    from runner_core import TASKS, _build_model, _evaluate_task

    before = os.stat(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu")
    after = os.stat(args.checkpoint)
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise RuntimeError("checkpoint changed during load; retry from the next stable save")

    device = torch.device(args.device)
    model = _build_model(args.scenario)
    for stage in range(args.stage + 1):
        model.activate_stage(stage)
    model.load_state_dict(state)
    model.to(device)
    scores = {}
    for index, task in enumerate(TASKS[args.scenario][: args.stage + 1]):
        scores[task.code] = _evaluate_task(
            model, args.scenario, task, index, args.data_root, "test", args.batch_size, device,
        )
    means = [score["benchmark_mean"] for score in scores.values()]
    print(json.dumps({
        "scenario": args.scenario,
        "checkpoint": args.checkpoint.name,
        "stage": args.stage + 1,
        "scores": scores,
        "seen_mean": sum(means) / len(means),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
