#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--scenario", choices=("class", "organ", "domain"), required=True)
    parser.add_argument("--expected-stages", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    summary = json.loads((args.output / "summary.json").read_text())
    rows = [json.loads(line) for line in (args.output / "train.jsonl").read_text().splitlines()]
    epoch_rows = [row for row in rows if "loss" in row]
    is_joint = args.method == "zs-joint"
    expected_stages = args.expected_stages or (1 if is_joint else 2)
    expected_epoch_rows = manifest["epochs_per_task"] * expected_stages
    state = torch.load(args.output / f"s{expected_stages:02d}_state.pt", map_location="cpu")
    failures = []
    if manifest["method"] != args.method or summary["method"] != args.method:
        failures.append("method mismatch")
    if summary["completed_stages"] != expected_stages or len(epoch_rows) != expected_epoch_rows:
        failures.append("stage completion mismatch")
    if manifest.get("selection_split") != "validation" or manifest.get("test_for_selection") is not False:
        failures.append("selection provenance contract violation")
    if "validation_matrix" not in summary:
        failures.append("missing validation matrix")
    if not manifest.get("test_evaluation", False):
        if any(value is not None for row in summary["matrix"] for value in row):
            failures.append("test metrics present in validation-only run")
    if is_joint:
        matrix = summary["matrix"]
        if args.scenario != "domain" or manifest.get("training_mode") != "joint":
            failures.append("joint mode contract violation")
        if len(matrix) != 1 or len(matrix[0]) != 6 or any(value is None for value in matrix[0]):
            failures.append("joint matrix contract violation")
        if any(summary[key] is not None for key in ("BWTR", "RMA", "E-FWT")):
            failures.append("undefined joint continual metric is populated")
    uses_derpp = args.method in {"zs-derpp", "zs-derpp-mib"}
    uses_mib = args.method in {"zs-mib", "zs-derpp-mib"}
    uses_replay = args.method.endswith("-der") or uses_derpp
    replay_flags = (
        manifest["history_images"], manifest["replay"],
        summary["history_images"], summary["replay"],
    )
    if any(replay_flags) != uses_replay or len(set(replay_flags)) != 1:
        failures.append("history image or replay contract violation")
    if state["stage"] != expected_stages - 1 or not state["model"]:
        failures.append("final state contract violation")
    if args.method.endswith("-ewc"):
        fisher = json.loads((args.output / "fisher.json").read_text())
        if len(fisher) != expected_stages or min(row["nonzero"] for row in fisher) <= 0:
            failures.append("invalid Fisher")
        if state["continual"] is None or not state["continual"]["anchor"]:
            failures.append("missing EWC state")
    elif args.method.endswith("-gpm"):
        gpm = json.loads((args.output / "gpm.json").read_text())
        if len(gpm) != expected_stages or min(len(row["layers"]) for row in gpm) <= 0:
            failures.append("invalid GPM")
        if state["continual"] is None or not state["continual"]["bases"]:
            failures.append("missing GPM state")
    elif args.method.endswith("-der"):
        continual = state["continual"]
        if continual is None or continual["examples"] is None:
            failures.append("missing DER buffer state")
        elif continual["examples"].shape[0] > manifest["der_buffer_size"]:
            failures.append("DER buffer overflow")
        if max(row["der_penalty"] for row in epoch_rows) <= 0:
            failures.append("DER replay objective did not run")
    elif uses_derpp:
        continual = state["continual"]
        if continual is None or continual["examples"] is None:
            failures.append("missing DER++ buffer state")
        elif continual["examples"].shape[0] > manifest["der_buffer_size"]:
            failures.append("DER++ buffer overflow")
        if max(row["derpp_feature_loss"] for row in epoch_rows) <= 0:
            failures.append("DER++ feature objective did not run")
        if max(row["derpp_pce_loss"] for row in epoch_rows) <= 0:
            failures.append("DER++ replay PCE did not run")
        if max(row["derpp_global_loss"] for row in epoch_rows) <= 0:
            failures.append("DER++ replay global objective did not run")
        coverage = continual.get("coverage")
        if coverage is None:
            failures.append("missing DER++ coverage audit")
        else:
            stored = coverage["stored_examples_by_task"]
            if sum(stored.values()) != continual["examples"].shape[0]:
                failures.append("DER++ coverage count mismatch")
            if expected_stages > 1 and not coverage["replay_examples_by_task"]:
                failures.append("DER++ replay coverage missing")
    elif state["continual"] is not None:
        failures.append("unexpected continual state")
    if uses_mib and epoch_rows[-1]["mib_kd_loss"] <= 0:
        failures.append("MiB KD did not run")
    if args.method.startswith("zs-") and min(row["zs_global_loss"] for row in epoch_rows) <= 0:
        failures.append("ZS global objective did not run at every stage")
    keys = tuple(state["model"])
    if args.scenario == "organ" and not any(key.startswith("heads.1.") for key in keys):
        failures.append("second organ head missing")
    if args.scenario == "class" and not any(key.startswith("blocks.1.") for key in keys):
        failures.append("second class block missing")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "scenario": args.scenario,
        "method": args.method,
        "completed_stages": summary["completed_stages"],
        "epoch_rows": len(epoch_rows),
        "selection_split": manifest.get("selection_split"),
        "test_evaluation": manifest.get("test_evaluation"),
        "derpp_coverage": (
            None
            if not uses_derpp or state["continual"] is None
            else state["continual"].get("coverage")
        ),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
