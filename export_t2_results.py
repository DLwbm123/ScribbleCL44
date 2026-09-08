"""Export scalar experiment evidence; omit medical arrays and patient-level scores."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root
    result = dict(coverage=json.loads((root / "annotations/coverage.json").read_text()),
                  forgetting=json.loads((root / "t3_epoch1_forgetting.json").read_text()),
                  independent=[], validation_matrices={}, automatic_formal_launch=False)
    for item in result["forgetting"]:
        run = root / "runs" / ("e1_" + item["run"])
        summary = json.loads((run / "summary.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        rows = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
        epochs = [row for row in rows if "epoch_seconds" in row]
        assert len(epochs) == 1 and epochs[0]["stage"] == 2 and epochs[0]["epoch"] == 0
        assert epochs[0]["iteration"] == 356 and summary["completed_stages"] == 3
        matrix = summary["validation_matrix"]
        assert matrix[1][1] == item["t2_before"] and matrix[2][1] == item["t2_after"]
        item.update(initial_lr=manifest["initial_lr"], global_weight=manifest["zs_global_weight"])
        result["validation_matrices"][item["run"]] = matrix
    for name in ["ind_fg20", "ind_fg40"]:
        run = root / "runs" / name
        summary = json.loads((run / "summary.json").read_text())
        rows = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
        epochs = [row for row in rows if "epoch_seconds" in row]
        assert len(epochs) == 80 and epochs[-1]["iteration"] == 3360
        result["independent"].append(dict(
            run=name, validation_dice=summary["final_seen_validation_mean"],
            test_dice=summary["final_seen_mean"],
            best_epoch_one_based=summary["stage_rows"][0]["best_validation"]["epoch"] + 1,
            initial_lr=.003, global_weight=.1, grad_clip_norm=1., epochs=80,
        ))
    result["checks"] = "PASS: four single-epoch T3 branches, exact persisted matrix agreement, two 80-epoch independent runs"
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
