"""Organ T2 adapter around the preserved Domain shared training loop.

The reference checkout is an explicit dependency; no training loop is copied.
Only the model container and annotation path vary. Validation selects checkpoints.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys


def configure(reference, model, annotation):
    reference.TASKS["domain"] = (reference.Task("T2", "Task_incre", "UCL.h5", (1,)),)
    reference._build_model = lambda scenario: (
        reference.OrganModel() if model == "organ" else reference.DomainModel())
    reference._sparse_path = lambda *args: annotation


def arguments(args, output):
    return ["--data-root", str(args.data_root), "--sparse-root", str(args.annotation.parent),
            "--output", str(output), "--device", "cuda:0", "--seed", "42",
            "--independent-reference", "--independent-task", "1", "--method", "zs-sequential",
            "--epochs-per-task", "80", "--batch-size", "4", "--lr", ".03", "--workers", "8",
            "--validate-every", "42", "--pce-loss-weight", "1", "--zs-global-weight", "1",
            "--zs-spatial-loss-weight", ".01", "--zs-spatial-warmup-epochs", str(args.spatial_start_epoch - 2)]


def canonical(state):
    return {k.replace("heads.0.", "head."): v for k, v in state.items()}


def parity(reference, args):
    import torch
    from check_independent_a_parity import capture
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    train = reference.H5Slices(args.data_root / "Task_incre/UCL.h5", "train",
                               args.annotation, augment=True)
    loader = reference._loader(train, 4, True, 8, 42)
    assert len(train) == 166 and len(loader) == 42
    batch = next(iter(loader))
    train.close()
    captures = []
    for model in ("domain", "domain", "organ"):
        configure(reference, model, args.annotation)
        argv = arguments(args, args.output / f"{model}_{len(captures)}")
        argv[argv.index("--validate-every") + 1] = "1"
        captures.append(capture(reference, argv + ["--independent-skip-test"], batch))
    results = {"scope": "fixed augmented UCL batch through real loops; no test access",
               "actual_batches_per_epoch": 42, "capture_helper_batches_per_epoch": 76,
               "note": "Existing capture helper uses synthetic length 76 for both models; production uses 42.",
               "limits": {"initial": 0, "gradients": 1e-5, "updated": 1e-6}, "passed": True}
    for key, limit in results["limits"].items():
        a, b, c = [canonical(value[key]) for value in captures]
        assert a.keys() == b.keys() == c.keys()
        delta = max(float((a[k].double() - c[k].double()).abs().max()) for k in a)
        repeat = max(float((a[k].double() - b[k].double()).abs().max()) for k in a)
        results[key] = {"organ_max_abs_diff": delta, "domain_repeat_max_abs_diff": repeat}
        results["passed"] &= delta <= limit
    for key in ("pce_loss", "global_loss", "total_loss", "lr_before", "lr_after", "optimizer_weight_decay"):
        values = [value[key] for value in captures]
        results[key] = values
        results["passed"] &= abs(values[0] - values[2]) < 1e-7
    (args.output / "parity.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    assert results["passed"], "Model-container parity failed; do not launch training"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["domain", "organ"], default="organ")
    parser.add_argument("--check-parity", action="store_true")
    parser.add_argument("--spatial-start-epoch", type=int, default=11,
                        help="First spatial-active epoch, one-based; reference uses zero-based epoch > warmup")
    args = parser.parse_args()
    if not 1 <= args.spatial_start_epoch <= 80:
        parser.error("spatial-start-epoch must be between 1 and 80")
    sys.path.insert(0, str(args.reference_source.resolve()))
    reference = importlib.import_module("runner_core")
    assert Path(reference.__file__).resolve() == (args.reference_source / "runner_core.py").resolve()
    configure(reference, args.model, args.annotation)
    if args.check_parity:
        parity(reference, args)
        return

    import torch
    real_sgd = torch.optim.SGD
    class FiniteSGD(real_sgd):
        def step(self, closure=None):
            parameters = [p for group in self.param_groups for p in group["params"]]
            if not bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]).all()):
                raise FloatingPointError("Non-finite gradient; no clipping or silent correction")
            result = super().step(closure)
            if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):
                raise FloatingPointError("Non-finite updated parameter")
            return result
    torch.optim.SGD = FiniteSGD
    argv = arguments(args, args.output)
    sys.argv = [str(Path(__file__).resolve()), *argv]
    reference.main("domain")
    # Retain the dispatcher's own manifest rather than relabeling it as a native Organ loop.
    (args.output / "adapter.json").write_text(json.dumps({
        "experiment": "Organ T2 independent", "model_container": args.model,
        "training_dispatcher": "Domain independent shared stage loop",
        "reference_source": str(args.reference_source), "annotation": str(args.annotation),
        "selection_split": "val", "finite_checks_only": True, "gradient_clipping": False,
        "command": argv,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
