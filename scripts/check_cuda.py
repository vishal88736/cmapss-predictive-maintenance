"""Verify actual CUDA execution, optionally including staged training and WGAN-GP."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def check_cuda(device_name="cuda"):
    if torch.version.cuda is None:
        raise RuntimeError("This Python environment has CPU-only PyTorch. Run bash scripts/setup_env.sh --cuda to install the CUDA build.")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("This check requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is installed, but the GPU is not visible to this process. Run nvidia-smi in the same terminal; a restricted workspace can hide GPU devices.")
    tensor = torch.ones(32, 32, device=device, requires_grad=True)
    (tensor @ tensor).mean().backward()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(tensor.grad).all()):
        raise FloatingPointError("CUDA tensor backward produced non-finite gradients.")
    properties = torch.cuda.get_device_properties(device)
    return {"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "device": str(device), "gpu": properties.name,
            "gpu_memory_GiB": round(properties.total_memory / 1024 ** 3, 2),
            "tensor_forward_backward": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="CUDA device, such as cuda:0.")
    parser.add_argument("--training-check", action="store_true", help="One batch per stage using full model dimensions; not accuracy evidence.")
    args = parser.parse_args()
    report = check_cuda(args.device)
    if args.training_check:
        from grace_rul.config import DEFAULT_CONFIG
        from grace_rul.data import load_training_subset, split_engines
        from grace_rul.training import fit_pipeline
        config = {**DEFAULT_CONFIG, "device": args.device,
                  "predictor_epochs": 1, "generator_epochs": 1, "joint_epochs": 1,
                  "maximum_engines": 18, "maximum_windows": DEFAULT_CONFIG["batch_size"],
                  "maximum_batches": 1, "adversarial_warmup": 0,
                  "tune_before_training": False, "run_matched_benchmarks": False,
                  "quick_run": True}
        frame = load_training_subset(ROOT / "data", "FD001")
        groups = split_engines(frame, config["split_seed"], config["maximum_engines"],
                               config["validation_fraction"], config["calibration_fraction_of_training"])
        fit, validation = [frame[frame.unit.isin(groups[name])] for name in ["fit", "validation"]]
        torch.cuda.reset_peak_memory_stats(args.device)
        model, _, history, score = fit_pipeline(fit, validation, "FD001", config, config["seed"])
        torch.cuda.synchronize(args.device)
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("Training did not keep model parameters on CUDA.")
        report["training_check"] = {"passed": True, "stages": history.stage.tolist(),
            "batch_size": config["batch_size"], "gradient_penalty_batch_size": config["gradient_penalty_batch_size"],
            "model_dimensions": "full defaults", "validation_RMSE": score,
            "peak_allocated_GiB": round(torch.cuda.max_memory_allocated(args.device) / 1024 ** 3, 3),
            "scope": "execution only; one batch per stage; official test files excluded"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
