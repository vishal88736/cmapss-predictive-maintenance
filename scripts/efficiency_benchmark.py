"""Measure cost of the frozen artifact rather than only its parameter count.

Reports trainable parameters per module, multiply-accumulate count for one
70-cycle window, single-window and batched latency on the requested device, and
the cost of the recovery path relative to a bypass. Latency is a wall-clock
measurement of this machine, not a hardware-independent claim.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import resource
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from grace_rul.artifacts import load_bundle
from grace_rul.training import device_for

DEFAULT_BUNDLE = ROOT / "artifacts/pooled_repair_20261101T230338Z_9ea6_dpk/final_model_verified/grace_rul_ALL.pt"
MODULES = ("detector", "generator", "predictor", "corrector")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per_module = {}
    for name in MODULES:
        module = getattr(model, name, None)
        if module is not None:
            per_module[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, per_module


def multiply_accumulates(model, window, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None, None
    results = {}
    for label, options in [("full", {}), ("bypass", {"use_repair": False})]:
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            for _ in range(2):
                model(window, torch.ones(window.shape[:2], dtype=torch.bool, device=device), **options)
        results[label] = int(counter.get_total_flops())
    return results["full"], results["bypass"]


def latency(model, batch_size, device, repeats):
    """Paired, interleaved timing so CPU drift cannot favour one path."""
    window = torch.randn(batch_size, model.config["sequence_length"], 24, device=device)
    valid = torch.ones(window.shape[:2], dtype=torch.bool, device=device)
    variants = [("full", {}), ("bypass", {"use_repair": False}), ("plain", {"use_repair": False, "use_correction": False})]
    samples = {label: [] for label, _ in variants}
    with torch.no_grad():
        for _ in range(3):
            for _, options in variants:
                model(window, valid, **options)
        for _ in range(repeats):
            for label, options in variants:
                start = time.perf_counter()
                model(window, valid, **options)
                samples[label].append(time.perf_counter() - start)
    return {label: statistics.median(values) for label, values in samples.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args(argv)
    device = device_for({"device": args.device})
    root, bundle_path = ROOT, Path(args.bundle).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output_dir or root / "artifacts" / f"efficiency_benchmark_{stamp}").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Efficiency output directory must be empty.")
    output.mkdir(parents=True, exist_ok=True)

    rss_baseline = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    frozen_hash = sha(bundle_path)
    bundle, loaded = load_bundle(bundle_path, device=str(device))
    model = loaded["POOLED"]["model"]
    model.eval()
    total, per_module = parameters(model)

    window = torch.randn(1, model.config["sequence_length"], 24, device=device)
    flops_full, flops_bypass = multiply_accumulates(model, window, device)
    per_window = latency(model, 1, device, args.repeats)
    per_batch = latency(model, 64, device, args.repeats)
    engines = 707
    rows = [
        {"metric": "trainable_parameters", "value": total, "unit": "parameters", "note": "exported artifact"},
        *[{"metric": f"parameters_{name}", "value": value, "unit": "parameters", "note": "module"}
          for name, value in per_module.items()],
        {"metric": "model_size", "value": round(total * 4 / 1024 ** 2, 3), "unit": "MB",
         "note": "float32 weights only"},
        {"metric": "mac_per_window_full", "value": flops_full, "unit": "multiply-accumulates",
         "note": "one 70-cycle window, recovery enabled"},
        {"metric": "mac_per_window_bypass", "value": flops_bypass, "unit": "multiply-accumulates",
         "note": "one 70-cycle window, recovery disabled"},
        {"metric": "latency_window_full", "value": round(1000 * per_window["full"], 3), "unit": "ms",
         "note": "batch 1, median"},
        {"metric": "latency_window_bypass", "value": round(1000 * per_window["bypass"], 3), "unit": "ms",
         "note": "batch 1, median, recovery disabled"},
        {"metric": "latency_window_plain", "value": round(1000 * per_window["plain"], 3), "unit": "ms",
         "note": "batch 1, median, recovery and corrector disabled"},
        {"metric": "latency_batch64_full", "value": round(1000 * per_batch["full"], 3), "unit": "ms",
         "note": "batch 64, median"},
        {"metric": "throughput_engines_per_second", "value": round(64 / per_batch["full"], 1),
         "unit": "engines/s", "note": "batch 64, full path"},
        {"metric": "latency_per_fleet", "value": round(engines * per_window["full"], 3), "unit": "s",
         "note": f"{engines} engines, one window each, batch 1"},
        {"metric": "rss_baseline", "value": rss_baseline, "unit": "MB", "note": "interpreter and torch only"},
        {"metric": "peak_rss", "value": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
         "unit": "MB", "note": "process peak after load and benchmark"},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(output / "efficiency_metrics.csv", index=False)
    if sha(bundle_path) != frozen_hash:
        raise RuntimeError("The frozen bundle changed during benchmarking.")
    manifest = {
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Wall-clock measurement of this machine; not a hardware-independent claim.",
        "bundle": str(bundle_path), "bundle_sha256": frozen_hash, "device": str(device),
        "torch_version": torch.__version__, "repeats": args.repeats,
        "sequence_length": model.config["sequence_length"],
        "python": platform.python_version(), "processor": platform.processor() or platform.machine(),
        "multiply_accumulates_note": "torch FlopCounterMode counts multiply-accumulates for supported ops only; "
                                     "it is a complexity proxy, not total energy or memory traffic.",
        "outputs": {"efficiency_metrics.csv": sha(output / "efficiency_metrics.csv")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(table.to_string(index=False), flush=True)
    print(f"Saved efficiency benchmark: {output}", flush=True)


if __name__ == "__main__":
    main()
