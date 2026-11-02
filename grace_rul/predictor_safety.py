"""Training/selection objectives for large RUL errors; no inference rules."""

import math

import numpy as np
import pandas as pd
import torch


def tail_rul_loss(prediction, target, *, scale, fraction=.1):
    """Mean squared loss on the largest fitting errors in the current batch.

    Targets enter training only. There is no engine-specific rule, prediction
    floor, output replacement or monotonic post-processing at inference.
    """
    if not np.isfinite([scale, fraction]).all() or scale <= 0 or not 0 < fraction <= 1:
        raise ValueError("Tail-loss scale and fraction must be positive and finite.")
    if prediction.shape != target.shape or prediction.ndim != 1 or not len(prediction):
        raise ValueError("Tail loss needs equally shaped nonempty prediction/target vectors.")
    error = ((prediction - target) / scale).square()
    return error.topk(max(1, math.ceil(fraction * error.numel()))).values.mean()


def engine_tail_score(records, fraction=.1):
    """Mean of the worst fraction of engine maximum-prefix absolute errors."""
    frame = pd.DataFrame(records)
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Engine tail fraction must be in (0, 1].")
    if frame.empty or not {"unit", "cycle", "prediction", "target"}.issubset(frame):
        raise ValueError("Engine-tail selection requires identified engine-prefix predictions.")
    if frame.duplicated(["unit", "cycle"]).any():
        raise ValueError("Engine-tail selection contains duplicate prefixes.")
    errors = (frame.prediction-frame.target).abs()
    if not np.isfinite(errors).all():
        raise ValueError("Engine-tail errors must be finite.")
    maxima = errors.groupby(frame.unit).max()
    count = max(1, math.ceil(fraction * len(maxima)))
    return float(maxima.nlargest(count).mean())


def fault_accuracy_guards(reports, references, tolerance):
    """Protect fault RUL accuracy globally and within each source."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Fault accuracy tolerance must be finite and nonnegative.")
    guards = {}
    for family, reference in references.items():
        measured = reports.get(family, {})
        guards[f"{family}_rul"] = measured.get("RMSE", float("inf")) <= reference["RMSE"] * (1+tolerance)
        for source, target in reference.get("source_metrics", {}).items():
            value = measured.get("source_metrics", {}).get(source, {}).get("RMSE", float("inf"))
            guards[f"{family}_rul_{source}"] = value <= target["RMSE"] * (1+tolerance)
    return {key: bool(value) for key, value in guards.items()}


def restore_refinement_checkpoint(path, model, optimizer, scheduler, generator, *,
                                  config, source_hash, implementation):
    """Restore an interrupted matching local run, including sampled-window RNG."""
    from .checkpoints import restore_random_state
    # RNG tensors must remain on CPU even when resuming a CUDA model. Optimizer
    # loading moves its moment tensors according to the destination parameters.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    comparable = lambda recipe: {key:value for key,value in recipe.items() if key != "device"}
    if (saved["source_bundle_sha256"] != source_hash or comparable(saved["config"]) != comparable(config) or
            saved["implementation_sha256"] != implementation):
        raise ValueError("Resume source, recipe or implementation differs; use a fresh directory.")
    model.load_state_dict(saved["state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    generator.set_state(saved["loader_rng"].cpu())
    restore_random_state(saved["rng"])
    return saved
