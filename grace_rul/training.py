"""Staged training, grouped out-of-fold correction, calibration, and evaluation."""

import copy
import math
import random
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from .data import FAULTS, OperatingRegimeScaler, WindowDataset
from .model import GraceRUL, TemporalCritic
from .normalization import transform_spec
from .checkpoints import load_checkpoint, pipeline_identity, random_state, restore_random_state, save_checkpoint
from .calibration import calibrate_endpoint_intervals, interval_bounds


def seed_all(seed, cpu_threads=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(config):
    requested = config.get("device", "cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Training device must be cpu, cuda, cuda:<index>, or auto.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            detail = " Install CUDA PyTorch with bash scripts/setup_env.sh --cuda." if torch.version.cuda is None else " Check GPU visibility with nvidia-smi in the same terminal; restricted workspaces can hide GPU devices."
            raise RuntimeError("CUDA GPU training was requested, but PyTorch cannot access a CUDA GPU." + detail + " CPU execution requires an explicit device='cpu' setting.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("The requested CUDA device index is not available.")
    return device


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=device.type == "cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def loader(dataset, config, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if shuffle and config["engine_balanced_sampling"]:
        engines = np.asarray([engine for engine, _ in dataset.samples])
        counts = np.bincount(engines, minlength=len(dataset.engines))
        sampler = WeightedRandomSampler(1. / counts[engines], len(dataset), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle and sampler is None, sampler=sampler, num_workers=0,
                      pin_memory=device_for(config).type == "cuda",
                      generator=generator)


def metrics(prediction, truth):
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    error = prediction - truth
    with np.errstate(over="ignore"):
        score = np.expm1(np.where(error < 0, -error / 13., error / 10.))
    variance = ((truth - truth.mean()) ** 2).sum()
    return {"N": len(truth), "RMSE": float(np.sqrt(np.mean(error ** 2))), "MAE": float(np.abs(error).mean()),
            "NASA_Score": float(score.sum()), "NASA_Score_Mean": float(score.mean()),
            "R2": float(1 - (error ** 2).sum() / variance) if variance > 0 else None}


def masked_mse(error, mask):
    return (error.square() * mask).sum() / mask.sum().clamp_min(1)


def masked_repair_loss(error, mask, config):
    if config.get("repair_loss", "mse") == "mse":
        return masked_mse(error, mask)
    values = F.smooth_l1_loss(error, torch.zeros_like(error),
                              beta=config["repair_huber_beta"], reduction="none")
    return (values * mask).sum() / mask.sum().clamp_min(1)


def intact_identity_loss(repaired, reference, intact, config):
    """Penalize healthy-sensor displacement in the same space as its guard.

    Asinh compresses large regime-normalized readings. A small encoded error
    can otherwise conceal a large physical-scale displacement. Float64 and the
    sinh difference identity avoid subtracting two large reconstructed values;
    masked faulty entries never enter the inverse transform.
    """
    if config.get("identity_loss_space", "encoded") != "original_linear":
        return masked_mse(repaired - reference, intact)
    a, b = repaired.double()[intact], reference.double()[intact]
    method, scale = transform_spec(config)
    error = (2 * scale * torch.cosh((a + b) / (2 * scale)) *
             torch.sinh((a - b) / (2 * scale))) if method == "asinh" else a - b
    budget = max(float(config["max_intact_repair_rmse"]), 1e-8)
    return (error / budget).square().sum() / max(1, error.numel())


def _log_linear_group_mse(prediction, reference, mask, config):
    """Exact original-linear log MSE per window/channel, without sinh overflow.

    Zero-error groups use a finite unused branch so logsumexp never receives
    only minus infinities in backward. Masked entries are excluded before any
    transform; zero-error entries contribute zero, not an artificial error floor.
    """
    mask = mask.bool()
    a = prediction.double().masked_fill(~mask, 0.)
    b = reference.double().masked_fill(~mask, 0.)
    nonzero = mask & (a != b)
    difference = torch.where(nonzero, a - b, torch.ones_like(a))
    method, scale = transform_spec(config)
    if method == "asinh":
        middle = ((a + b) / (2 * scale)).abs()
        delta = (difference / (2 * scale)).abs()
        log_cosh = middle + F.softplus(-2 * middle) - math.log(2.)
        log_sinh = delta + torch.log(-torch.expm1(-2 * delta)) - math.log(2.)
        log_abs = math.log(2 * scale) + log_cosh + log_sinh
    else:
        log_abs = difference.abs().log()
    has_error = nonzero.any(1)
    logs = (2 * log_abs).masked_fill(~nonzero, -torch.inf)
    logs = torch.where(has_error.unsqueeze(1), logs, torch.zeros_like(logs))
    counts = mask.sum(1)
    log_mean = torch.logsumexp(logs, 1) - counts.clamp_min(1).double().log()
    return log_mean, counts, has_error


def missing_imputation_loss(repaired, reference, missing, config):
    """Balance physical-scale missing recovery against its mean-fill baseline.

    Each supported window/channel contributes log(1 + error MSE /
    (zero-fill MSE + budget squared)). All inputs to the normalization are
    fitting-batch references; no validation targets or statistics train this loss.
    The unchanged acceptance guard still requires measured improvement.
    """
    error, counts, supported = _log_linear_group_mse(repaired, reference, missing, config)
    base, _, base_supported = _log_linear_group_mse(torch.zeros_like(reference), reference, missing, config)
    budget = max(float(config["max_intact_repair_rmse"]), 1e-8)
    base = base.masked_fill(~base_supported, -torch.inf)
    denominator = torch.logaddexp(base, base.new_full(base.shape, 2 * math.log(budget)))
    values = torch.where(supported, F.softplus(error - denominator), torch.zeros_like(error))
    return values[counts > 0].mean() if bool((counts > 0).any()) else values.sum() * 0.


def intact_tail_identity_loss(repaired, reference, intact, config):
    """Protect rare window/channel groups that a whole-batch mean can hide.

    Fault labels are training-only. Every group averages over its intact real
    timesteps; the upper fraction of those group errors drives this extra term.
    A truly altered reading is excluded, so large legitimate repairs remain
    learnable. The original mean identity term is retained separately.
    """
    mask = intact.bool()
    if config.get("identity_tail_penalty", "mse") == "log_mse":
        logs, counts, supported = _log_linear_group_mse(repaired, reference, mask, config)
        budget = max(float(config["max_intact_repair_rmse"]), 1e-8)
        values = torch.where(supported, F.softplus(logs - 2 * math.log(budget)), torch.zeros_like(logs))
        groups = values[counts > 0]
        if not groups.numel():
            return values.sum() * 0.
        k = max(1, math.ceil(groups.numel() * config.get("identity_tail_fraction", .1)))
        return groups.topk(k).values.mean()
    a = repaired.double().masked_fill(~mask, 0.)
    b = reference.double().masked_fill(~mask, 0.)
    method, scale = transform_spec(config)
    error = (2 * scale * torch.cosh((a + b) / (2 * scale)) *
             torch.sinh((a - b) / (2 * scale))) if method == "asinh" else a - b
    budget = max(float(config["max_intact_repair_rmse"]), 1e-8)
    counts = mask.sum(1)
    groups = ((error / budget).square().sum(1) / counts.clamp_min(1))[counts > 0]
    if not groups.numel():
        return error.sum() * 0.
    k = max(1, math.ceil(groups.numel() * config.get("identity_tail_fraction", .1)))
    return groups.topk(k).values.mean()


def finite_fault_loss(logits, altered, available, config):
    if config.get("detector_loss", "weighted") == "weighted":
        positives = (altered & available).sum((0, 1))
        negatives = available.sum((0, 1)) - positives
        weight = ((negatives + 1) / (positives + 1)).sqrt().clamp(1., config["detector_positive_weight_cap"])
        values = F.binary_cross_entropy_with_logits(logits, altered.float(), pos_weight=weight, reduction="none")
        return (values * available).sum() / available.sum().clamp_min(1)
    # Balance finite positives/negatives within each channel. A rare fault must
    # not disappear in a window-wide majority of intact readings and commands.
    values = F.binary_cross_entropy_with_logits(logits, altered.float(), reduction="none")
    positive, negative = altered & available, ~altered & available
    np_, nn_ = positive.sum((0, 1)), negative.sum((0, 1))
    pos = (values * positive).sum((0, 1)) / np_.clamp_min(1)
    neg = (values * negative).sum((0, 1)) / nn_.clamp_min(1)
    count = (np_ > 0).float() + (nn_ > 0).float()
    channel_loss = (pos + neg) / count.clamp_min(1)
    return (channel_loss * (count > 0)).sum() / (count > 0).sum().clamp_min(1)


def repair_trend_loss(repaired, reference, altered, valid, config=None):
    pairs = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1)
    pairs = pairs & (altered[:, 1:] | altered[:, :-1])
    error = (repaired[:, 1:] - repaired[:, :-1]) - (reference[:, 1:] - reference[:, :-1])
    return masked_repair_loss(error, pairs, config or {})


@contextmanager
def evaluation_mode(module):
    """Disable dropout for a reference target and restore all prior modes."""
    states = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        yield
    finally:
        for child, training in states:
            child.training = training


def branch_objective(model, result, target):
    names = ["local_prediction", "global_prediction"]
    if model.config["predictor_family"] == "gated":
        branch = model.config.get("predictor_branch", "both")
        names = [name for name in names if branch == "both" or name.startswith(branch)]
    scale = model.config["target_scale"]
    return sum(F.mse_loss(result[name] / scale, target / scale) for name in names) / len(names)


def objectives(model, batch, critic=None, consistency=True):
    result = model(batch["x"], batch["valid"], use_correction=False)
    valid = batch["valid"].unsqueeze(-1)
    changed = batch["altered"] & valid
    intact = ~batch["altered"] & valid
    error = result["repaired"] - batch["reference"][..., 3:]
    rec = masked_repair_loss(error, changed, model.config)
    identity = intact_identity_loss(result["repaired"], batch["reference"][..., 3:], intact, model.config)
    # The deployed projection bounds actual finite moves. Penalize the gated
    # proposal before projection too, so saturated unsafe proposals still learn
    # from their healthy-cell labels instead of receiving a clipped zero grad.
    identity_tail = (intact_tail_identity_loss(result["repair_proposal"], batch["reference"][..., 3:], intact, model.config)
        if model.config.get("identity_tail_loss_weight", 0.) > 0 else identity * 0.)
    generator_rec = masked_repair_loss(result["generated"] - batch["reference"][..., 3:], changed, model.config)
    imputation = (missing_imputation_loss(result["repaired"], batch["reference"][..., 3:],
                    changed & ~result["mask"].bool(), model.config)
                  if model.config.get("imputation_loss_weight", 0.) > 0 else identity * 0.)
    trend = repair_trend_loss(result["repaired"], batch["reference"][..., 3:], batch["altered"], batch["valid"], model.config)
    available = result["mask"].bool() & valid
    detection = finite_fault_loss(result["logits"], batch["altered"], available, model.config)
    scale = model.config["target_scale"]
    rul = F.mse_loss(result["base"] / scale, batch["target"] / scale)
    branch = branch_objective(model, result, batch["target"])
    cons = clean_rul = bypass_rul = result["base"].sum() * 0
    clean_grad = model.config.get("clean_rul_loss_weight", 0.) > 0 and any(p.requires_grad for p in model.predictor.parameters())
    if consistency or clean_grad:
        with torch.set_grad_enabled(torch.is_grad_enabled() and clean_grad), evaluation_mode(model):
            if clean_grad:
                # A deterministic clean anchor still needs cuDNN RNN training
                # mode for backward. These recurrent modules have no dropout.
                for module in model.predictor.modules():
                    if isinstance(module, torch.nn.RNNBase):
                        module.train()
            teacher = model(batch["reference"], batch["valid"], use_repair=False, use_correction=False)["base"]
        if clean_grad:
            clean_rul = F.mse_loss(teacher / scale, batch["target"] / scale)
    if consistency:
        available_fraction = result["mask"].sum((1, 2)) / (21 * batch["valid"].sum(1)).clamp_min(1)
        change = (torch.nan_to_num(batch["x"][..., 3:]) - batch["reference"][..., 3:]).abs()
        mild = (available_fraction >= .9) & ((change * available).sum((1, 2)) / available.sum((1, 2)).clamp_min(1) <= .05)
        if bool(mild.any()):
            cons = F.mse_loss(result["base"][mild] / scale, teacher[mild].detach() / scale)
    if (model.config["train_repair"] and model.config.get("bypass_rul_loss_weight", 0.) > 0 and
            any(p.requires_grad for p in model.predictor.parameters())):
        bypass = model(batch["x"], batch["valid"], use_repair=False, use_correction=False)
        bypass_rul = (F.mse_loss(bypass["base"] / scale, batch["target"] / scale) +
                      model.config["branch_loss_weight"] * branch_objective(model, bypass, batch["target"]))
    adv = result["base"].sum() * 0 if critic is None else -critic(result["repaired"], batch["reference"][..., :3], batch["valid"]).mean()
    terms = {"rul": rul, "reconstruction": rec, "generator_reconstruction": generator_rec,
             "identity": identity, "identity_tail": identity_tail, "imputation": imputation, "detection": detection, "trend": trend,
             "adversarial": adv, "consistency": cons, "branch": branch, "clean_rul": clean_rul,
             "bypass_rul": bypass_rul}
    if not model.config["train_repair"]:
        for name in ["reconstruction", "generator_reconstruction", "identity", "identity_tail", "imputation", "detection", "trend", "adversarial"]:
            terms[name] = rul * 0
    loss = sum(model.config.get(f"{name}_loss_weight", 0.) * value for name, value in terms.items())
    return loss, {**result, "loss_terms": {name: value.detach() for name, value in terms.items()}}


@contextmanager
def _unfused_rnn_backends():
    """Allow RNN second derivatives without changing backend precision flags."""
    cudnn_enabled = torch.backends.cudnn.enabled
    mkldnn_enabled = torch.backends.mkldnn.enabled
    try:
        torch.backends.cudnn.enabled = False
        torch.backends.mkldnn.enabled = False
        yield
    finally:
        torch.backends.mkldnn.enabled = mkldnn_enabled
        torch.backends.cudnn.enabled = cudnn_enabled


def critic_update(critic, optimizer, result, batch, config):
    real, fake = batch["reference"][..., 3:], result["repaired"].detach()
    settings, valid = batch["reference"][..., :3], batch["valid"]
    optimizer.zero_grad(set_to_none=True)
    for p in critic.parameters():
        p.requires_grad_(True)
    # GP needs expensive recurrent second derivatives. Estimate its expectation
    # on a random sub-batch; the Wasserstein term still uses the entire batch.
    count = min(len(real), int(config["gradient_penalty_batch_size"]))
    indices = torch.randperm(len(real), device=real.device)[:count]
    gp_real, gp_fake, gp_settings, gp_valid = real[indices], fake[indices], settings[indices], valid[indices]
    # Only the interpolation needs an unfused RNN for GP second derivatives.
    with _unfused_rnn_backends():
        alpha = torch.rand(count, 1, 1, device=real.device)
        interpolation = (alpha * gp_real + (1 - alpha) * gp_fake).requires_grad_(True)
        interpolated_score = critic(interpolation, gp_settings, gp_valid)
        gradient = torch.autograd.grad(interpolated_score.sum(), interpolation, create_graph=True)[0]
        norm = (gradient * gp_valid.unsqueeze(-1)).flatten(1).norm(2, 1)
        penalty = config["gradient_penalty"] * ((norm - 1) ** 2).mean()
        penalty.backward()
    wasserstein = critic(fake, settings, valid).mean() - critic(real, settings, valid).mean()
    if not torch.isfinite(wasserstein + penalty):
        raise FloatingPointError("Non-finite WGAN-GP critic loss.")
    wasserstein.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    for p in critic.parameters():
        p.requires_grad_(False)
    return float((wasserstein + penalty).detach())


@torch.no_grad()
def evaluate(model, dataset, config, fault_thresholds=None, *, capture_recovery_tails=False, **forward_options):
    model.eval()
    prediction, target, scores, labels, records = [], [], [], [], []
    rec_squared, rec_count, intact_squared, intact_count = 0., 0, 0., 0
    rec_absolute, intact_absolute, tp, fp, fn, tn = 0., 0., 0, 0, 0, 0
    linear_rec, linear_input, linear_intact = 0., 0., 0.
    channel_squared, channel_counts, window_rmse, window_intact_counts = np.zeros(21), np.zeros(21), [], []
    representation, compression_scale = transform_spec(getattr(dataset, "fault_profile", None))
    for batch in loader(dataset, config):
        batch = move_batch(batch, next(model.parameters()).device)
        result = model(batch["x"], batch["valid"], **forward_options)
        prediction.extend(result["prediction"].cpu().tolist()); target.extend(batch["target"].cpu().tolist())
        changed = batch["altered"] & batch["valid"].unsqueeze(-1)
        intact = ~batch["altered"] & batch["valid"].unsqueeze(-1)
        squared = (result["repaired"] - batch["reference"][..., 3:]).square()
        rec_squared += float((squared * changed).sum()); rec_count += int(changed.sum())
        intact_squared += float((squared * intact).sum()); intact_count += int(intact.sum())
        absolute = squared.sqrt()
        rec_absolute += float((absolute * changed).sum())
        intact_absolute += float((absolute * intact).sum())
        def linear(values):
            values = values.double()
            return compression_scale * torch.sinh(values / compression_scale) if representation == "asinh" else values
        linear_reference = linear(batch["reference"][..., 3:])
        linear_error = linear(result["repaired"]) - linear_reference
        input_error = linear(result["observed"]) - linear_reference
        linear_rec += float(linear_error.masked_fill(~changed, 0).square().sum())
        linear_input += float(input_error.masked_fill(~changed, 0).square().sum())
        linear_intact += float(linear_error.masked_fill(~intact, 0).square().sum())
        if capture_recovery_tails:
            healthy_squared = linear_error.masked_fill(~intact, 0).square()
            channel_squared += healthy_squared.sum((0, 1)).cpu().numpy()
            channel_counts += intact.sum((0, 1)).cpu().numpy()
            counts_per_window = intact.sum((1, 2))
            per_window = (healthy_squared.sum((1, 2)) / counts_per_window.clamp_min(1)).sqrt()
            window_rmse.extend(per_window.masked_fill(counts_per_window == 0, float("nan")).cpu().tolist())
            window_intact_counts.extend(counts_per_window.cpu().tolist())
        finite = result["mask"].bool() & batch["valid"].unsqueeze(-1)
        thresholds = result["fault_scores"].new_tensor(fault_thresholds if fault_thresholds is not None else [.5] * 21)
        flags = result["fault_scores"] >= thresholds
        if model.config.get("finite_repair_gate") == "validated_threshold":
            flags &= thresholds < 1
        truth = batch["altered"]
        tp += int((flags & truth & finite).sum()); fp += int((flags & ~truth & finite).sum())
        fn += int((~flags & truth & finite).sum()); tn += int((~flags & ~truth & finite).sum())
        scores.extend(result["fault_scores"][finite].cpu().tolist()); labels.extend(batch["altered"][finite].cpu().tolist())
        counts = changed.sum((1, 2)).cpu().tolist()
        totals = (21 * batch["valid"].sum(1)).cpu().tolist()
        records.extend({"unit": int(u), "cycle": int(t), "prediction": float(p), "target": float(y), "raw_target": float(r),
                        "altered_entries": int(c), "valid_sensor_entries": int(n), "active_fault": bool(c)}
                       for u, t, p, y, r, c, n in zip(batch["unit"], batch["cycle"], result["prediction"], batch["target"], batch["raw_target"], counts, totals))
    report = {**metrics(prediction, target), "reconstruction_RMSE": (rec_squared / rec_count) ** .5 if rec_count else None,
              "reconstruction_MAE": rec_absolute / rec_count if rec_count else None,
              "intact_repair_RMSE": (intact_squared / intact_count) ** .5 if intact_count else 0.,
              "intact_repair_MAE": intact_absolute / intact_count if intact_count else 0.,
              "sensor_representation": representation,
              "linear_reconstruction_RMSE": (linear_rec / rec_count) ** .5 if rec_count else None,
              "linear_input_RMSE": (linear_input / rec_count) ** .5 if rec_count else None,
              "linear_intact_repair_RMSE": (linear_intact / intact_count) ** .5 if intact_count else 0.,
              "linear_reconstruction_ratio": ((linear_rec / linear_input) ** .5 if linear_input > 0 else
                                               0. if linear_rec == 0 else float("inf")),
              "fault_precision": tp / (tp + fp) if tp + fp else None,
              "fault_recall": tp / (tp + fn) if tp + fn else None,
              "fault_false_positive_rate": fp / (fp + tn) if fp + tn else None,
              "fault_F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
              "fault_threshold_scope": "validation-selected" if fault_thresholds is not None else "default 0.5",
              "fault_PR_AUC": float(average_precision_score(labels, scores)) if any(labels) and not all(labels) else None}
    records = pd.DataFrame(records)
    if capture_recovery_tails:
        records["linear_intact_repair_RMSE"] = window_rmse
        records["intact_sensor_entries"] = window_intact_counts
        supported = np.asarray(window_rmse)[np.asarray(window_intact_counts) > 0]
        channels = channel_counts > 0
        report.update({
            "linear_intact_channel_RMSE_max": float(np.sqrt(channel_squared[channels] / channel_counts[channels]).max()) if channels.any() else None,
            "linear_intact_window_N": len(supported),
            "linear_intact_window_RMSE_p95": float(np.quantile(supported, .95)) if len(supported) else None,
            "linear_intact_window_RMSE_p99": float(np.quantile(supported, .99)) if len(supported) else None,
            "linear_intact_window_RMSE_max": float(supported.max()) if len(supported) else None,
        })
    report.update(exposure_metrics(records))
    return report, records


def exposure_metrics(records, target=None):
    """Separate observed damage from a scenario's nominal fault/severity label."""
    active = records.active_fault.to_numpy(bool)
    target = records.target.to_numpy() if target is None else np.asarray(target)
    report = {"affected_endpoints": int(active.sum()), "affected_endpoint_fraction": float(active.mean()),
              "damaged_entry_fraction": int(records.altered_entries.sum()) / max(1, int(records.valid_sensor_entries.sum()))}
    for label, mask in [("active_fault", active), ("unaffected", ~active)]:
        report[f"{label}_N"] = int(mask.sum())
        measured = metrics(records.prediction.to_numpy()[mask], target[mask]) if mask.any() else {}
        for key in ["RMSE", "MAE", "NASA_Score"]:
            report[f"{label}_{key}"] = measured.get(key)
    return report


def corrupt_validation_rmse(report):
    active = report.get("active_fault_RMSE")
    return report["RMSE"] if active is None else active


def validation_datasets(frame, scaler, config):
    """Build fixed, training-held selection contexts; mixed keeps its guards."""
    seed = config["validation_seed"]
    clean = WindowDataset(frame, scaler, config, mode="validation", seed=seed, family="clean")
    mixed = WindowDataset(frame, scaler, config, mode="validation", seed=seed, family="mixed")
    contexts = None
    if config.get("rul_selection", "mixed") == "fault_macro":
        contexts = {family: WindowDataset(frame, scaler, config, mode="validation", seed=seed, family=family)
                    for family in FAULTS}
        contexts["mixed"] = mixed
    return clean, mixed, contexts


def validation_score(model, clean_dataset, corrupt_dataset, config, *, selection_datasets=None, **options):
    clean, _ = evaluate(model, clean_dataset, config, **options)
    damaged, _ = evaluate(model, corrupt_dataset, config, **options)
    fault_score = corrupt_validation_rmse(damaged)
    if config.get("rul_selection", "mixed") == "fault_macro":
        if not isinstance(selection_datasets, dict) or set(selection_datasets) != {*FAULTS, "mixed"}:
            raise ValueError("Balanced RUL selection requires every declared fault family and mixed context.")
        scores = {"mixed": fault_score}
        for family in FAULTS:
            report, _ = evaluate(model, selection_datasets[family], config, **options)
            scores[family] = corrupt_validation_rmse(report)
        fault_score = float(np.mean(list(scores.values())))
        damaged = {**damaged, "selection_family_RMSE": scores, "selection_fault_macro_RMSE": fault_score}
    score = .5 * (clean["RMSE"] + fault_score)
    if not np.isfinite(score):
        raise FloatingPointError("Non-finite clean/corrupted validation RMSE.")
    return score, clean, damaged


def validate_training_config(config):
    transform_spec(config)
    for name, choices, default in [
            ("identity_tail_penalty", {"mse", "log_mse"}, "mse"),
            ("recovery_sampling", {"random", "balanced"}, "random"),
            ("clean_pretraining_selection", {"mixed_score", "clean_rmse"}, "mixed_score"),
            ("rul_selection", {"mixed", "fault_macro"}, "mixed")]:
        value = config.get(name, default)
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"Unsupported {name}.")
    levels = config.get("recovery_train_severities", [.1, .2, .3])
    if (not isinstance(levels, (list, tuple)) or not levels or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or not 0 < x <= .5 for x in levels) or
            len(set(levels)) != len(levels)):
        raise ValueError("recovery_train_severities must contain distinct finite levels in (0, .5].")
    if config.get("repair_loss", "mse") not in {"mse", "smooth_l1"}:
        raise ValueError("repair_loss must be mse or smooth_l1.")
    if config.get("detector_loss", "weighted") not in {"weighted", "balanced"}:
        raise ValueError("detector_loss must be weighted or balanced.")
    if not isinstance(config.get("validate_bypass", False), bool):
        raise ValueError("validate_bypass must be a boolean.")
    if config.get("identity_loss_space", "encoded") not in {"encoded", "original_linear"}:
        raise ValueError("identity_loss_space must be encoded or original_linear.")
    if config.get("recovery_safety", "none") not in {"none", "fitted_support"}:
        raise ValueError("recovery_safety must be none or fitted_support.")
    if not 0 < config.get("identity_tail_fraction", .1) <= 1:
        raise ValueError("identity_tail_fraction must be in (0, 1].")
    if not isinstance(config.get("recovery_validation", False), bool):
        raise ValueError("recovery_validation must be a boolean.")
    if config.get("recovery_validation", False):
        levels = config.get("recovery_validation_severities", [])
        if (not isinstance(levels, (list, tuple)) or not levels or len(set(levels)) != len(levels) or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or not 0 < x <= 1 for x in levels)):
            raise ValueError("recovery_validation_severities must contain distinct levels in (0, 1].")
        for key in ["recovery_validation_max_channel_intact_rmse", "recovery_validation_max_window_intact_rmse", "recovery_validation_max_reconstruction_rmse"]:
            value = config.get(key)
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value < 0):
                raise ValueError(f"{key} must be None or finite and nonnegative.")
    if config.get("finite_repair_gate", "continuous") not in {"continuous", "validated_threshold"}:
        raise ValueError("Unknown finite_repair_gate.")
    if config.get("detector_training", "joint") not in {"joint", "separate"}:
        raise ValueError("detector_training must be joint or separate.")
    if not isinstance(config.get("detach_repair_gate", False), bool):
        raise ValueError("detach_repair_gate must be a boolean.")
    if config.get("finite_repair_gate") == "validated_threshold" and config.get("detector_training") != "separate":
        raise ValueError("Validated repair thresholds require a separately fitted, frozen detector.")
    if config.get("calibration_policy", "engine_prefix_max") not in {"engine_prefix_max", "random_prefix"}:
        raise ValueError("Unknown calibration_policy.")
    if not np.isfinite(config["interval_alpha"]) or not 0 < config["interval_alpha"] < 1:
        raise ValueError("interval_alpha must be strictly between zero and one.")
    if config.get("detector_training") == "separate":
        for name in ["detector_refinement_epochs", "detector_refinement_patience"]:
            if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 <= config["detector_max_false_positive_rate"] < 1:
            raise ValueError("detector_max_false_positive_rate must be in [0, 1).")
        for name in ["detector_refinement_learning_rate", "detector_refinement_min_improvement"]:
            if not np.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f"{name} must be finite and positive.")
    if not np.isfinite(config.get("repair_huber_beta", 1.)) or config.get("repair_huber_beta", 1.) <= 0:
        raise ValueError("repair_huber_beta must be finite and positive.")
    for name in ["predictor_epochs", "generator_epochs", "joint_epochs", "correction_epochs",
                 "patience", "batch_size", "gradient_penalty_batch_size", "critic_steps",
                 "validation_prefixes", "correction_prefixes", "window_stride", "lr_patience", "bootstrap_samples", "drift_ramp_cycles",
                 "block_gap_cycles", "curriculum_epochs", "sequence_length", "local_length", "background_size",
                 "explanation_examples", "explanation_repeats", "explanation_samples", "transformer_dim", "transformer_heads",
                 "transformer_layers", "generator_hidden", "generator_layers", "cnn_channels", "gru_hidden", "oof_folds"]:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    horizon = config["fault_start_horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("fault_start_horizon must be a nonnegative integer.")
    burst = config["intermittent_mean_burst"]
    if isinstance(burst, bool) or not isinstance(burst, int) or burst < 2:
        raise ValueError("intermittent_mean_burst must be an integer >= 2.")
    if not np.isfinite(config["fault_noise_autocorrelation"]) or not 0 <= config["fault_noise_autocorrelation"] < 1:
        raise ValueError("fault_noise_autocorrelation must be finite and in [0, 1).")
    if not np.isfinite(config["fault_transition_gain"]) or not 0 <= config["fault_transition_gain"] <= 3:
        raise ValueError("fault_transition_gain must be finite and in [0, 3].")
    for name in [k for k in config if k.endswith("_loss_weight")] + [name for name in ["max_clean_rmse_increase", "max_clean_nasa_increase", "max_intact_repair_rmse", "max_reconstruction_rmse", "max_linear_reconstruction_ratio"] if name in config]:
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if config["reconstruction_loss_weight"] <= 0 or config["rul_loss_weight"] <= 0:
        raise ValueError("RUL and reconstruction weights must stay positive.")
    if not 0 < config["lr_factor"] < 1 or config["minimum_learning_rate"] <= 0:
        raise ValueError("Invalid learning-rate scheduler settings.")
    if not config["correction_bounds"] or any(not np.isfinite(b) or b <= 0 for b in config["correction_bounds"]):
        raise ValueError("correction_bounds must contain finite positive bounds.")
    for name in ["learning_rate", "repair_learning_rate", "joint_learning_rate", "minimum_learning_rate",
                 "correction_bound", "detector_positive_weight_cap", "target_scale", "train_rul_cap"]:
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if config["detector_positive_weight_cap"] < 1:
        raise ValueError("detector_positive_weight_cap must be at least one.")
    if config["oof_folds"] < 2:
        raise ValueError("oof_folds must be at least two.")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must contain distinct nonnegative integers.")
    from .config import SUBSETS
    if not config["subsets"] or len(set(config["subsets"])) != len(config["subsets"]) or set(config["subsets"]) - set(SUBSETS):
        raise ValueError("subsets must contain distinct supported C-MAPSS subsets.")
    if config["predictor_family"] not in {"gated", "cnn_gru", "cnn_lstm", "transformer"}:
        raise ValueError("Unsupported predictor_family.")
    if config["predictor_branch"] not in {"both", "local", "global"}:
        raise ValueError("Unsupported predictor_branch.")
    if config["local_length"] > config["sequence_length"] or not 0 <= config["clean_fraction"] < 1:
        raise ValueError("Invalid local window length or clean fraction.")
    for name in ["train_repair", "train_corrupted_predictor", "equal_fusion", "benchmark_tune"]:
        if not isinstance(config[name], bool):
            raise ValueError(f"{name} must be a boolean.")
    if config["transformer_dim"] % 2 or config["transformer_dim"] % config["transformer_heads"]:
        raise ValueError("Transformer dimension must be even and divisible by its head count.")


def checkpoint_guards(clean, damaged, clean_reference, config, *, use_repair=True, clean_nasa_reference=None):
    guards = {"clean_rmse": clean["RMSE"] <= clean_reference * (1 + config["max_clean_rmse_increase"])}
    # The asymmetric score can worsen while RMSE improves. Guard the original
    # clean validation reference across every subsequent stage, including OOF
    # correction; never spend a fresh tolerance at each stage.
    if clean_nasa_reference is not None and np.isfinite(clean_nasa_reference) and "max_clean_nasa_increase" in config:
        score = clean.get("NASA_Score_Mean", float("inf"))
        guards["clean_nasa"] = bool(np.isfinite(score) and score <= clean_nasa_reference * (1 + config["max_clean_nasa_increase"]) + 1e-12)
    if config["train_repair"] and use_repair:
        guards.update(
            intact=damaged.get("linear_intact_repair_RMSE", damaged.get("intact_repair_RMSE", 0.)) <= config["max_intact_repair_rmse"],
            linear_reconstruction=damaged.get("linear_reconstruction_ratio", 1.) <= config["max_linear_reconstruction_ratio"],
            reconstruction=(damaged.get("reconstruction_RMSE") or 0.) <= config["max_reconstruction_rmse"])
    return {key: bool(value) for key, value in guards.items()}


def acceptable_checkpoint(clean, damaged, clean_reference, config, *, use_repair=True, clean_nasa_reference=None):
    return all(checkpoint_guards(clean, damaged, clean_reference, config,
        use_repair=use_repair, clean_nasa_reference=clean_nasa_reference).values())


def learning_rate_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config["lr_factor"],
        patience=config["lr_patience"], min_lr=config["minimum_learning_rate"])


class _Progress:
    def __init__(self, prefix, config, enabled=True):
        self.prefix, self.enabled = prefix, enabled
        self.interval = max(0., float(config.get("progress_interval_seconds", 30.)))
        self.last = time.perf_counter()

    def emit(self, message, force=False):
        now = time.perf_counter()
        if self.enabled and (force or now - self.last >= self.interval):
            print(f"{self.prefix} {message}", flush=True)
            self.last = now


def fit_pipeline(train_frame, validation_frame, subset, config, seed, verbose=True, *, checkpoint_path=None, progress_label=None):
    validate_training_config(config)
    seed_all(seed, config["cpu_threads"])
    device = device_for(config)
    progress = _Progress(progress_label or f"{subset} seed={seed}", config, verbose)
    identity = pipeline_identity(train_frame, validation_frame, subset, config, seed) if checkpoint_path else None
    checkpoint = load_checkpoint(checkpoint_path, identity)
    scaler = OperatingRegimeScaler().fit(train_frame, subset, config["split_seed"],
        sensor_transform=config.get("sensor_transform", "linear"), sensor_transform_scale=config.get("sensor_transform_scale", 1.))
    model, critic = GraceRUL(config).to(device), TemporalCritic(config).to(device)
    if config.get("recovery_safety") == "fitted_support":
        from .recovery_safety import fit_recovery_envelope
        model.set_recovery_envelope(fit_recovery_envelope(train_frame, scaler))
    train = WindowDataset(train_frame, scaler, config, seed=seed)
    clean_validation, corrupt_validation, selection_contexts = validation_datasets(validation_frame, scaler, config)
    selection_options = {"selection_datasets": selection_contexts} if selection_contexts is not None else {}
    history, best_score, clean_reference = [], float("inf"), float("inf")
    warm_recovery, detector_audit, detector_history = None, None, []
    selected_recovery_audit, recovery_audits, recovery_matrix = None, [], None
    clean_nasa_reference = float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        critic.load_state_dict(checkpoint["critic"])
        history, best_score = checkpoint["history"], checkpoint["best_score"]
        clean_reference = checkpoint["clean_reference"]
        clean_nasa_reference = checkpoint.get("clean_nasa_reference", float("inf"))
        warm_recovery = checkpoint.get("warm_recovery")
        detector_audit, detector_history = checkpoint.get("detector_audit"), checkpoint.get("detector_history", [])
        selected_recovery_audit = checkpoint.get("selected_recovery_audit")
        recovery_audits = checkpoint.get("recovery_audits", [])
        restore_random_state(checkpoint["random_state"])
        progress.emit(f"resuming checkpoint: {checkpoint['stage']} epoch {checkpoint['next_epoch']} completed", force=True)
    stages = ["predictor"]
    if config["train_repair"]:
        stages.append("generator")
    if config["train_corrupted_predictor"]:
        stages.append("joint")
    for stage_index, stage in enumerate(stages):
        if checkpoint and (stage_index < checkpoint["stage_index"] or
                           (stage_index == checkpoint["stage_index"] and checkpoint["stage_complete"])):
            continue
        if stage == "generator" and config.get("detector_training") == "separate" and not (
                checkpoint and stage_index == checkpoint["stage_index"]):
            from .detection import refine_bypassed_detector
            progress.emit("fitting finite-fault detector independently of RUL/recovery gradients", force=True)
            previous_repair = bool(model.repair_enabled)
            model.repair_enabled.fill_(False)
            detector_checkpoint = checkpoint_path.with_name(checkpoint_path.stem + "_detector.pt") if checkpoint_path else None
            try:
                refined = refine_bypassed_detector(model, train_frame, validation_frame, scaler, config,
                    seed + 7001, checkpoint_path=detector_checkpoint)
                detector_audit, detector_history = refined.attrs["audit"], refined.to_dict("records")
                if config.get("finite_repair_gate") == "validated_threshold":
                    model.finite_repair_thresholds.copy_(torch.tensor(detector_audit["thresholds"], device=device))
            finally:
                model.repair_enabled.fill_(previous_repair)
        for name, parameter in model.named_parameters():
            if stage == "predictor" or not config["train_repair"]:
                enabled = name.startswith("predictor")
            elif stage == "generator":
                enabled = name.startswith(("detector", "generator"))
            else:
                enabled = not name.startswith("corrector")
            if config.get("detector_training") == "separate" and name.startswith("detector"):
                enabled = False
            parameter.requires_grad_(enabled)
        model.zero_grad(set_to_none=True)
        lr = config["learning_rate"] if stage == "predictor" else config["repair_learning_rate"] if stage == "generator" else config["joint_learning_rate"]
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=.01)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config["repair_learning_rate"], betas=(.5, .999))
        scheduler = learning_rate_scheduler(optimizer, config)
        critic_scheduler = learning_rate_scheduler(critic_optimizer, config)
        # The clean predictor remains a deployment-compatible fallback when
        # learned repair cannot improve robustness within the declared limits.
        best = copy.deepcopy(model.state_dict()) if stage != "predictor" else None
        if stage == "generator":
            best["repair_enabled"] = torch.tensor(False, device=device)
        best_score = best_score if stage != "predictor" else float("inf")
        best_selection_score = best_score
        stale, start_epoch = 0, 0
        # Keep the deployable fallback in `best`, but continue optimizing the
        # detector/generator learned in their dedicated phase. Rejection is an
        # export decision, not a reason to restart joint recovery from zero.
        if stage == "joint" and warm_recovery and not (checkpoint and stage_index == checkpoint["stage_index"]):
            model.load_state_dict({**model.state_dict(), **warm_recovery})
        if checkpoint and stage_index == checkpoint["stage_index"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            critic_scheduler.load_state_dict(checkpoint["critic_scheduler"])
            best_score, best, stale = checkpoint["best_score"], checkpoint["best"], checkpoint["stale"]
            best_selection_score = checkpoint.get("best_selection_score", best_score)
            start_epoch = checkpoint["next_epoch"]
        total_epochs = config[f"{stage}_epochs"]
        for epoch in range(start_epoch, total_epochs):
            started = time.perf_counter()
            model.train(); train.set_epoch(epoch, min(1., (epoch + 1) / min(config["curriculum_epochs"], total_epochs)))
            if config.get("detector_training") == "separate":
                model.detector.eval()
            model.repair_enabled.fill_(config["train_repair"])
            if stage == "generator":
                model.predictor.eval()
                # cuDNN GRU backward to generator inputs needs training mode;
                # this GRU has no dropout. Predictor dropout stays disabled.
                for module in model.predictor.modules():
                    if isinstance(module, torch.nn.RNNBase):
                        module.train()
            losses = []
            adversarial = config["train_repair"] and stage != "predictor" and epoch >= config["adversarial_warmup"] and config["adversarial_loss_weight"] > 0
            term_totals = {}
            batches = loader(train, config, shuffle=True, seed=seed + epoch)
            total_batches = min(len(batches), config.get("maximum_batches") or len(batches))
            label = f"{stage} {epoch + 1}/{total_epochs}"
            progress.emit(f"{label}: training {total_batches} batches" + (" with WGAN-GP critic" if adversarial else ""), force=True)
            for step, batch in enumerate(batches):
                if config.get("maximum_batches") and step >= config["maximum_batches"]:
                    break
                batch = move_batch(batch, device)
                if adversarial:
                    progress.emit(f"{label}: batch {step + 1}/{total_batches}, critic gradient penalty")
                    with torch.no_grad():
                        initial = model(batch["x"], batch["valid"], use_correction=False)
                    for _ in range(config["critic_steps"]):
                        critic_update(critic, critic_optimizer, initial, batch, config)
                optimizer.zero_grad(set_to_none=True)
                if stage == "predictor":
                    result = model(batch["reference"], batch["valid"], use_repair=False, use_correction=False)
                    scale = config["target_scale"]
                    loss = config["rul_loss_weight"] * F.mse_loss(result["base"] / scale, batch["target"] / scale) + config["branch_loss_weight"] * branch_objective(model, result, batch["target"])
                else:
                    loss, result = objectives(model, batch, critic if adversarial else None)
                    for name, value in result["loss_terms"].items():
                        term_totals[name] = term_totals.get(name, 0.) + float(value)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {stage} training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                progress.emit(f"{label}: batch {step + 1}/{total_batches} complete, loss={losses[-1]:.4f}, elapsed={time.perf_counter() - started:.1f}s",
                              force=step == 0)
            progress.emit(f"{label}: validating clean and corrupted engines", force=True)
            score, clean, damaged = validation_score(model, clean_validation, corrupt_validation, config,
                                                     use_repair=stage != "predictor", use_correction=False, **selection_options)
            repair_guards = checkpoint_guards(clean, damaged, clean_reference, config, clean_nasa_reference=clean_nasa_reference)
            accepted = stage == "predictor" or all(repair_guards.values())
            selected_repair = stage != "predictor" and config["train_repair"]
            candidate_recovery_audit = None
            matrix_status = "not_applicable"
            if selected_repair and config.get("recovery_validation", False):
                if accepted and score < best_score:
                    from .recovery_validation import RecoveryValidationMatrix, audit_recovery_candidate
                    if recovery_matrix is None:
                        recovery_matrix = RecoveryValidationMatrix(validation_frame, scaler, config, seed=config["validation_seed"])
                    progress.emit(f"{label}: checking recovery across {len(recovery_matrix.datasets)} held-validation conditions", force=True)
                    candidate_recovery_audit = audit_recovery_candidate(model, recovery_matrix, config,
                        use_repair=True, stop_on_failure=True)
                    accepted = candidate_recovery_audit["complete"] and candidate_recovery_audit["accepted"]
                    matrix_status = "passed" if accepted else "rejected"
                    recovery_audits.append({"stage": stage, "epoch": epoch + 1,
                        "validation_macro_RMSE": score, "audit": candidate_recovery_audit})
                else:
                    matrix_status = "not_improving" if accepted else "primary_guards_rejected"
                    accepted = False
            candidate_metrics = {"repair_validation_macro_RMSE": score,
                                 "repair_checkpoint_accepted": accepted,
                                 "repair_matrix_status": matrix_status,
                                 "repair_matrix_complete": candidate_recovery_audit["complete"] if candidate_recovery_audit else False,
                                 "repair_matrix_worst_intact_RMSE": candidate_recovery_audit["worst"].get("linear_intact_repair_RMSE") if candidate_recovery_audit else None,
                                 "repair_validation_clean_RMSE": clean["RMSE"],
                                 "repair_validation_clean_NASA_Score_Mean": clean.get("NASA_Score_Mean"),
                                 "repair_validation_corrupt_NASA_Score_Mean": damaged.get("NASA_Score_Mean"),
                                 "repair_validation_linear_intact_repair_RMSE": damaged.get("linear_intact_repair_RMSE"),
                                 **{f"repair_guard_{name}": passed for name, passed in repair_guards.items()},
                                 "repair_validation_reconstruction_RMSE": damaged.get("reconstruction_RMSE"),
                                 "repair_validation_linear_reconstruction_RMSE": damaged.get("linear_reconstruction_RMSE"),
                                 "repair_validation_linear_reconstruction_ratio": damaged.get("linear_reconstruction_ratio")}
            if stage == "joint" and config["train_repair"] and config.get("validate_bypass", False):
                bypass_score, bypass_clean, bypass_damaged = validation_score(
                    model, clean_validation, corrupt_validation, config, use_repair=False, use_correction=False, **selection_options)
                bypass_guards = checkpoint_guards(bypass_clean, bypass_damaged, clean_reference, config,
                    use_repair=False, clean_nasa_reference=clean_nasa_reference)
                bypass_accepted = all(bypass_guards.values())
                candidate_metrics.update(bypass_validation_macro_RMSE=bypass_score,
                    bypass_checkpoint_accepted=bypass_accepted,
                    bypass_validation_clean_RMSE=bypass_clean["RMSE"],
                    bypass_validation_clean_NASA_Score_Mean=bypass_clean.get("NASA_Score_Mean"),
                    **{f"bypass_guard_{name}": passed for name, passed in bypass_guards.items()})
                if bypass_accepted and (not accepted or bypass_score < score):
                    score, clean, damaged, accepted = bypass_score, bypass_clean, bypass_damaged, True
                    selected_repair = False
            clean_anchor = stage == "predictor" and config.get("clean_pretraining_selection", "mixed_score") == "clean_rmse"
            selection_score = clean["RMSE"] if clean_anchor else score
            checkpoint_selected = accepted and selection_score < best_selection_score
            row = {"subset": subset, "seed": seed, "stage": stage, "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                   "validation_macro_RMSE": score, "validation_clean_RMSE": clean["RMSE"],
                   "validation_clean_NASA_Score_Mean": clean.get("NASA_Score_Mean"),
                   "validation_corrupt_NASA_Score_Mean": damaged.get("NASA_Score_Mean"),
                   "validation_corrupt_RMSE": corrupt_validation_rmse(damaged), "validation_corrupt_all_RMSE": damaged["RMSE"],
                   "validation_affected_endpoint_fraction": damaged.get("affected_endpoint_fraction"), "validation_reconstruction_RMSE": damaged["reconstruction_RMSE"],
                   "validation_intact_repair_RMSE": damaged.get("intact_repair_RMSE", 0.),
                   "validation_linear_reconstruction_RMSE": damaged.get("linear_reconstruction_RMSE"),
                   "validation_linear_intact_repair_RMSE": damaged.get("linear_intact_repair_RMSE"),
                   "validation_linear_reconstruction_ratio": damaged.get("linear_reconstruction_ratio"),
                   "validation_path": "repair" if selected_repair else "bypass",
                   "sensor_representation": scaler.state["sensor_transform"],
                   "checkpoint_accepted": accepted, "learning_rate": optimizer.param_groups[0]["lr"],
                   "checkpoint_selected": checkpoint_selected, "selection_score": selection_score,
                   "selection_policy": "clean_rmse" if clean_anchor else config.get("rul_selection", "mixed"),
                   "validation_fault_macro_RMSE": damaged.get("selection_fault_macro_RMSE"),
                   **{f"validation_{family}_selection_RMSE": value for family, value in damaged.get("selection_family_RMSE", {}).items()},
                   "duration_seconds": time.perf_counter() - started,
                   **candidate_metrics,
                   **{f"loss_{name}": value / len(losses) for name, value in term_totals.items()}}
            history.append(row)
            progress.emit(f"{stage} {epoch + 1}: loss={row['train_loss']:.4f}, validation RMSE={score:.3f}, elapsed={row['duration_seconds']:.1f}s", force=True)
            if checkpoint_selected:
                best_score, best, stale = score, copy.deepcopy(model.state_dict()), 0
                best_selection_score = selection_score
                if stage != "predictor":
                    selected_recovery_audit = candidate_recovery_audit if selected_repair else None
                if stage != "predictor":
                    best["repair_enabled"] = torch.tensor(selected_repair, device=device)
                if stage == "predictor":
                    clean_reference = clean["RMSE"]
                    clean_nasa_reference = clean.get("NASA_Score_Mean", float("inf"))
            else:
                stale += 1
            scheduler.step(selection_score)
            critic_scheduler.step(selection_score)
            stage_complete = stale >= config["patience"] or epoch + 1 == total_epochs
            if stage_complete:
                if stage == "generator" and not bool(best["repair_enabled"]):
                    warm_recovery = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()
                                     if name.startswith(("detector.", "generator."))}
                model.load_state_dict(best)
            if checkpoint_path:
                save_checkpoint(checkpoint_path, {"identity": identity, "stage": stage, "stage_index": stage_index,
                    "next_epoch": epoch + 1, "stage_complete": stage_complete, "model": model.state_dict(),
                    "critic": critic.state_dict(), "optimizer": optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "critic_scheduler": critic_scheduler.state_dict(),
                    "clean_reference": clean_reference,
                    "clean_nasa_reference": clean_nasa_reference,
                    "detector_audit": detector_audit, "detector_history": detector_history,
                    "selected_recovery_audit": selected_recovery_audit, "recovery_audits": recovery_audits,
                    "warm_recovery": warm_recovery,
                    "best": best, "best_score": best_score, "best_selection_score": best_selection_score,
                    "stale": stale, "history": history, "random_state": random_state()})
            if stale >= config["patience"]:
                progress.emit(f"{stage}: early stopping after {stale} epochs without validation improvement; restoring best weights", force=True)
                break
        model.load_state_dict(best)
    if not config["train_repair"]:
        model.repair_enabled.fill_(False)
    for p in model.parameters():
        p.requires_grad_(True)
    model.pretrained_clean_rmse = float(clean_reference)
    model.pretrained_clean_nasa = float(clean_nasa_reference)
    model.detector_audit, model.detector_history = detector_audit, pd.DataFrame(detector_history)
    model.recovery_validation_audit, model.recovery_validation_history = selected_recovery_audit, recovery_audits
    return model.eval(), scaler, pd.DataFrame(history), best_score


def fit_corrector(model, fit_frame, validation_frame, subset, config, seed, *, checkpoint_dir=None):
    """Upstream temporary models never fit or select on their OOF engines."""
    units = np.sort(fit_frame.unit.unique())
    features, bases, targets, audit, cv_reports = [], [], [], [], []
    folds = KFold(config["oof_folds"], shuffle=True, random_state=config["split_seed"])
    progress = _Progress(f"{subset} seed={seed}", config)
    for fold, (train_indices, held_indices) in enumerate(folds.split(units)):
        label = f"{subset} seed={seed} OOF {fold + 1}/{config['oof_folds']}"
        progress.emit(f"OOF fold {fold + 1}/{config['oof_folds']}: training temporary upstream model", force=True)
        available = np.random.default_rng(config["split_seed"] + fold).permutation(units[train_indices])
        ninner = max(1, len(available) // 5)
        inner_val, inner_fit = available[:ninner], available[ninner:]
        held = units[held_indices]
        checkpoint_path = checkpoint_dir / f"oof_{subset}_{seed}_fold{fold + 1}.pt" if checkpoint_dir else None
        fold_model, fold_scaler, _, _ = fit_pipeline(fit_frame[fit_frame.unit.isin(inner_fit)], fit_frame[fit_frame.unit.isin(inner_val)],
            subset, config, seed + 100 + fold, checkpoint_path=checkpoint_path, progress_label=label)
        assert not (set(fold_scaler.state["fit_units"]) & set(held))
        # Report independent held-engine predictions before the final corrector
        # learns from these OOF targets. This is upstream CV, not corrected-model CV.
        for family in ["clean", "mixed"]:
            progress.emit(f"OOF fold {fold + 1}/{config['oof_folds']}: evaluating held-out {family} engines", force=True)
            held_dataset = WindowDataset(fit_frame[fit_frame.unit.isin(held)], fold_scaler, config,
                                         mode="validation", seed=seed + fold, family=family)
            cv_report, _ = evaluate(fold_model, held_dataset, config, use_correction=False)
            cv_reports.append({"subset": subset, "seed": seed, "fold": fold + 1, "family": family,
                               "scope": "engine-held-out detector/recovery/predictor; final corrector excluded", **cv_report})
        progress.emit(f"OOF fold {fold + 1}/{config['oof_folds']}: collecting correction features", force=True)
        with torch.no_grad():
            for family in ["clean", "mixed"]:
                dataset = WindowDataset(fit_frame[fit_frame.unit.isin(held)], fold_scaler, config,
                                        mode="correction", seed=seed + fold, family=family)
                for batch in loader(dataset, config):
                    batch = move_batch(batch, device_for(config))
                    out = fold_model(batch["x"], batch["valid"], use_correction=False)
                    features.append(out["correction_features"].cpu()); bases.append(out["base"].cpu()); targets.append(batch["target"].cpu())
        audit.append({"fold": fold, "fit_units": fold_scaler.state["fit_units"], "inner_validation_units": [int(i) for i in inner_val], "held_out_units": [int(i) for i in held]})
        print(f"{subset}: out-of-fold correction features {fold + 1}/{config['oof_folds']}", flush=True)
        del fold_model
    x, base, y = torch.cat(features).to(device_for(config)), torch.cat(bases).to(device_for(config)), torch.cat(targets).to(device_for(config))
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.corrector.parameters():
        p.requires_grad_(True)
    # The final model's scaler is attached explicitly by the caller.
    clean_val, val, selection_contexts = validation_datasets(validation_frame, model.scaler, config)
    selection_options = {"selection_datasets": selection_contexts} if selection_contexts is not None else {}
    best_score, clean, damaged = validation_score(model, clean_val, val, config, use_correction=False, **selection_options)
    # Keep the final pipeline within the same original clean-predictor guard;
    # successive stages must not each spend another copy of the tolerance.
    clean_reference = min(clean["RMSE"], getattr(model, "pretrained_clean_rmse", clean["RMSE"]))
    clean_nasa_reference = min(clean.get("NASA_Score_Mean", float("inf")),
        getattr(model, "pretrained_clean_nasa", float("inf")))
    # Retain a zero-correction candidate if every trained candidate is worse.
    torch.nn.init.zeros_(model.corrector[-1].weight)
    torch.nn.init.zeros_(model.corrector[-1].bias)
    initial = copy.deepcopy(model.corrector.state_dict())
    best, best_bound = copy.deepcopy(initial), float(model.correction_bound)
    history = [{"subset": subset, "seed": seed, "epoch": 0, "train_loss": None,
                "validation_macro_RMSE": best_score, "validation_clean_RMSE": clean["RMSE"],
                "validation_clean_NASA_Score_Mean": clean.get("NASA_Score_Mean"),
                "validation_corrupt_RMSE": corrupt_validation_rmse(damaged), "checkpoint_accepted": True,
                "checkpoint_selected": True, "selection_policy": config.get("rul_selection", "mixed"),
                "validation_fault_macro_RMSE": damaged.get("selection_fault_macro_RMSE"),
                **{f"validation_{family}_selection_RMSE": value for family, value in damaged.get("selection_family_RMSE", {}).items()},
                "learning_rate": config["learning_rate"], "correction_bound": best_bound}]
    for bound in config["correction_bounds"]:
        model.corrector.load_state_dict(initial)
        model.correction_bound.fill_(bound)
        optimizer = torch.optim.AdamW(model.corrector.parameters(), lr=config["learning_rate"])
        scheduler = learning_rate_scheduler(optimizer, config)
        trial_best, stale = history[0]["validation_macro_RMSE"], 0
        progress.emit(f"fitting residual corrector: bound={bound}, up to {config['correction_epochs']} epochs", force=True)
        for epoch in range(config["correction_epochs"]):
            optimizer.zero_grad(set_to_none=True)
            delta = model.correction_bound * model.corrector(x).tanh().flatten()
            loss = F.mse_loss((base + delta).clamp_min(0) / config["target_scale"], y / config["target_scale"]) + config["correction_penalty"] * (delta / config["target_scale"]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite residual correction loss.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.corrector.parameters(), 5., error_if_nonfinite=True); optimizer.step()
            score, clean, damaged = validation_score(model, clean_val, val, config, **selection_options)
            guards = checkpoint_guards(clean, damaged, clean_reference, config,
                use_repair=False, clean_nasa_reference=clean_nasa_reference)
            accepted = all(guards.values())
            history.append({"subset": subset, "seed": seed, "epoch": epoch + 1, "train_loss": float(loss.detach()),
                            "validation_macro_RMSE": score, "validation_clean_RMSE": clean["RMSE"],
                            "validation_clean_NASA_Score_Mean": clean.get("NASA_Score_Mean"),
                            **{f"guard_{name}": passed for name, passed in guards.items()},
                            "validation_corrupt_RMSE": corrupt_validation_rmse(damaged), "checkpoint_accepted": accepted,
                            "checkpoint_selected": accepted and score < best_score,
                            "selection_policy": config.get("rul_selection", "mixed"),
                            "validation_fault_macro_RMSE": damaged.get("selection_fault_macro_RMSE"),
                            **{f"validation_{family}_selection_RMSE": value for family, value in damaged.get("selection_family_RMSE", {}).items()},
                            "learning_rate": optimizer.param_groups[0]["lr"], "correction_bound": float(bound)})
            progress.emit(f"corrector bound={bound} epoch={epoch + 1}: loss={float(loss.detach()):.4f}, validation RMSE={score:.3f}", force=True)
            if accepted and score < trial_best:
                trial_best, stale = score, 0
            else:
                stale += 1
            if accepted and score < best_score:
                best, best_score, best_bound = copy.deepcopy(model.corrector.state_dict()), score, float(bound)
            scheduler.step(score)
            if stale >= config["patience"]:
                break
    model.corrector.load_state_dict(best)
    model.correction_bound.fill_(best_bound)
    model.eval()
    return pd.DataFrame(history), audit, pd.DataFrame(cv_reports)


@torch.no_grad()
def select_fault_thresholds(model, validation_frame, scaler, config, seed):
    """Protect intact channels in clean and every structured-fault context."""
    from .detection import select_detector_thresholds
    # The format-7 detector is frozen after its separate fitting stage. Keep the
    # exported alarms and recovery gates on the same validation-selected rule.
    if config.get("finite_repair_gate") == "validated_threshold":
        return model.finite_repair_thresholds.detach().cpu().tolist()
    return select_detector_thresholds(model, validation_frame, scaler, config, seed)


@torch.no_grad()
def calibrate(model, frame, scaler, config, seed):
    return calibrate_endpoint_intervals(model, frame, scaler, config, seed, evaluator=evaluate)
