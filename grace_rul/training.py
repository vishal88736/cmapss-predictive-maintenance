"""Staged training, grouped out-of-fold correction, calibration, and evaluation."""

import copy
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
from .data import OperatingRegimeScaler, WindowDataset
from .model import GraceRUL, TemporalCritic
from .checkpoints import load_checkpoint, pipeline_identity, random_state, restore_random_state, save_checkpoint


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


def repair_trend_loss(repaired, reference, altered, valid):
    pairs = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1)
    pairs = pairs & (altered[:, 1:] | altered[:, :-1])
    error = (repaired[:, 1:] - repaired[:, :-1]) - (reference[:, 1:] - reference[:, :-1])
    return masked_mse(error, pairs)


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


def objectives(model, batch, critic=None, consistency=True):
    result = model(batch["x"], batch["valid"], use_correction=False)
    valid = batch["valid"].unsqueeze(-1)
    changed = batch["altered"] & valid
    intact = ~batch["altered"] & valid
    error = result["repaired"] - batch["reference"][..., 3:]
    rec, identity = masked_mse(error, changed), masked_mse(error, intact)
    generator_rec = masked_mse(result["generated"] - batch["reference"][..., 3:], changed)
    trend = repair_trend_loss(result["repaired"], batch["reference"][..., 3:], batch["altered"], batch["valid"])
    available = result["mask"].bool() & valid
    positives = (batch["altered"] & available).sum((0, 1))
    negatives = available.sum((0, 1)) - positives
    positive_weight = ((negatives + 1) / (positives + 1)).sqrt().clamp(1., model.config["detector_positive_weight_cap"])
    detection = (F.binary_cross_entropy_with_logits(result["logits"], batch["altered"].float(),
                 pos_weight=positive_weight, reduction="none") * available).sum() / available.sum().clamp_min(1)
    scale = model.config["target_scale"]
    rul = F.mse_loss(result["base"] / scale, batch["target"] / scale)
    branch = .5 * (F.mse_loss(result["local_prediction"] / scale, batch["target"] / scale) + F.mse_loss(result["global_prediction"] / scale, batch["target"] / scale))
    cons = result["base"].sum() * 0
    if consistency:
        with torch.no_grad(), evaluation_mode(model):
            teacher = model(batch["reference"], batch["valid"], use_repair=False, use_correction=False)["base"]
        available_fraction = result["mask"].sum((1, 2)) / (21 * batch["valid"].sum(1)).clamp_min(1)
        change = (torch.nan_to_num(batch["x"][..., 3:]) - batch["reference"][..., 3:]).abs()
        mild = (available_fraction >= .9) & ((change * available).sum((1, 2)) / available.sum((1, 2)).clamp_min(1) <= .05)
        if bool(mild.any()):
            cons = F.mse_loss(result["base"][mild] / scale, teacher[mild] / scale)
    adv = result["base"].sum() * 0 if critic is None else -critic(result["repaired"], batch["reference"][..., :3], batch["valid"]).mean()
    terms = {"rul": rul, "reconstruction": rec, "generator_reconstruction": generator_rec,
             "identity": identity, "detection": detection, "trend": trend,
             "adversarial": adv, "consistency": cons, "branch": branch}
    loss = sum(model.config[f"{name}_loss_weight"] * value for name, value in terms.items())
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
def evaluate(model, dataset, config, fault_thresholds=None, **forward_options):
    model.eval()
    prediction, target, scores, labels, records = [], [], [], [], []
    rec_squared, rec_count, intact_squared, intact_count = 0., 0, 0., 0
    rec_absolute, intact_absolute, tp, fp, fn, tn = 0., 0., 0, 0, 0, 0
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
        finite = result["mask"].bool() & batch["valid"].unsqueeze(-1)
        thresholds = result["fault_scores"].new_tensor(fault_thresholds if fault_thresholds is not None else [.5] * 21)
        flags = result["fault_scores"] >= thresholds
        truth = batch["altered"]
        tp += int((flags & truth & finite).sum()); fp += int((flags & ~truth & finite).sum())
        fn += int((~flags & truth & finite).sum()); tn += int((~flags & ~truth & finite).sum())
        scores.extend(result["fault_scores"][finite].cpu().tolist()); labels.extend(batch["altered"][finite].cpu().tolist())
        records.extend({"unit": int(u), "cycle": int(t), "prediction": float(p), "target": float(y), "raw_target": float(r)}
                       for u, t, p, y, r in zip(batch["unit"], batch["cycle"], result["prediction"], batch["target"], batch["raw_target"]))
    report = {**metrics(prediction, target), "reconstruction_RMSE": (rec_squared / rec_count) ** .5 if rec_count else None,
              "reconstruction_MAE": rec_absolute / rec_count if rec_count else None,
              "intact_repair_RMSE": (intact_squared / intact_count) ** .5 if intact_count else 0.,
              "intact_repair_MAE": intact_absolute / intact_count if intact_count else 0.,
              "fault_precision": tp / (tp + fp) if tp + fp else None,
              "fault_recall": tp / (tp + fn) if tp + fn else None,
              "fault_false_positive_rate": fp / (fp + tn) if fp + tn else None,
              "fault_F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
              "fault_threshold_scope": "validation-selected" if fault_thresholds is not None else "default 0.5",
              "fault_PR_AUC": float(average_precision_score(labels, scores)) if any(labels) and not all(labels) else None}
    return report, pd.DataFrame(records)


def validation_score(model, clean_dataset, corrupt_dataset, config, **options):
    clean, _ = evaluate(model, clean_dataset, config, **options)
    damaged, _ = evaluate(model, corrupt_dataset, config, **options)
    score = .5 * (clean["RMSE"] + damaged["RMSE"])
    if not np.isfinite(score):
        raise FloatingPointError("Non-finite clean/corrupted validation RMSE.")
    return score, clean, damaged


def validate_training_config(config):
    for name in ["predictor_epochs", "generator_epochs", "joint_epochs", "correction_epochs",
                 "patience", "batch_size", "gradient_penalty_batch_size", "critic_steps",
                 "validation_prefixes", "correction_prefixes", "window_stride", "lr_patience", "bootstrap_samples"]:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name in [k for k in config if k.endswith("_loss_weight")] + ["max_clean_rmse_increase", "max_intact_repair_rmse", "max_reconstruction_rmse"]:
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if config["reconstruction_loss_weight"] <= 0 or config["rul_loss_weight"] <= 0:
        raise ValueError("RUL and reconstruction weights must stay positive.")
    if not 0 < config["lr_factor"] < 1 or config["minimum_learning_rate"] <= 0:
        raise ValueError("Invalid learning-rate scheduler settings.")
    if not config["correction_bounds"] or any(not np.isfinite(b) or b <= 0 for b in config["correction_bounds"]):
        raise ValueError("correction_bounds must contain finite positive bounds.")
    for name in ["learning_rate", "repair_learning_rate", "joint_learning_rate", "minimum_learning_rate",
                 "correction_bound", "detector_positive_weight_cap"]:
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if config["detector_positive_weight_cap"] < 1:
        raise ValueError("detector_positive_weight_cap must be at least one.")
    seeds = config["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must contain distinct nonnegative integers.")
    from .config import SUBSETS
    if not config["subsets"] or len(set(config["subsets"])) != len(config["subsets"]) or set(config["subsets"]) - set(SUBSETS):
        raise ValueError("subsets must contain distinct supported C-MAPSS subsets.")
    if config["predictor_family"] not in {"gated", "cnn_gru", "cnn_lstm", "transformer"}:
        raise ValueError("Unsupported predictor_family.")


def acceptable_checkpoint(clean, damaged, clean_reference, config):
    return (clean["RMSE"] <= clean_reference * (1 + config["max_clean_rmse_increase"]) and
            damaged.get("intact_repair_RMSE", 0.) <= config["max_intact_repair_rmse"] and
            (damaged.get("reconstruction_RMSE") or 0.) <= config["max_reconstruction_rmse"])


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
    scaler = OperatingRegimeScaler().fit(train_frame, subset, config["split_seed"])
    model, critic = GraceRUL(config).to(device), TemporalCritic(config).to(device)
    train = WindowDataset(train_frame, scaler, config, seed=seed)
    clean_validation = WindowDataset(validation_frame, scaler, config, mode="validation", seed=config["validation_seed"], family="clean")
    corrupt_validation = WindowDataset(validation_frame, scaler, config, mode="validation", seed=config["validation_seed"])
    history, best_score, clean_reference = [], float("inf"), float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        critic.load_state_dict(checkpoint["critic"])
        history, best_score = checkpoint["history"], checkpoint["best_score"]
        clean_reference = checkpoint["clean_reference"]
        restore_random_state(checkpoint["random_state"])
        progress.emit(f"resuming checkpoint: {checkpoint['stage']} epoch {checkpoint['next_epoch']} completed", force=True)
    stages = ["predictor", "generator", "joint"] if config["train_repair"] else ["predictor"]
    for stage_index, stage in enumerate(stages):
        if checkpoint and (stage_index < checkpoint["stage_index"] or
                           (stage_index == checkpoint["stage_index"] and checkpoint["stage_complete"])):
            continue
        for name, parameter in model.named_parameters():
            enabled = name.startswith("predictor") if stage == "predictor" else name.startswith(("detector", "generator")) if stage == "generator" else not name.startswith("corrector")
            parameter.requires_grad_(enabled)
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
        stale, start_epoch = 0, 0
        if checkpoint and stage_index == checkpoint["stage_index"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            critic_scheduler.load_state_dict(checkpoint["critic_scheduler"])
            best_score, best, stale = checkpoint["best_score"], checkpoint["best"], checkpoint["stale"]
            start_epoch = checkpoint["next_epoch"]
        total_epochs = config[f"{stage}_epochs"]
        for epoch in range(start_epoch, total_epochs):
            started = time.perf_counter()
            model.train(); train.set_epoch(epoch, (epoch + 1) / max(1, config[f"{stage}_epochs"]))
            model.repair_enabled.fill_(True)
            if stage == "generator":
                model.predictor.eval()
                # cuDNN GRU backward to generator inputs needs training mode;
                # this GRU has no dropout. Predictor dropout stays disabled.
                for module in model.predictor.modules():
                    if isinstance(module, torch.nn.RNNBase):
                        module.train()
            losses = []
            adversarial = stage != "predictor" and epoch >= config["adversarial_warmup"] and config["adversarial_loss_weight"] > 0
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
                    loss = config["rul_loss_weight"] * F.mse_loss(result["base"] / scale, batch["target"] / scale) + .5 * config["branch_loss_weight"] * (F.mse_loss(result["local_prediction"] / scale, batch["target"] / scale) + F.mse_loss(result["global_prediction"] / scale, batch["target"] / scale))
                else:
                    loss, result = objectives(model, batch, critic if adversarial else None)
                    for name, value in result["loss_terms"].items():
                        term_totals[name] = term_totals.get(name, 0.) + float(value)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {stage} training loss.")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
                losses.append(float(loss.detach()))
                progress.emit(f"{label}: batch {step + 1}/{total_batches} complete, loss={losses[-1]:.4f}, elapsed={time.perf_counter() - started:.1f}s",
                              force=step == 0)
            progress.emit(f"{label}: validating clean and corrupted engines", force=True)
            score, clean, damaged = validation_score(model, clean_validation, corrupt_validation, config,
                                                     use_repair=stage != "predictor", use_correction=False)
            accepted = stage == "predictor" or acceptable_checkpoint(clean, damaged, clean_reference, config)
            row = {"subset": subset, "seed": seed, "stage": stage, "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                   "validation_macro_RMSE": score, "validation_clean_RMSE": clean["RMSE"], "validation_corrupt_RMSE": damaged["RMSE"], "validation_reconstruction_RMSE": damaged["reconstruction_RMSE"],
                   "validation_intact_repair_RMSE": damaged.get("intact_repair_RMSE", 0.),
                   "checkpoint_accepted": accepted, "learning_rate": optimizer.param_groups[0]["lr"],
                   "duration_seconds": time.perf_counter() - started,
                   **{f"loss_{name}": value / len(losses) for name, value in term_totals.items()}}
            history.append(row)
            progress.emit(f"{stage} {epoch + 1}: loss={row['train_loss']:.4f}, validation RMSE={score:.3f}, elapsed={row['duration_seconds']:.1f}s", force=True)
            if accepted and score < best_score:
                best_score, best, stale = score, copy.deepcopy(model.state_dict()), 0
                if stage == "predictor":
                    clean_reference = clean["RMSE"]
            else:
                stale += 1
            scheduler.step(score)
            critic_scheduler.step(score)
            stage_complete = stale >= config["patience"] or epoch + 1 == total_epochs
            if stage_complete:
                model.load_state_dict(best)
            if checkpoint_path:
                save_checkpoint(checkpoint_path, {"identity": identity, "stage": stage, "stage_index": stage_index,
                    "next_epoch": epoch + 1, "stage_complete": stage_complete, "model": model.state_dict(),
                    "critic": critic.state_dict(), "optimizer": optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "critic_scheduler": critic_scheduler.state_dict(),
                    "clean_reference": clean_reference,
                    "best": best, "best_score": best_score, "stale": stale, "history": history, "random_state": random_state()})
            if stale >= config["patience"]:
                progress.emit(f"{stage}: early stopping after {stale} epochs without validation improvement; restoring best weights", force=True)
                break
        model.load_state_dict(best)
    if not config["train_repair"]:
        model.repair_enabled.fill_(False)
    for p in model.parameters():
        p.requires_grad_(True)
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
                                         mode="validation", seed=seed + fold, family="clean" if family == "clean" else None)
            cv_report, _ = evaluate(fold_model, held_dataset, config, use_correction=False)
            cv_reports.append({"subset": subset, "seed": seed, "fold": fold + 1, "family": family,
                               "scope": "engine-held-out detector/recovery/predictor; final corrector excluded", **cv_report})
        progress.emit(f"OOF fold {fold + 1}/{config['oof_folds']}: collecting correction features", force=True)
        with torch.no_grad():
            for family in ["clean", None]:
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
    val = WindowDataset(validation_frame, model.scaler, config, mode="validation", seed=config["validation_seed"])
    clean_val = WindowDataset(validation_frame, model.scaler, config, mode="validation", seed=config["validation_seed"], family="clean")
    best_score, clean, damaged = validation_score(model, clean_val, val, config, use_correction=False)
    clean_reference = clean["RMSE"]
    # Retain a zero-correction candidate if every trained candidate is worse.
    torch.nn.init.zeros_(model.corrector[-1].weight)
    torch.nn.init.zeros_(model.corrector[-1].bias)
    initial = copy.deepcopy(model.corrector.state_dict())
    best, best_bound = copy.deepcopy(initial), float(model.correction_bound)
    history = [{"subset": subset, "seed": seed, "epoch": 0, "train_loss": None,
                "validation_macro_RMSE": best_score, "validation_clean_RMSE": clean["RMSE"],
                "validation_corrupt_RMSE": damaged["RMSE"], "checkpoint_accepted": True,
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
            score, clean, damaged = validation_score(model, clean_val, val, config)
            accepted = clean["RMSE"] <= clean_reference * (1 + config["max_clean_rmse_increase"])
            history.append({"subset": subset, "seed": seed, "epoch": epoch + 1, "train_loss": float(loss.detach()),
                            "validation_macro_RMSE": score, "validation_clean_RMSE": clean["RMSE"],
                            "validation_corrupt_RMSE": damaged["RMSE"], "checkpoint_accepted": accepted,
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
    """Select sensor-specific thresholds against finite fault labels on validation only."""
    scores, labels = [[] for _ in range(21)], [[] for _ in range(21)]
    for family in ["clean", "bias", "drift", "noise"]:
        dataset = WindowDataset(validation_frame, scaler, config, mode="validation", seed=seed, family=family)
        for batch in loader(dataset, config):
            batch = move_batch(batch, device_for(config)); out = model(batch["x"], batch["valid"])
            for j in range(21):
                valid = batch["valid"] & out["mask"][..., j].bool()
                scores[j].extend(out["fault_scores"][..., j][valid].cpu().tolist())
                labels[j].extend(batch["altered"][..., j][valid].cpu().tolist())
    thresholds = []
    for s, y in zip(scores, labels):
        s, y = np.asarray(s), np.asarray(y, bool)
        choices = []
        for threshold in np.linspace(.05, .99, 60):
            flags = s >= threshold
            fp = (flags & ~y).sum() / max(1, (~y).sum())
            tp = (flags & y).sum(); f1 = 2 * tp / max(1, flags.sum() + y.sum())
            if fp <= .05:
                choices.append((f1, -threshold, threshold))
        thresholds.append(float(max(choices)[2]) if choices else 1.0)
    return thresholds


@torch.no_grad()
def calibrate(model, frame, scaler, config, seed):
    dataset = WindowDataset(frame, scaler, config, mode="calibration", seed=seed, family="clean")
    _, records = evaluate(model, dataset, config)
    residuals = records.target.to_numpy() - records.prediction.to_numpy()
    alpha = config["interval_alpha"]
    lower, upper = np.quantile(residuals, [alpha / 2, 1 - alpha / 2])
    return {"lower_offset": float(lower), "upper_offset": float(upper), "alpha": alpha,
            "engines": int(len(records)), "scope": "unmodified capped-target calibration endpoints; corrupted/OOD coverage is not guaranteed"}, records


def interval_bounds(predictions, calibration):
    p = np.asarray(predictions)
    lo = np.maximum(0, p + calibration["lower_offset"])
    hi = np.maximum(lo, p + calibration["upper_offset"])
    return lo, hi
