"""Staged training, grouped out-of-fold correction, calibration, and evaluation."""

import copy
import random
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
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
            raise RuntimeError("CUDA GPU training was requested, but PyTorch cannot access a CUDA GPU. Run nvidia-smi and the GPU check in README.md in your terminal. Run training in a GPU-enabled environment; CPU execution requires an explicit device='cpu' setting.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("The requested CUDA device index is not available.")
    return device


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=device.type == "cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def loader(dataset, config, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle, num_workers=0,
                      pin_memory=device_for(config).type == "cuda",
                      generator=torch.Generator().manual_seed(seed))


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


def objectives(model, batch, critic=None, consistency=True):
    result = model(batch["x"], batch["valid"], use_correction=False)
    valid = batch["valid"].unsqueeze(-1)
    changed = batch["altered"] & valid
    intact = ~batch["altered"] & valid
    error = result["repaired"] - batch["reference"][..., 3:]
    rec, identity = masked_mse(error, changed), masked_mse(error, intact)
    available = result["mask"].bool() & valid
    detection = (F.binary_cross_entropy_with_logits(result["logits"], batch["altered"].float(), reduction="none") * available).sum() / available.sum().clamp_min(1)
    scale = model.config["target_scale"]
    rul = F.mse_loss(result["base"] / scale, batch["target"] / scale)
    branch = .5 * (F.mse_loss(result["local_prediction"] / scale, batch["target"] / scale) + F.mse_loss(result["global_prediction"] / scale, batch["target"] / scale))
    cons = result["base"].sum() * 0
    if consistency:
        with torch.no_grad():
            teacher = model(batch["reference"], batch["valid"], use_repair=False, use_correction=False)["base"]
        available_fraction = result["mask"].sum((1, 2)) / (21 * batch["valid"].sum(1)).clamp_min(1)
        change = (torch.nan_to_num(batch["x"][..., 3:]) - batch["reference"][..., 3:]).abs()
        mild = (available_fraction >= .9) & ((change * available).sum((1, 2)) / available.sum((1, 2)).clamp_min(1) <= .05)
        if bool(mild.any()):
            cons = F.mse_loss(result["base"][mild] / scale, teacher[mild] / scale)
    adv = result["base"].sum() * 0 if critic is None else -critic(result["repaired"], batch["reference"][..., :3], batch["valid"]).mean()
    loss = rul + rec + .2 * identity + .1 * detection + .01 * adv + .1 * cons + .1 * branch
    return loss, result


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
    # Only the interpolation needs an unfused RNN for GP second derivatives.
    with _unfused_rnn_backends():
        alpha = torch.rand(len(real), 1, 1, device=real.device)
        interpolation = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        interpolated_score = critic(interpolation, settings, valid)
        gradient = torch.autograd.grad(interpolated_score.sum(), interpolation, create_graph=True)[0]
        norm = (gradient * valid.unsqueeze(-1)).flatten(1).norm(2, 1)
        penalty = config["gradient_penalty"] * ((norm - 1) ** 2).mean()
        penalty.backward()
    wasserstein = critic(fake, settings, valid).mean() - critic(real, settings, valid).mean()
    wasserstein.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.)
    optimizer.step()
    for p in critic.parameters():
        p.requires_grad_(False)
    return float((wasserstein + penalty).detach())


@torch.no_grad()
def evaluate(model, dataset, config, **forward_options):
    model.eval()
    prediction, target, scores, labels, records = [], [], [], [], []
    rec_squared, rec_count, intact_squared, intact_count = 0., 0, 0., 0
    for batch in loader(dataset, config):
        batch = move_batch(batch, next(model.parameters()).device)
        result = model(batch["x"], batch["valid"], **forward_options)
        prediction.extend(result["prediction"].cpu().tolist()); target.extend(batch["target"].cpu().tolist())
        changed = batch["altered"] & batch["valid"].unsqueeze(-1)
        intact = ~batch["altered"] & batch["valid"].unsqueeze(-1)
        squared = (result["repaired"] - batch["reference"][..., 3:]).square()
        rec_squared += float((squared * changed).sum()); rec_count += int(changed.sum())
        intact_squared += float((squared * intact).sum()); intact_count += int(intact.sum())
        finite = result["mask"].bool() & batch["valid"].unsqueeze(-1)
        scores.extend(result["fault_scores"][finite].cpu().tolist()); labels.extend(batch["altered"][finite].cpu().tolist())
        records.extend({"unit": int(u), "cycle": int(t), "prediction": float(p), "target": float(y), "raw_target": float(r)}
                       for u, t, p, y, r in zip(batch["unit"], batch["cycle"], result["prediction"], batch["target"], batch["raw_target"]))
    report = {**metrics(prediction, target), "reconstruction_RMSE": (rec_squared / rec_count) ** .5 if rec_count else None,
              "intact_repair_RMSE": (intact_squared / intact_count) ** .5 if intact_count else 0.,
              "fault_PR_AUC": float(average_precision_score(labels, scores)) if any(labels) and not all(labels) else None}
    return report, pd.DataFrame(records)


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
    seed_all(seed, config["cpu_threads"])
    device = device_for(config)
    progress = _Progress(progress_label or f"{subset} seed={seed}", config, verbose)
    identity = pipeline_identity(train_frame, validation_frame, subset, config, seed) if checkpoint_path else None
    checkpoint = load_checkpoint(checkpoint_path, identity)
    scaler = OperatingRegimeScaler().fit(train_frame, subset, seed)
    model, critic = GraceRUL(config).to(device), TemporalCritic(config).to(device)
    train = WindowDataset(train_frame, scaler, config, seed=seed)
    clean_validation = WindowDataset(validation_frame, scaler, config, mode="validation", seed=seed + 1, family="clean")
    corrupt_validation = WindowDataset(validation_frame, scaler, config, mode="validation", seed=seed + 1)
    history, best_score = [], float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        critic.load_state_dict(checkpoint["critic"])
        history, best_score = checkpoint["history"], checkpoint["best_score"]
        restore_random_state(checkpoint["random_state"])
        progress.emit(f"resuming checkpoint: {checkpoint['stage']} epoch {checkpoint['next_epoch']} completed", force=True)
    stages = ["predictor", "generator", "joint"]
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
        best_score, best, stale, start_epoch = float("inf"), None, 0, 0
        if checkpoint and stage_index == checkpoint["stage_index"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            best_score, best, stale = checkpoint["best_score"], checkpoint["best"], checkpoint["stale"]
            start_epoch = checkpoint["next_epoch"]
        total_epochs = config[f"{stage}_epochs"]
        for epoch in range(start_epoch, total_epochs):
            started = time.perf_counter()
            model.train(); train.set_epoch(epoch, (epoch + 1) / max(1, config[f"{stage}_epochs"]))
            losses = []
            adversarial = stage != "predictor" and epoch >= config["adversarial_warmup"]
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
                    loss = F.mse_loss(result["base"] / scale, batch["target"] / scale) + .05 * (F.mse_loss(result["local_prediction"] / scale, batch["target"] / scale) + F.mse_loss(result["global_prediction"] / scale, batch["target"] / scale))
                else:
                    loss, _ = objectives(model, batch, critic if adversarial else None)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {stage} training loss.")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                losses.append(float(loss.detach()))
                progress.emit(f"{label}: batch {step + 1}/{total_batches} complete, loss={losses[-1]:.4f}, elapsed={time.perf_counter() - started:.1f}s",
                              force=step == 0)
            progress.emit(f"{label}: validating clean and corrupted engines", force=True)
            clean, _ = evaluate(model, clean_validation, config, use_repair=stage != "predictor", use_correction=False)
            damaged, _ = evaluate(model, corrupt_validation, config, use_repair=stage != "predictor", use_correction=False)
            score = .5 * (clean["RMSE"] + damaged["RMSE"])
            row = {"subset": subset, "seed": seed, "stage": stage, "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                   "validation_macro_RMSE": score, "validation_clean_RMSE": clean["RMSE"], "validation_corrupt_RMSE": damaged["RMSE"], "validation_reconstruction_RMSE": damaged["reconstruction_RMSE"],
                   "duration_seconds": time.perf_counter() - started}
            history.append(row)
            progress.emit(f"{stage} {epoch + 1}: loss={row['train_loss']:.4f}, validation RMSE={score:.3f}, elapsed={row['duration_seconds']:.1f}s", force=True)
            if score < best_score:
                best_score, best, stale = score, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            stage_complete = stale >= config["patience"] or epoch + 1 == total_epochs
            if stage_complete:
                model.load_state_dict(best)
            if checkpoint_path:
                save_checkpoint(checkpoint_path, {"identity": identity, "stage": stage, "stage_index": stage_index,
                    "next_epoch": epoch + 1, "stage_complete": stage_complete, "model": model.state_dict(),
                    "critic": critic.state_dict(), "optimizer": optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
                    "best": best, "best_score": best_score, "stale": stale, "history": history, "random_state": random_state()})
            if stale >= config["patience"]:
                progress.emit(f"{stage}: early stopping after {stale} epochs without validation improvement; restoring best weights", force=True)
                break
        model.load_state_dict(best)
    for p in model.parameters():
        p.requires_grad_(True)
    return model.eval(), scaler, pd.DataFrame(history), best_score


def fit_corrector(model, fit_frame, validation_frame, subset, config, seed, *, checkpoint_dir=None):
    """Upstream temporary models never fit or select on their OOF engines."""
    units = np.sort(fit_frame.unit.unique())
    features, bases, targets, audit, cv_reports = [], [], [], [], []
    folds = KFold(config["oof_folds"], shuffle=True, random_state=seed)
    progress = _Progress(f"{subset} seed={seed}", config)
    for fold, (train_indices, held_indices) in enumerate(folds.split(units)):
        label = f"{subset} seed={seed} OOF {fold + 1}/{config['oof_folds']}"
        progress.emit(f"OOF fold {fold + 1}/{config['oof_folds']}: training temporary upstream model", force=True)
        available = np.random.default_rng(seed + fold).permutation(units[train_indices])
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
        dataset = WindowDataset(fit_frame[fit_frame.unit.isin(held)], fold_scaler, config, mode="correction", seed=seed + fold)
        progress.emit(f"OOF fold {fold + 1}/{config['oof_folds']}: collecting correction features", force=True)
        with torch.no_grad():
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
    optimizer = torch.optim.AdamW(model.corrector.parameters(), lr=config["learning_rate"])
    history, best, best_score, stale = [], None, float("inf"), 0
    # The final model's scaler is attached explicitly by the caller.
    val = WindowDataset(validation_frame, model.scaler, config, mode="validation", seed=seed + 1)
    progress.emit(f"fitting residual corrector for up to {config['correction_epochs']} epochs", force=True)
    for epoch in range(config["correction_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        delta = config["correction_bound"] * model.corrector(x).tanh().flatten()
        loss = F.mse_loss((base + delta).clamp_min(0) / config["target_scale"], y / config["target_scale"]) + config["correction_penalty"] * (delta / config["target_scale"]).square().mean()
        loss.backward(); optimizer.step()
        report, _ = evaluate(model, val, config)
        history.append({"subset": subset, "seed": seed, "epoch": epoch + 1, "train_loss": float(loss.detach()), "validation_macro_RMSE": report["RMSE"]})
        progress.emit(f"corrector {epoch + 1}: loss={float(loss.detach()):.4f}, validation RMSE={report['RMSE']:.3f}", force=True)
        if report["RMSE"] < best_score:
            best, best_score, stale = copy.deepcopy(model.corrector.state_dict()), report["RMSE"], 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.corrector.load_state_dict(best)
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
