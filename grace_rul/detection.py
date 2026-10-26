"""Validation-selected finite-fault detection, independent of bypassed RUL.

The sigmoid outputs are discrimination scores, not calibrated probabilities of
physical sensor failure. Outages provide missing-input context; their missing
cells never become finite-fault targets. No official test data are used here.
"""

from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .checkpoints import (load_checkpoint, pipeline_identity, random_state,
                          restore_random_state, save_checkpoint)
from .data import FAULTS, WindowDataset


DETECTOR_FAMILIES = ("clean", *FAULTS, "mixed")


def balanced_finite_fault_loss(logits, altered, available):
    """Average present positive/negative classes per sensor, excluding NaNs/pad."""
    values = F.binary_cross_entropy_with_logits(logits, altered.float(), reduction="none")
    positive, negative = altered & available, ~altered & available
    positives, negatives = positive.sum((0, 1)), negative.sum((0, 1))
    pos = (values * positive).sum((0, 1)) / positives.clamp_min(1)
    neg = (values * negative).sum((0, 1)) / negatives.clamp_min(1)
    classes = (positives > 0).float() + (negatives > 0).float()
    present = classes > 0
    return (((pos + neg) / classes.clamp_min(1)) * present).sum() / present.sum().clamp_min(1)


def _detector_inputs(batch, device):
    x = batch["x"].to(device)
    valid = batch["valid"].to(device).bool()
    settings = torch.nan_to_num(x[..., :3]) * valid.unsqueeze(-1)
    mask = torch.isfinite(x[..., 3:]) & valid.unsqueeze(-1)
    observed = torch.nan_to_num(x[..., 3:], nan=0., posinf=0., neginf=0.) * valid.unsqueeze(-1)
    return observed, mask.float(), settings, valid


class _FittingFaultMixture(Dataset):
    """Cover all contexts without multiplying an epoch's number of windows.

    Assignment rotates each epoch. Each underlying dataset still corrupts an
    entire engine trajectory once, so overlapping windows remain consistent.
    """
    def __init__(self, frame, scaler, config, seed):
        self.datasets = [WindowDataset(frame, scaler, config, mode="train", seed=seed, family=f)
                         for f in DETECTOR_FAMILIES]
        self.samples, self.engines = self.datasets[0].samples, self.datasets[0].engines
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        for dataset in self.datasets:
            dataset.set_epoch(epoch, progress=1.)

    def __getitem__(self, index):
        return self.datasets[(index + self.epoch) % len(self.datasets)][index]


def _loader(dataset, config, seed, *, shuffle=False):
    generator = torch.Generator().manual_seed(int(seed))
    sampler = None
    if shuffle and config.get("engine_balanced_sampling", True):
        engines = np.asarray([engine for engine, _ in dataset.samples])
        counts = np.bincount(engines, minlength=len(dataset.engines))
        sampler = WeightedRandomSampler(1. / counts[engines], len(dataset), replacement=True,
                                        generator=generator)
    return DataLoader(dataset, batch_size=int(config["batch_size"]), num_workers=0,
                      shuffle=shuffle and sampler is None, sampler=sampler, generator=generator)


@torch.no_grad()
def _validation_scores(model, frame, scaler, config, seed):
    """Return finite-only, sensor-wise scores for each fixed validation context."""
    device = next(model.detector.parameters()).device
    states = [(child, child.training) for child in model.detector.modules()]
    model.detector.eval()
    contexts = {}
    try:
        for family in DETECTOR_FAMILIES:
            dataset = WindowDataset(frame, scaler, config, mode="validation", seed=seed, family=family)
            scores, labels = [[] for _ in range(21)], [[] for _ in range(21)]
            for batch in _loader(dataset, config, seed):
                inputs = _detector_inputs(batch, device)
                probability = model.detector(*inputs).sigmoid().cpu().numpy()
                available = inputs[1].bool().cpu().numpy()
                if not np.isfinite(probability[available]).all():
                    raise RuntimeError("Non-finite validation detector scores; candidate selection is unsafe.")
                altered = batch["altered"].numpy()
                for channel in range(21):
                    finite = available[..., channel]
                    scores[channel].append(probability[..., channel][finite])
                    labels[channel].append(altered[..., channel][finite])
            contexts[family] = [
                (np.concatenate(s) if s else np.empty(0, np.float32),
                 np.concatenate(y) if y else np.empty(0, bool))
                for s, y in zip(scores, labels)]
    finally:
        for child, training in states:
            child.training = training
    return contexts


def _select_thresholds(contexts, maximum_fpr):
    """Maximize macro context F1 with an intact-cell guard in every context."""
    thresholds = []
    for channel in range(21):
        pairs = [values[channel] for values in contexts.values()]
        pooled = np.concatenate([score for score, _ in pairs])
        if not len(pooled):
            thresholds.append(1.)
            continue
        # Include observed score boundaries: useful classifiers can still emit
        # scores below .05, especially after the conservative -4 head prior.
        quantiles = np.quantile(pooled, np.linspace(0., 1., 129)).astype(np.float32)
        # Evaluation constructs float32 tensor thresholds. Use representable
        # float32 cuts so a double-precision nextafter cannot round back down.
        cuts = np.unique(np.r_[np.linspace(0., 1., 65, dtype=np.float32), quantiles,
                               np.nextafter(quantiles, np.float32(np.inf)),
                               np.nextafter(np.float32(1.), np.float32(0.)), 1.].astype(np.float32))
        cuts = cuts[(cuts >= 0.) & (cuts <= 1.)]
        feasible = np.ones(len(cuts), bool)
        objectives = []
        for scores, labels in pairs:
            negatives, positives = np.sort(scores[~labels]), np.sort(scores[labels])
            fp = len(negatives) - np.searchsorted(negatives, cuts, side="left")
            tp = len(positives) - np.searchsorted(positives, cuts, side="left")
            # 1.0 means explicitly disabled, including saturated sigmoid scores.
            fp[cuts == 1.] = 0
            tp[cuts == 1.] = 0
            if len(negatives):
                feasible &= fp / len(negatives) <= maximum_fpr
            if len(positives):
                objectives.append(2. * tp / np.maximum(1, tp + fp + len(positives)))
        objective = np.mean(objectives, axis=0) if objectives else np.zeros(len(cuts))
        choices = np.flatnonzero(feasible)
        best = max(choices, key=lambda i: (objective[i], cuts[i])) if len(choices) else len(cuts) - 1
        thresholds.append(float(cuts[best]))
    return thresholds


def _summarize(contexts, thresholds):
    families, all_fpr = {}, []
    for family, channels in contexts.items():
        tp = fp = positives = negatives = 0
        for (scores, labels), threshold in zip(channels, thresholds):
            flags = (scores >= threshold) & (threshold < 1.)
            npos, nneg = int(labels.sum()), int((~labels).sum())
            true = int((flags & labels).sum())
            false = int((flags & ~labels).sum())
            if nneg:
                all_fpr.append(false / nneg)
            tp += true; fp += false; positives += npos; negatives += nneg
        families[family] = {"F1": 2. * tp / max(1, tp + fp + positives),
                            "recall": tp / max(1, positives),
                            "false_positive_rate": fp / max(1, negatives),
                            "positive_cells": positives, "intact_cells": negatives}
    scored = [result["F1"] for result in families.values() if result["positive_cells"]]
    return {"score": float(np.mean(scored)) if scored else 0.,
            "clean_false_positive_rate": families.get("clean", {}).get("false_positive_rate", 0.),
            "max_context_false_positive_rate": max(all_fpr, default=0.),
            "contexts": families}


def select_detector_thresholds(model, validation_frame, scaler, config, seed, *, return_audit=False):
    """Validation-only thresholds guarded against clean and fault-context FPs.

    The bound is an empirical validation guard, not a guarantee on unseen
    engines, other severities, or real failures. Thresholding also does not
    calibrate the scores into physical failure probabilities.
    """
    maximum_fpr = float(config.get("detector_max_false_positive_rate", .05))
    if not 0 <= maximum_fpr < 1:
        raise ValueError("detector_max_false_positive_rate must be in [0, 1).")
    contexts = _validation_scores(model, validation_frame, scaler, config, seed)
    thresholds = _select_thresholds(contexts, maximum_fpr)
    if return_audit:
        audit = _summarize(contexts, thresholds)
        audit.update(scope="validation engines only; finite labels only", families=list(DETECTOR_FAMILIES),
                     maximum_false_positive_rate=maximum_fpr)
        return thresholds, audit
    return thresholds


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _scope(fit_frame, validation_frame, scaler):
    fitting = sorted(int(unit) for unit in fit_frame.unit.unique())
    validation = sorted(int(unit) for unit in validation_frame.unit.unique())
    if not fitting or not validation:
        raise ValueError("Detector refinement requires nonempty fitting and validation engines.")
    if set(fitting) & set(validation):
        raise ValueError("Detector fitting and validation engines must be disjoint.")
    if sorted(scaler.state.get("fit_units", [])) != fitting:
        raise ValueError("Detector refinement requires a scaler fitted on exactly the fitting engines.")
    return fitting, validation


def refine_bypassed_detector(model, fit_frame, validation_frame, scaler, config, seed, *, checkpoint_path=None):
    """Refine only a bypassed detector, retaining its unchanged validation baseline.

    The return value is an epoch DataFrame with a JSON-compatible ``audit`` in
    ``history.attrs``. Accepted weights replace only ``model.detector``. Model
    modes, requires-grad flags, and caller RNG state are restored. A completed
    checkpoint is reusable; an interrupted checkpoint resumes at an epoch
    boundary with optimizer/RNG state. Changed frames, scaler, configuration, or
    originating model weights reject stale checkpoint reuse.
    """
    epochs = int(config.get("detector_refinement_epochs", 15))
    skip = ("active_recovery" if bool(model.repair_enabled) else
            "detector_disabled" if not model.config.get("train_repair", True) else
            "disabled" if epochs <= 0 else None)
    if skip:
        history = pd.DataFrame([{"stage": "detector_refinement", "epoch": 0, "accepted": False,
                                 "status": "skipped_" + skip}])
        history.attrs["audit"] = {"status": "skipped_" + skip, "accepted": False,
                                  "rul_path_unchanged": True}
        return history
    fitting, validation = _scope(fit_frame, validation_frame, scaler)
    maximum_fpr = float(config.get("detector_max_false_positive_rate", .05))
    learning_rate = float(config.get("detector_refinement_learning_rate", .001))
    patience = int(config.get("detector_refinement_patience", 5))
    improvement = float(config.get("detector_refinement_min_improvement", 1e-4))
    if not 0 <= maximum_fpr < 1 or learning_rate <= 0 or patience < 1 or improvement < 0:
        raise ValueError("Invalid detector refinement FPR, learning rate, patience, or minimum improvement.")
    identity = pipeline_identity(fit_frame, validation_frame, "finite-detector-refinement", config, seed)
    identity["detector_families"] = list(DETECTOR_FAMILIES)
    identity["scaler"] = hashlib.sha256(json.dumps(scaler.state, sort_keys=True).encode()).hexdigest()
    identity["originating_model"] = _state_hash({k: v for k, v in model.state_dict().items()
                                                if not k.startswith(("detector.", "corrector."))
                                                and k not in {"correction_bound", "finite_repair_thresholds"}})
    initial = deepcopy(model.detector.state_dict())
    initial_hash = _state_hash(initial)
    saved = load_checkpoint(checkpoint_path, identity)
    if saved is not None and initial_hash not in {saved["initial_detector_hash"],
                                                   _state_hash(saved["selected_detector"]),
                                                   _state_hash(saved["current_detector"])}:
        raise ValueError("Detector checkpoint does not match the originating detector weights.")
    if saved is not None and saved["complete"]:
        model.detector.load_state_dict(saved["selected_detector"])
        history = pd.DataFrame(saved["history"])
        history.attrs["audit"] = {**saved["audit"], "checkpoint_reused": True}
        return history

    modes = [(child, child.training) for child in model.modules()]
    grad_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    caller_rng = random_state()
    selected = deepcopy(initial)
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.detector.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(model.detector.parameters(), lr=learning_rate)
        dataset = _FittingFaultMixture(fit_frame, scaler, config, seed)
        if not len(dataset):
            raise ValueError("Detector refinement has no fitting windows.")
        if saved is None:
            thresholds, baseline = select_detector_thresholds(model, validation_frame, scaler, config,
                                                              seed, return_audit=True)
            best, best_epoch, stale, start = baseline["score"], 0, 0, 0
            selected_thresholds = thresholds
            selected_metrics = baseline
            history = [{"stage": "detector_refinement", "epoch": 0, "train_loss": None,
                        "accepted": False, "status": "unchanged_baseline", **{
                            key: baseline[key] for key in ("score", "clean_false_positive_rate",
                                                          "max_context_false_positive_rate")}}]
        else:
            model.detector.load_state_dict(saved["current_detector"])
            optimizer.load_state_dict(saved["optimizer"])
            selected = saved["selected_detector"]
            baseline, selected_metrics = saved["baseline"], saved["selected_metrics"]
            selected_thresholds = saved["thresholds"]
            best, best_epoch, stale, start = saved["best_score"], saved["best_epoch"], saved["stale"], saved["epoch"]
            initial_hash = saved["initial_detector_hash"]
            history = saved["history"]
            restore_random_state(saved["random_state"])
        device = next(model.detector.parameters()).device
        for epoch in range(start, epochs):
            if stale >= patience:
                break
            model.detector.train()
            dataset.set_epoch(epoch)
            losses = []
            for step, batch in enumerate(_loader(dataset, config, seed + 1009 * (epoch + 1), shuffle=True)):
                if config.get("maximum_batches") is not None and step >= config["maximum_batches"]:
                    break
                inputs = _detector_inputs(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.detector(*inputs)
                loss = balanced_finite_fault_loss(logits, batch["altered"].to(device), inputs[1].bool())
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite detector refinement loss; the current candidate was rejected.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.detector.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            thresholds, metrics = select_detector_thresholds(model, validation_frame, scaler, config,
                                                             seed, return_audit=True)
            safe = (metrics["max_context_false_positive_rate"] <= maximum_fpr + 1e-12 and
                    metrics["clean_false_positive_rate"] <= maximum_fpr + 1e-12)
            accepted = safe and metrics["score"] > best + improvement
            if accepted:
                selected = deepcopy(model.detector.state_dict())
                selected_thresholds, selected_metrics = thresholds, metrics
                best, best_epoch, stale = metrics["score"], epoch + 1, 0
            else:
                stale += 1
            history.append({"stage": "detector_refinement", "epoch": epoch + 1,
                            "train_loss": float(np.mean(losses)) if losses else None,
                            "accepted": bool(accepted), "status": "candidate", **{
                                key: metrics[key] for key in ("score", "clean_false_positive_rate",
                                                             "max_context_false_positive_rate")}})
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, {
                    "identity": identity, "complete": False, "epoch": epoch + 1,
                    "initial_detector_hash": initial_hash, "current_detector": model.detector.state_dict(),
                    "selected_detector": selected, "optimizer": optimizer.state_dict(),
                    "random_state": random_state(), "history": history, "baseline": baseline,
                    "selected_metrics": selected_metrics, "thresholds": selected_thresholds,
                    "best_score": best, "best_epoch": best_epoch, "stale": stale})
        audit = {"status": "refined" if best_epoch else "unchanged_baseline", "accepted": best_epoch > 0,
                 "selected_epoch": best_epoch, "fitting_engines": fitting, "validation_engines": validation,
                 "families": list(DETECTOR_FAMILIES), "thresholds": selected_thresholds,
                 "baseline": baseline, "selected": selected_metrics, "rul_path_unchanged": True,
                 "checkpoint_reused": False,
                 "selection_metric": "macro F1 across validation contexts containing finite faults",
                 "score_interpretation": "discrimination scores, not calibrated physical failure probabilities"}
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, {
                "identity": identity, "complete": True, "epoch": history[-1]["epoch"],
                "initial_detector_hash": initial_hash, "current_detector": model.detector.state_dict(),
                "selected_detector": selected, "history": history, "audit": audit})
        result = pd.DataFrame(history)
        result.attrs["audit"] = audit
        return result
    finally:
        model.detector.load_state_dict(selected)
        for parameter, requires_grad in grad_flags:
            parameter.requires_grad_(requires_grad)
        for child, training in modes:
            child.training = training
        restore_random_state(caller_rng)
