"""Held-validation recovery guards across declared structured-fault contexts.

No data files are opened here. Callers supply complete training-held validation
engines, disjoint from the scaler's fitting engines. This audit measures sensor
recovery without the RUL corrector; it does not replace clean RUL/NASA guards or
establish safety on unseen faults, official test engines, or a deployed fleet.
"""

import hashlib
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SETTINGS, SENSORS
from .data import FAULTS, WindowDataset
from .normalization import transform_spec


DEFAULT_SEVERITIES = (.1, .2, .3, .4, .5)
RECOVERY_FAMILIES = ("clean", *FAULTS, "mixed")
CACHED_SCENARIO_OPTIONS = ("sequence_length", "validation_prefixes", "clean_fraction", "fault_start_horizon",
    "drift_ramp_cycles", "block_gap_cycles", "fault_noise_autocorrelation", "fault_transition_gain", "intermittent_mean_burst")
RECOVERY_STATE_BUFFERS = {"finite_repair_thresholds", "recovery_centers", "recovery_lower",
                          "recovery_upper", "recovery_regime_count", "repair_enabled"}


def _sensor_config(config):
    method, scale = transform_spec(config)
    return {"sensor_transform": method, "sensor_transform_scale": scale,
            "train_repair": config.get("train_repair", True),
            "finite_repair_gate": config.get("finite_repair_gate", "continuous"),
            "detach_repair_gate": config.get("detach_repair_gate", False),
            "recovery_safety": config.get("recovery_safety", "none"),
            "max_intact_repair_rmse": float(config.get("max_intact_repair_rmse", .15))}


def recovery_state_sha256(model):
    """Bind the sensor-repair computation, excluding RUL prediction/correction.

    An accepted sensor audit stays valid after fitting the corrector or changing
    predictor weights, because those modules cannot change repaired readings.
    Empty-state fixture modules are valid and still bind their sensor config.
    This fingerprint detects stale proofs; it is not a signed attestation.
    """
    digest = hashlib.sha256(b"grace-recovery-state-v1\0")
    options = _sensor_config(getattr(model, "config", {}))
    digest.update(json.dumps(options, sort_keys=True, allow_nan=False).encode())
    for name, value in sorted(model.state_dict().items()):
        if not (name.startswith(("detector.", "generator.")) or name in RECOVERY_STATE_BUFFERS):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        # Byte views handle all torch floating dtypes, including bfloat16.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _severities(config):
    values = config.get("recovery_validation_severities", DEFAULT_SEVERITIES)
    if isinstance(values, (str, bytes)):
        raise ValueError("recovery_validation_severities must be a nonempty sequence.")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError("recovery_validation_severities must be a nonempty sequence.") from exc
    try:
        valid = (bool(values) and not any(isinstance(value, (bool, np.bool_)) for value in values) and
                 all(np.isfinite(float(value)) and 0 < float(value) <= 1 for value in values))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Recovery validation severities must be finite numbers in (0, 1].")
    values = tuple(float(value) for value in values)
    if len(set(values)) != len(values):
        raise ValueError("Recovery validation severities must be distinct.")
    return values


def _budget(config, name, default=None):
    value = config.get(name, default)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and nonnegative, or None when optional.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative, or None when optional.") from exc
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative, or None when optional.")
    return value


def _guard_limits(config):
    return {name: _budget(config, name, default) for name, default in [
        ("max_intact_repair_rmse", .15), ("max_linear_reconstruction_ratio", 1.),
        ("recovery_validation_max_channel_intact_rmse", None),
        ("recovery_validation_max_window_intact_rmse", None),
        ("recovery_validation_max_reconstruction_rmse", None)]}


def _engine_ids(values):
    try:
        ids = list(values)
    except TypeError as exc:
        raise ValueError("Recovery audit engine scope must be a sequence.") from exc
    if not ids or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1 for value in ids):
        raise ValueError("Recovery audit engine scope requires positive integer IDs.")
    if len(set(ids)) != len(ids):
        raise ValueError("Recovery audit engine scope contains duplicate IDs.")
    return sorted(int(value) for value in ids)


def validate_recovery_audit(audit, config, partitions, model=None):
    """Reject a stale/partial sensor-recovery proof before export or loading.

    Acceptance binds the complete declared context matrix, held-validation
    engines, prefix/scenario settings, guard limits, and optionally the actual
    sensor-repair state. The persisted metrics are checked against their guards;
    this validates the artifact contract, not the honesty of arbitrary files.
    """
    if (not isinstance(audit, dict) or audit.get("complete") is not True or
            audit.get("accepted") is not True or audit.get("mode") != "exact" or
            audit.get("use_repair") is not True or audit.get("use_correction") is not False):
        raise ValueError("Active recovery requires a complete accepted exact recovery audit.")
    if not isinstance(partitions, dict) or "validation" not in partitions:
        raise ValueError("Recovery audit requires saved validation partitions.")
    validation = _engine_ids(partitions["validation"])
    if _engine_ids(audit.get("validation_units", [])) != validation:
        raise ValueError("Recovery audit validation engines disagree with saved partitions.")
    for scope in ("fit", "calibration"):
        if set(validation) & set(partitions.get(scope, [])):
            raise ValueError("Recovery audit validation engines overlap fitting/calibration engines.")
    severities = _severities(config)
    cases = [("clean", 0.), *((family, severity) for family in RECOVERY_FAMILIES[1:] for severity in severities)]
    if (audit.get("families") != list(RECOVERY_FAMILIES) or audit.get("severities") != list(severities) or
            audit.get("expected_contexts") != len(cases) or audit.get("evaluated_contexts") != len(cases)):
        raise ValueError("Recovery audit contexts/severities disagree with the declared matrix.")
    prefixes = _positive_integer(config["validation_prefixes"], "validation_prefixes")
    length = _positive_integer(config["sequence_length"], "sequence_length")
    expected_options = {name: config.get(name) for name in CACHED_SCENARIO_OPTIONS}
    if (audit.get("seed") != config["validation_seed"] or audit.get("validation_prefixes") != prefixes or
            audit.get("sequence_length") != length or audit.get("scenario_options") != expected_options or
            audit.get("prefix_policy") != {"mode": "seeded_validation_prefixes", "prefixes_per_engine": prefixes} or
            audit.get("representation") != list(transform_spec(config))):
        raise ValueError("Recovery audit seed, prefixes, sequence or scenario options disagree with configuration.")
    limits = _guard_limits(config)
    if audit.get("guard_limits") != limits:
        raise ValueError("Recovery audit guard budgets disagree with configuration.")
    contexts = audit.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != len(cases):
        raise ValueError("Recovery audit context reports are incomplete.")
    expected_windows = None
    metrics = [
        ("max_intact_repair_rmse", "linear_intact_repair_RMSE", "intact"),
        ("recovery_validation_max_channel_intact_rmse", "linear_intact_channel_RMSE_max", "intact_channel"),
        ("recovery_validation_max_window_intact_rmse", "linear_intact_window_RMSE_max", "intact_window"),
        ("recovery_validation_max_reconstruction_rmse", "reconstruction_RMSE", "reconstruction")]
    for context, (family, severity) in zip(contexts, cases):
        if (not isinstance(context, dict) or context.get("family") != family or context.get("severity") != severity or
                context.get("complete") is not True or context.get("accepted") is not True):
            raise ValueError("Recovery audit context axes or acceptance are inconsistent.")
        windows = _positive_integer(context.get("windows"), "recovery audit windows")
        if (windows != context.get("expected_windows") or windows < len(validation) or
                windows > len(validation) * prefixes or (expected_windows is not None and windows != expected_windows)):
            raise ValueError("Recovery audit window counts do not cover the declared prefixes.")
        expected_windows = windows
        guards = context.get("guards")
        if (not isinstance(guards, dict) or not {"finite", "intact"}.issubset(guards) or
                any(value is not True for value in guards.values()) or context.get("failed_guards") != [] or
                context.get("nonfinite_entries") != 0):
            raise ValueError("Recovery audit contains failed or missing guard flags.")
        _positive_integer(context.get("intact_cells"), "recovery audit intact cells")
        changed = context.get("changed_cells")
        if isinstance(changed, bool) or not isinstance(changed, int) or changed < 0:
            raise ValueError("Recovery audit changed-cell count is invalid.")
        checks = list(metrics)
        if family != "clean" and changed:
            checks.append(("max_linear_reconstruction_ratio", "linear_reconstruction_ratio", "linear_reconstruction"))
        for key, metric, guard in checks:
            budget = limits[key]
            if budget is None:
                continue
            value = context.get(metric)
            if value is None and not changed and metric == "reconstruction_RMSE":
                continue
            if (guard not in guards or isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not np.isfinite(value) or not 0 <= value <= budget):
                raise ValueError(f"Recovery audit metric {metric} violates its declared guard.")
    if audit.get("failing_contexts") != [] or not isinstance(audit.get("worst"), dict) or audit["worst"].get("nonfinite_contexts") != 0:
        raise ValueError("Recovery audit worst/failure summary is inconsistent.")
    maximum = max(context["linear_intact_repair_RMSE"] for context in contexts)
    if audit["worst"].get("linear_intact_repair_RMSE") != maximum:
        raise ValueError("Recovery audit worst intact metric is inconsistent.")
    fingerprint = audit.get("recovery_state_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
        raise ValueError("Recovery audit is missing a valid sensor-repair state fingerprint.")
    if model is not None and fingerprint != recovery_state_sha256(model):
        raise ValueError("Recovery audit fingerprint disagrees with the sensor-repair state or clipping policy.")
    return audit


class RecoveryValidationMatrix:
    """Cache deterministic windows/trajectories for a declared validation matrix.

    Clean data are severity-independent and appear once. The default therefore
    contains 41 unique cases: clean plus eight fault/mixed families at five
    severities. The same per-engine prefix seed is used in every case. Dataset
    trajectory caches persist across candidate audits, with no epoch resampling.
    ``maximum_batches`` and ``maximum_windows`` never restrict an exact audit.
    """

    def __init__(self, validation_frame, scaler, config, *, seed=None):
        required = ["unit", "cycle", *SETTINGS, *SENSORS]
        if validation_frame.empty or any(column not in validation_frame for column in required):
            raise ValueError("Recovery validation needs nonempty complete held-engine trajectories.")
        identifiers = validation_frame[["unit", "cycle"]].to_numpy(float)
        if (not np.isfinite(identifiers).all() or (identifiers < 1).any() or
                not np.equal(identifiers, np.floor(identifiers)).all()):
            raise ValueError("Recovery validation engine identifiers and cycles must be positive finite integers.")
        if validation_frame.duplicated(["unit", "cycle"]).any():
            raise ValueError("Recovery validation contains duplicate engine cycles.")
        if not np.isfinite(validation_frame[[*SETTINGS, *SENSORS]].to_numpy(float)).all():
            raise ValueError("Recovery validation references and settings must be finite before simulation.")
        state = getattr(scaler, "state", None)
        if not isinstance(state, dict) or "fit_units" not in state:
            raise ValueError("Recovery validation requires scaler fitting-engine provenance.")
        self.validation_units = sorted(int(unit) for unit in validation_frame.unit.unique())
        if set(self.validation_units) & set(state["fit_units"]):
            raise ValueError("Recovery validation engines overlap the scaler's fitting engines.")
        if transform_spec(state) != transform_spec(config):
            raise ValueError("Recovery validation scaler and model representations disagree.")
        self.severities = _severities(config)
        self.seed = config["validation_seed"] if seed is None else seed
        if (isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0):
            raise ValueError("Recovery validation seed must be a nonnegative integer.")
        self.seed = int(self.seed)
        _positive_integer(config["validation_prefixes"], "validation_prefixes")
        _positive_integer(config["sequence_length"], "sequence_length")
        self.config = {**config, "maximum_windows": None, "maximum_batches": None}
        self.representation = transform_spec(state)
        self.frame = validation_frame.sort_values(["unit", "cycle"]).copy()
        cases = [("clean", 0.), *((family, severity) for family in RECOVERY_FAMILIES[1:]
                                  for severity in self.severities)]
        self.case_keys = tuple(cases)
        self.datasets = {(family, severity): WindowDataset(self.frame, scaler, self.config,
            mode="validation", seed=self.seed, family=family, severity=severity)
            for family, severity in cases}
        if any(not len(dataset) for dataset in self.datasets.values()):
            raise ValueError("Every recovery validation context needs at least one observed window.")


def _linear_difference(a, b, representation):
    """Original linear-z differences, avoiding cancellation of large inverses."""
    method, scale = representation
    a, b = a.double(), b.double()
    if method == "linear":
        return a - b
    equal = a == b
    # Equal values have zero error even if their separate inverse would overflow.
    midpoint = ((a + b) / (2 * scale)).masked_fill(equal, 0.)
    half_delta = ((a - b) / (2 * scale)).masked_fill(equal, 0.)
    return 2 * scale * torch.cosh(midpoint) * torch.sinh(half_delta)


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _case_audit(model, dataset, family, severity, config, representation, use_repair, max_batches):
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    batches = DataLoader(dataset, batch_size=_positive_integer(config["batch_size"], "batch_size"),
                         shuffle=False, num_workers=0)
    channel_squared = np.zeros(21, dtype=float)
    channel_count = np.zeros(21, dtype=np.int64)
    changed_count = seen = nonfinite = 0
    encoded_changed_squared = linear_changed_squared = linear_input_squared = 0.
    window_errors = []
    for batch_index, batch in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        x, reference = batch["x"].to(device), batch["reference"].to(device)[..., 3:]
        valid = batch["valid"].to(device).bool().unsqueeze(-1).expand_as(reference)
        changed = batch["altered"].to(device).bool() & valid
        intact = ~batch["altered"].to(device).bool() & valid
        result = model(x, batch["valid"].to(device), use_repair=use_repair, use_correction=False)
        repaired = result["repaired"]
        if repaired.shape != reference.shape:
            raise ValueError("Recovery validation model returned an unexpected sensor shape.")
        observed = result["observed"] if "observed" in result else torch.nan_to_num(x[..., 3:], nan=0., posinf=0., neginf=0.)
        invalid_repaired = ~torch.isfinite(repaired) & valid
        # Padding is excluded before inverse diagnostics; it cannot contaminate
        # errors through 0*inf or NaN multiplication.
        repaired, reference = repaired.masked_fill(~valid, 0.), reference.masked_fill(~valid, 0.)
        observed = observed.masked_fill(~valid, 0.)
        linear_error = _linear_difference(repaired, reference, representation)
        input_error = _linear_difference(observed, reference, representation)
        nonfinite += int((invalid_repaired | (~torch.isfinite(linear_error) & valid) |
                          (~torch.isfinite(input_error) & valid)).sum())
        healthy_squared = linear_error.masked_fill(~intact, 0.).square()
        channel_squared += healthy_squared.sum((0, 1)).cpu().numpy()
        channel_count += intact.sum((0, 1)).cpu().numpy()
        encoded_changed_squared += float((repaired - reference).masked_fill(~changed, 0.).double().square().sum())
        linear_changed_squared += float(linear_error.masked_fill(~changed, 0.).square().sum())
        linear_input_squared += float(input_error.masked_fill(~changed, 0.).square().sum())
        changed_count += int(changed.sum())
        per_window_count = intact.sum((1, 2)).cpu().numpy()
        per_window_squared = healthy_squared.sum((1, 2)).cpu().numpy()
        for unit, cycle, squared, count in zip(batch["unit"], batch["cycle"], per_window_squared, per_window_count):
            if count:
                window_errors.append({"unit": int(unit), "cycle": int(cycle),
                    "linear_intact_RMSE": float(np.sqrt(squared / count)), "intact_cells": int(count)})
        seen += len(x)
    intact_count = int(channel_count.sum())
    channel_rmse = np.sqrt(np.divide(channel_squared, channel_count,
        out=np.zeros(21, dtype=float), where=channel_count > 0))
    window_rmse = np.asarray([row["linear_intact_RMSE"] for row in window_errors])
    intact_rmse = float(np.sqrt(channel_squared.sum() / intact_count)) if intact_count else 0.
    ratio = (float(np.sqrt(linear_changed_squared / linear_input_squared)) if linear_input_squared > 0 else
             0. if linear_changed_squared == 0 else math.inf) if changed_count else None
    encoded_rmse = float(np.sqrt(encoded_changed_squared / changed_count)) if changed_count else None
    values = {"linear_intact_repair_RMSE": intact_rmse,
        "linear_intact_channel_RMSE_max": float(channel_rmse.max()),
        "linear_intact_window_RMSE_p95": float(np.quantile(window_rmse, .95)) if len(window_rmse) else 0.,
        "linear_intact_window_RMSE_p99": float(np.quantile(window_rmse, .99)) if len(window_rmse) else 0.,
        "linear_intact_window_RMSE_max": float(window_rmse.max()) if len(window_rmse) else 0.,
        "reconstruction_RMSE": encoded_rmse,
        "linear_reconstruction_RMSE": float(np.sqrt(linear_changed_squared / changed_count)) if changed_count else None,
        "linear_input_RMSE": float(np.sqrt(linear_input_squared / changed_count)) if changed_count else None,
        "linear_reconstruction_ratio": ratio}
    finite = nonfinite == 0 and all(value is None or np.isfinite(value) for value in values.values())
    guards = {"finite": bool(finite),
              "intact": bool(np.isfinite(intact_rmse) and intact_rmse <= _budget(config, "max_intact_repair_rmse", .15))}
    if family != "clean" and changed_count:
        guards["linear_reconstruction"] = bool(np.isfinite(ratio) and ratio <= _budget(config, "max_linear_reconstruction_ratio", 1.))
    for key, metric, name in [
            ("recovery_validation_max_channel_intact_rmse", "linear_intact_channel_RMSE_max", "intact_channel"),
            ("recovery_validation_max_window_intact_rmse", "linear_intact_window_RMSE_max", "intact_window"),
            ("recovery_validation_max_reconstruction_rmse", "reconstruction_RMSE", "reconstruction")]:
        budget = _budget(config, key)
        if budget is not None and values[metric] is not None:
            guards[name] = bool(np.isfinite(values[metric]) and values[metric] <= budget)
    worst_windows = sorted(window_errors, key=lambda row: (not np.isfinite(row["linear_intact_RMSE"]),
        row["linear_intact_RMSE"] if np.isfinite(row["linear_intact_RMSE"]) else math.inf), reverse=True)[:5]
    return {"family": family, "severity": float(severity), "windows": seen, "expected_windows": len(dataset),
        "complete": seen == len(dataset), "intact_cells": intact_count, "changed_cells": changed_count,
        "nonfinite_entries": nonfinite, "guards": guards, "accepted": all(guards.values()),
        "failed_guards": [name for name, passed in guards.items() if not passed],
        **{name: None if value is None else _number(value) for name, value in values.items()},
        "channel_intact_RMSE": [{"sensor": sensor, "intact_cells": int(count), "RMSE": _number(value) if count else None}
            for sensor, value, count in zip(SENSORS, channel_rmse, channel_count)],
        "worst_windows": [{**row, "linear_intact_RMSE": _number(row["linear_intact_RMSE"])} for row in worst_windows]}


@torch.no_grad()
def audit_recovery_candidate(model, matrix, config=None, *, use_repair=True, max_batches=None, stop_on_failure=False):
    """Certify only a complete scan of the declared held-validation matrix.

    A ``max_batches`` call is a shortlist audit and can reject a candidate, but
    never certifies acceptance, even when a small fixture fits within that cap.
    ``stop_on_failure`` can cheaply reject; early exits remain incomplete. Full
    acceptance requires all contexts and windows, with correction disabled.

    Channel/window maximum guards and an all-context encoded reconstruction cap
    are optional: ``recovery_validation_max_channel_intact_rmse``,
    ``recovery_validation_max_window_intact_rmse``, and
    ``recovery_validation_max_reconstruction_rmse`` default to None. Tail
    diagnostics are always reported. Existing primary mixed-RUL/encoded-recovery
    and clean RMSE/NASA guards remain the caller's separate responsibility.
    """
    if not isinstance(matrix, RecoveryValidationMatrix):
        raise TypeError("Expected a cached RecoveryValidationMatrix.")
    config = matrix.config if config is None else config
    if _severities(config) != matrix.severities:
        raise ValueError("Recovery validation severities changed after matrix construction.")
    if tuple(matrix.datasets) != matrix.case_keys:
        raise ValueError("Recovery validation matrix contexts were removed, replaced, or reordered.")
    if any(config.get(name) != matrix.config.get(name) for name in CACHED_SCENARIO_OPTIONS):
        raise ValueError("Recovery validation scenario configuration changed after caching.")
    if transform_spec(getattr(model, "config", config)) != matrix.representation:
        raise ValueError("Recovery validation model and cached representation disagree.")
    _positive_integer(config["batch_size"], "batch_size")
    for key, default in [("max_intact_repair_rmse", .15), ("max_linear_reconstruction_ratio", 1.)]:
        if _budget(config, key, default) is None:
            raise ValueError(f"{key} cannot be None.")
    for key in ["recovery_validation_max_channel_intact_rmse", "recovery_validation_max_window_intact_rmse",
                "recovery_validation_max_reconstruction_rmse"]:
        _budget(config, key)
    if max_batches is not None:
        max_batches = _positive_integer(max_batches, "max_batches")
    contexts = []
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for (family, severity), dataset in matrix.datasets.items():
            if dataset.epoch != 0 or dataset.progress != 1.:
                raise ValueError("Cached recovery validation trajectories were resampled or mutated.")
            report = _case_audit(model, dataset, family, severity, config, matrix.representation, use_repair, max_batches)
            contexts.append(report)
            if stop_on_failure and not report["accepted"]:
                break
    finally:
        for module, training in states:
            module.train(training)
    complete = (max_batches is None and len(contexts) == len(matrix.datasets) and
                all(context["complete"] for context in contexts))
    worst, worst_contexts = {}, {}
    for metric in ["linear_intact_repair_RMSE", "linear_intact_channel_RMSE_max", "linear_intact_window_RMSE_p95",
                   "linear_intact_window_RMSE_p99", "linear_intact_window_RMSE_max", "reconstruction_RMSE",
                   "linear_reconstruction_RMSE", "linear_reconstruction_ratio"]:
        measured = [context for context in contexts if context[metric] is not None]
        context = max(measured, key=lambda row: row[metric]) if measured else None
        worst[metric] = context[metric] if context else None
        worst_contexts[metric] = ({"family": context["family"], "severity": context["severity"], "value": context[metric]}
                                  if context else None)
    failing = [{"family": context["family"], "severity": context["severity"], "failed_guards": context["failed_guards"]}
               for context in contexts if not context["accepted"]]
    worst["contexts"] = worst_contexts
    worst["nonfinite_contexts"] = sum(not context["guards"]["finite"] for context in contexts)
    return {"complete": bool(complete), "accepted": bool(complete and not failing), "contexts": contexts,
        "worst": worst, "failing_contexts": failing, "expected_contexts": len(matrix.datasets),
        "evaluated_contexts": len(contexts), "validation_units": matrix.validation_units, "seed": matrix.seed,
        "families": list(RECOVERY_FAMILIES), "severities": list(matrix.severities),
        "use_repair": bool(use_repair), "use_correction": False,
        "mode": "shortlist" if max_batches is not None else "exact",
        "prefix_policy": {"mode": "seeded_validation_prefixes", "prefixes_per_engine": matrix.config["validation_prefixes"]},
        "validation_prefixes": int(matrix.config["validation_prefixes"]),
        "sequence_length": int(matrix.config["sequence_length"]),
        "scenario_options": {name: matrix.config.get(name) for name in CACHED_SCENARIO_OPTIONS},
        "representation": list(matrix.representation), "guard_limits": _guard_limits(config),
        "recovery_state_sha256": recovery_state_sha256(model),
        "scope": "Training-held validation engines and declared simulated contexts only; no official test reads. "
                 "Window-weighted healthy errors; this audit does not guarantee unseen/fleet performance or replace clean RUL/NASA guards."}
