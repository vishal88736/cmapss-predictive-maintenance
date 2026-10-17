"""Engine-disjoint data preparation and persistent synthetic sensor corruption."""

import copy
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from .config import COLUMNS, SETTINGS, SENSORS

FAULTS = ["intermittent", "outage", "block", "noise", "bias", "drift"]


def load_subset(data_dir, subset):
    root = Path(data_dir)
    train = pd.read_csv(root / f"train_{subset}.txt", sep=r"\s+", header=None).dropna(axis=1, how="all")
    test = pd.read_csv(root / f"test_{subset}.txt", sep=r"\s+", header=None).dropna(axis=1, how="all")
    if train.shape[1] != 26 or test.shape[1] != 26:
        raise ValueError(f"{subset}: expected 26 trajectory columns.")
    train.columns = test.columns = COLUMNS
    truth = np.loadtxt(root / f"RUL_{subset}.txt", ndmin=1)
    if len(truth) != test.unit.nunique():
        raise ValueError(f"{subset}: RUL count differs from test engine count.")
    for frame in [train, test]:
        frame[["unit", "cycle"]] = frame[["unit", "cycle"]].astype(int)
        frame.sort_values(["unit", "cycle"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame.duplicated(["unit", "cycle"]).any():
            raise ValueError("Duplicate engine cycles in source data.")
    return train, test, dict(zip(sorted(test.unit.unique()), truth.tolist()))


def load_training_subset(data_dir, subset):
    """Hyperparameter search has no reason to open official test inputs/labels."""
    frame = pd.read_csv(Path(data_dir) / f"train_{subset}.txt", sep=r"\s+", header=None).dropna(axis=1, how="all")
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"{subset}: expected 26 trajectory columns.")
    frame.columns = COLUMNS
    frame[["unit", "cycle"]] = frame[["unit", "cycle"]].astype(int)
    if frame.duplicated(["unit", "cycle"]).any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Training trajectories must have unique cycles and finite references.")
    return frame.sort_values(["unit", "cycle"]).reset_index(drop=True)


def source_hashes(data_dir, subset):
    return {name: hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest()
            for name in [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]}


def split_engines(frame, seed, maximum_engines=None, validation_fraction=.20, calibration_fraction_of_training=.10):
    """Outer 80/20 split; reserve calibration inside the 80% training pool.

    Returned fit/validation/calibration lists are mutually disjoint. The training
    pool is fit + calibration; only fit engines enter optimizers or normalization.
    """
    if not 0 < validation_fraction < 1 or not 0 < calibration_fraction_of_training < 1:
        raise ValueError("Validation and calibration fractions must be between zero and one.")
    units = np.sort(frame.unit.unique())
    units = np.random.default_rng(seed).permutation(units)
    if maximum_engines:
        units = units[:maximum_engines]
    if len(units) < 10:
        raise ValueError("At least ten training engines are needed for disjoint partitions.")
    nval = max(2, int(np.ceil(len(units) * validation_fraction)))
    ntrain = len(units) - nval
    ncal = max(2, int(np.ceil(ntrain * calibration_fraction_of_training)))
    nfit = ntrain - ncal
    if nfit < 3:
        raise ValueError("The fractions leave too few fitting engines for three-fold correction.")
    groups = {"fit": units[:nfit].tolist(), "calibration": units[nfit:ntrain].tolist(), "validation": units[ntrain:].tolist()}
    assert not (set(groups["fit"]) & set(groups["validation"]) or set(groups["fit"]) & set(groups["calibration"]) or set(groups["validation"]) & set(groups["calibration"]))
    return groups


class OperatingRegimeScaler:
    def __init__(self, state=None):
        self.state = state

    def fit(self, frame, subset, seed):
        settings = frame[SETTINGS].to_numpy(float)
        sm, ss = settings.mean(0), np.maximum(settings.std(0), 1e-6)
        normalized = (settings - sm) / ss
        k = 6 if subset in {"FD002", "FD004"} else 1
        centers = KMeans(k, n_init=10, random_state=seed).fit(normalized).cluster_centers_ if k > 1 else normalized.mean(0, keepdims=True)
        regimes = ((normalized[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        sensors = frame[SENSORS].to_numpy(float)
        # Quantized, almost-constant regime channels otherwise turn one normal
        # measurement increment into tens of standard deviations (e.g. FD002
        # sensor_10). Estimate a per-channel floor from fitting observations only.
        floors = []
        for column in sensors.T:
            increments = np.diff(np.unique(column))
            floors.append(max(1e-6, float(increments.min())) if len(increments) else 1.)
        floors = np.asarray(floors)
        means, stds = [], []
        for r in range(k):
            rows = sensors[regimes == r]
            if len(rows) < 10:
                rows = sensors
            means.append(rows.mean(0)); stds.append(np.maximum(rows.std(0), floors))
        self.state = {"setting_mean": sm.tolist(), "setting_std": ss.tolist(), "centers": centers.tolist(),
                      "sensor_mean": np.asarray(means).tolist(), "sensor_std": np.asarray(stds).tolist(),
                      "sensor_scale_floor": floors.tolist(),
                      "fit_units": sorted(int(i) for i in frame.unit.unique())}
        return self

    def transform(self, frame):
        s = self.state
        settings = (frame[SETTINGS].to_numpy(float) - np.asarray(s["setting_mean"])) / np.asarray(s["setting_std"])
        centers = np.asarray(s["centers"])
        regimes = ((settings[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
        sensors = (frame[SENSORS].to_numpy(float) - np.asarray(s["sensor_mean"])[regimes]) / np.asarray(s["sensor_std"])[regimes]
        return np.concatenate([settings, sensors], -1).astype(np.float32), regimes


def corrupt_trajectory(reference, rng, family=None, severity=.2, clean_fraction=.25, progress=1.,
                       *, start_horizon=100, drift_ramp_cycles=100, block_gap_cycles=50):
    """Generate persistent faults without using future trajectory length.

    Equal observed prefixes and RNG seeds produce equal corrupted prefixes, even
    for mixed faults. Timing is in observed rows, which are consecutive cycles in
    C-MAPSS. Fault labels supervise losses only; availability remains separate.
    """
    if isinstance(start_horizon, bool) or not isinstance(start_horizon, (int, np.integer)) or start_horizon < 0:
        raise ValueError("start_horizon must be a nonnegative integer.")
    if isinstance(drift_ramp_cycles, bool) or not isinstance(drift_ramp_cycles, (int, np.integer)) or drift_ramp_cycles < 1:
        raise ValueError("drift_ramp_cycles must be a positive integer.")
    if isinstance(block_gap_cycles, bool) or not isinstance(block_gap_cycles, (int, np.integer)) or block_gap_cycles < 1:
        raise ValueError("block_gap_cycles must be a positive integer.")
    x = reference.copy()
    altered = np.zeros(x[:, 3:].shape, dtype=bool)
    if family == "clean" or (family is None and rng.random() < clean_fraction):
        return x, altered
    if family == "mixed":
        families = rng.choice(FAULTS, size=2, replace=False).tolist()
    else:
        families = [family] if family else rng.choice(FAULTS, size=2 if progress > .4 and rng.random() < .6 else 1, replace=False).tolist()
    severity = float(severity if family else rng.choice([.1, .2, .3]) * max(.3, progress))
    n, c = altered.shape
    # A longer stochastic fault must not shift another fault's parameters/draws.
    # Derive every stream before generating arrays whose sizes depend on n.
    streams = [np.random.default_rng(rng.integers(0, np.iinfo(np.int64).max)) for _ in families]
    for fault, fault_rng in zip(families, streams):
        channels = fault_rng.choice(c, max(1, round(c * severity)), replace=False)
        start = int(fault_rng.integers(0, start_horizon + 1))
        if fault == "intermittent":
            mask = fault_rng.random((n, c)) < severity
            x[:, 3:][mask] = np.nan; altered |= mask
        elif fault == "outage":
            x[start:, 3 + channels] = np.nan; altered[start:, channels] = True
        elif fault == "block":
            span = max(5, int(100 * severity))
            # Recurring temporary losses remain possible late in a trajectory.
            # Every event is generated in order, independently of its final length.
            while start < n:
                x[start:start + span, 3 + channels] = np.nan
                gap = int(fault_rng.integers(max(1, block_gap_cycles // 2), block_gap_cycles + 1))
                start += span + gap
        elif fault == "noise":
            sigma = severity / 6  # .1/.2/.3 correspond approximately to A's mild noise range.
            x[:, 3 + channels] += fault_rng.normal(0, sigma, (n, len(channels)))
            altered[:, channels] = True
        elif fault == "bias":
            offsets = fault_rng.uniform(-2.5 * severity, 2.5 * severity, len(channels))
            x[start:, 3 + channels] += offsets; altered[start:, channels] = True
        elif fault == "drift":
            slopes = fault_rng.uniform(-severity, severity, len(channels))
            elapsed = np.arange(max(0, n - start), dtype=np.float32)
            ramp = np.minimum(elapsed / drift_ramp_cycles, 1.)[:, None]
            x[start:, 3 + channels] *= 1 + ramp * slopes; altered[start:, channels] = True
        else:
            raise ValueError(f"Unknown fault family: {fault}")
    # Multiplicative drift cannot alter an exactly zero standardized reading,
    # and its first ramp point is unchanged. Do not teach the detector to flag
    # measurements that remain identical to their reference.
    altered = x[:, 3:] != reference[:, 3:]
    return x, altered


def make_window(values, end, length):
    window = values[max(0, end - length + 1):end + 1]
    result = np.zeros((length, values.shape[-1]), dtype=np.float32)
    valid = np.zeros(length, dtype=bool)
    result[-len(window):] = window; valid[-len(window):] = True
    return result, valid


class WindowDataset(Dataset):
    def __init__(self, frame, scaler, config, mode="train", seed=42, family=None, severity=.2, truths=None):
        self.config, self.seed, self.family, self.severity = config, int(seed), family, severity
        self.epoch, self.progress, self.cache = 0, 1., {}
        self.engines, self.samples = [], []
        rng = np.random.default_rng(seed)
        for unit, group in frame.groupby("unit", sort=True):
            values, regimes = scaler.transform(group)
            cycles = group.cycle.to_numpy()
            y = (truths[int(unit)] + cycles[-1] - cycles) if truths is not None else cycles[-1] - cycles
            engine_index = len(self.engines)
            self.engines.append({"unit": int(unit), "values": values, "cycles": cycles,
                                 "regimes": regimes, "targets": y.astype(np.float32)})
            if mode == "train":
                ends = np.unique(np.r_[np.arange(0, len(values), config["window_stride"]), len(values) - 1])
            elif mode == "test":
                ends = [len(values) - 1]
            else:
                count = config["correction_prefixes"] if mode == "correction" else config["validation_prefixes"] if mode == "validation" else 1
                # A seed per engine keeps endpoints stable across model variants,
                # even when a fold contains a different list of engines.
                endpoint_rng = np.random.default_rng(np.random.SeedSequence([seed, int(unit), 991]))
                available = np.arange(min(19, len(values) - 1), len(values))
                ends = np.sort(endpoint_rng.choice(available, min(count, len(available)), replace=False))
            self.samples.extend((engine_index, int(end)) for end in ends)
        if mode == "train" and config.get("maximum_windows") and len(self.samples) > config["maximum_windows"]:
            self.samples = [self.samples[i] for i in sorted(rng.choice(len(self.samples), config["maximum_windows"], replace=False))]

    def set_epoch(self, epoch, progress=1.):
        self.epoch, self.progress, self.cache = int(epoch), progress, {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        engine_index, end = self.samples[index]
        engine = self.engines[engine_index]
        if engine_index not in self.cache:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, engine["unit"]]))
            self.cache[engine_index] = corrupt_trajectory(
                engine["values"], rng, self.family, self.severity, self.config["clean_fraction"], self.progress,
                start_horizon=self.config.get("fault_start_horizon", 100),
                drift_ramp_cycles=self.config.get("drift_ramp_cycles", 100),
                block_gap_cycles=self.config.get("block_gap_cycles", 50))
        corrupt, altered = self.cache[engine_index]
        length = self.config["sequence_length"]
        reference, valid = make_window(engine["values"], end, length)
        damaged, _ = make_window(corrupt, end, length)
        labels, _ = make_window(altered.astype(np.float32), end, length)
        raw_target = float(engine["targets"][end])
        return {"x": torch.from_numpy(damaged), "reference": torch.from_numpy(reference),
                "valid": torch.from_numpy(valid), "altered": torch.from_numpy(labels).bool(),
                "target": torch.tensor(min(self.config["train_rul_cap"], raw_target), dtype=torch.float32),
                "raw_target": torch.tensor(raw_target, dtype=torch.float32), "unit": engine["unit"], "cycle": int(engine["cycles"][end])}


def select_background_indices(dataset, size, seed):
    """Balance fitting references over regimes/life stages and prefer new engines.

    This selects indices only; it does not inspect test data or model predictions.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Background size must be a positive integer.")
    groups, metadata = {}, {}
    for index in np.random.default_rng(seed).permutation(len(dataset)).tolist():
        engine_index, end = dataset.samples[index]
        engine = dataset.engines[engine_index]
        regime = int(engine["regimes"][end]) if engine["regimes"] is not None else 0
        target = min(float(engine["targets"][end]), dataset.config["train_rul_cap"])
        stage = min(3, int(3 * target / dataset.config["target_scale"]))
        groups.setdefault((regime, stage), []).append(index)
        metadata[index] = {"unit": engine["unit"], "cycle": int(engine["cycles"][end]),
                           "regime": regime, "life_stage": stage, "capped_target": target}
    selected, engines = [], set()
    strata = sorted(groups)
    # Exhaust distinct engines before accepting another example from one engine.
    for prefer_new in [True, False]:
        while len(selected) < min(size, len(dataset)):
            added = False
            for stratum in strata:
                candidates = groups[stratum]
                position = next((i for i, idx in enumerate(candidates)
                                 if not prefer_new or metadata[idx]["unit"] not in engines), None)
                if position is None:
                    continue
                index = candidates.pop(position)
                selected.append(index); engines.add(metadata[index]["unit"])
                added = True
                if len(selected) == min(size, len(dataset)):
                    break
            if not added:
                break
    return selected, [metadata[index] for index in selected]


def window_view(dataset, length):
    """Change observed context while sharing endpoints and cached trajectory faults."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("Window length must be a positive integer.")
    if length == dataset.config["sequence_length"]:
        return dataset
    view = copy.copy(dataset)
    view.config = {**dataset.config, "sequence_length": length}
    return view
