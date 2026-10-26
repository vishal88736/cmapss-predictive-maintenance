"""Causal sensor-error scenarios, independent of the neural training runtime.

These are explicit simulation assumptions, not identified C-MAPSS sensor failures.
Raw-unit scales come from fitting engines; no RUL or future trajectory statistics
enter the simulator. See docs/STRUCTURED_SENSOR_FAULTS.md for equations and limits.
"""

import numpy as np

from .config import CORRUPTION_PROTOCOL, SENSORS
from .normalization import decode_sensors, encode_sensors, transform_spec

FAULTS = ["intermittent", "outage", "block", "noise", "bias", "drift", "stuck"]

# Measurement domains are hypotheses for common-mode errors, not known wiring.
# Demanded speeds (18, 19 in the source data) are commands, not failed sensors.
SENSOR_DOMAINS = {
    "temperature": (0, 1, 2, 3),
    "pressure": (4, 5, 6, 10),
    "fan_speed": (7, 12),
    "core_speed": (8, 13),
    "coolant_flow": (19, 20),
    "pressure_ratio": (9,),
    "fuel_flow_ratio": (11,),
    "bypass_ratio": (14,),
    "fuel_air_ratio": (15,),
    "bleed_enthalpy": (16,),
}
MIXED_SCENARIOS = {
    "connector": ("noise", "intermittent"),
    "calibration": ("bias", "drift"),
    "acquisition": ("stuck", "block"),
}


def _positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _physical_context(reference, profile, regimes):
    """Obtain fixed raw-unit scales and a causal operating-transition signal."""
    n = len(reference)
    if profile is None:
        # Standalone synthetic tests use unit sensor scales. Real datasets always
        # supply a profile fitted by OperatingRegimeScaler on fitting engines.
        means = np.zeros((n, 21))
        stds = np.ones((n, 21))
        scales = np.ones(21)
        settings = reference[:, :3]
    else:
        if "sensor_fault_scale" not in profile:
            raise ValueError("This scaler predates structured sensor faults; fit a new scaler on fitting engines.")
        centers = np.asarray(profile["centers"], float)
        r = np.asarray(regimes)
        if r.shape != (n,) or r.dtype.kind not in "iu" or np.any((r < 0) | (r >= len(centers))):
            raise ValueError("One valid fitting-regime index is required per observation.")
        mu, sd = np.asarray(profile["sensor_mean"], float), np.asarray(profile["sensor_std"], float)
        scales = np.asarray(profile["sensor_fault_scale"], float)
        if (mu.shape != (len(centers), 21) or sd.shape != mu.shape or scales.shape != (21,) or
                centers.shape != (len(centers), 3) or not all(np.isfinite(a).all() for a in [mu, sd, scales, centers]) or
                np.any(sd <= 0) or np.any(scales <= 0)):
            raise ValueError("Invalid fitting-only sensor fault profile.")
        means, stds = mu[r], sd[r]
        # Regime centers ignore tiny setting jitter in single-condition subsets.
        settings = centers[r]
    transition = np.zeros(n)
    if n > 1:
        transition[1:] = np.minimum(1., np.linalg.norm(np.diff(settings, axis=0), axis=1) / np.sqrt(3))
    stress = np.zeros(n)
    for t in range(1, n):
        stress[t] = .8 * stress[t - 1] + .2 * transition[t]
    return means, stds, scales, stress


def corrupt_trajectory(reference, rng, family=None, severity=.2, clean_fraction=.25, progress=1.,
                       *, start_horizon=100, drift_ramp_cycles=100, block_gap_cycles=50,
                       noise_autocorrelation=.9, transition_gain=1., intermittent_mean_burst=4,
                       profile=None, regimes=None, return_metadata=False):
    """Apply one sensor-domain failure or a related pair of failure mechanisms.

    Equal reference prefixes, fitted profiles and seeds give equal damaged
    prefixes. Severity controls mechanism magnitude/duration and domain coverage;
    it is not a percentage of damaged entries. Settings and speed commands stay
    intact. Metadata and altered-entry labels are audit/supervision only.
    """
    _positive_integer(start_horizon, "start_horizon", 0)
    _positive_integer(drift_ramp_cycles, "drift_ramp_cycles")
    _positive_integer(block_gap_cycles, "block_gap_cycles")
    _positive_integer(intermittent_mean_burst, "intermittent_mean_burst", 2)
    if not np.isfinite(noise_autocorrelation) or not 0 <= noise_autocorrelation < 1:
        raise ValueError("noise_autocorrelation must be finite and in [0, 1).")
    if not np.isfinite(transition_gain) or not 0 <= transition_gain <= 3:
        raise ValueError("transition_gain must be finite and in [0, 3].")
    if not np.isfinite(severity) or not (0 < severity <= .5 or (family == "clean" and severity == 0)):
        raise ValueError("severity must be finite and in (0, .5].")
    if not np.isfinite(clean_fraction) or not 0 <= clean_fraction <= 1:
        raise ValueError("clean_fraction must be in [0, 1].")
    if not np.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError("progress must be in [0, 1].")
    if family is not None and family not in [*FAULTS, "mixed", "clean"]:
        raise ValueError(f"Unknown fault family: {family}")
    reference = np.asarray(reference)
    if (reference.ndim != 2 or reference.shape[1] != 24 or not len(reference) or
            reference.dtype.kind != "f" or not np.isfinite(reference).all()):
        raise ValueError("Reference must be a nonempty finite floating array with 24 columns.")
    compressed = transform_spec(profile)[0] == "asinh"
    # Fault equations retain the original linear z-score/raw-unit semantics.
    # Compression is an invertible model representation, not a smaller fault.
    linear_reference = reference.astype(np.float64) if compressed else reference
    if compressed:
        linear_reference[:, 3:] = decode_sensors(reference[:, 3:], profile)
    x = linear_reference.copy()
    metadata = {"protocol": CORRUPTION_PROTOCOL, "requested_family": family, "mechanisms": [],
                "domain": None, "channels": [], "onset_observed_offset": None,
                "scale_source": "fitting_engines" if profile is not None else "unit_scale_synthetic_reference"}

    def result():
        if compressed:
            changed = x[:, 3:] != linear_reference[:, 3:]
            output = reference.copy()
            # Do not round-trip intact values: their exact stored bits and
            # changed-entry labels must survive simulation unchanged.
            output[:, 3:][changed] = encode_sensors(x[:, 3:], profile)[changed]
        else:
            output = x
        altered = output[:, 3:] != reference[:, 3:]
        return (output, altered, metadata) if return_metadata else (output, altered)

    if family == "clean" or (family is None and rng.random() < clean_fraction):
        return result()
    if family is None:
        severity = float(rng.choice([.1, .2, .3]) * max(.3, progress))
        family = "mixed" if progress > .4 and rng.random() < .6 else str(rng.choice(FAULTS))
    if family == "mixed":
        scenario = str(rng.choice(list(MIXED_SCENARIOS)))
        families = MIXED_SCENARIOS[scenario]
        metadata["scenario"] = scenario
    else:
        families = (family,)
        metadata["scenario"] = family
    domain = str(rng.choice(list(SENSOR_DOMAINS)))
    members = SENSOR_DOMAINS[domain]
    count = min(len(members), max(1, int(np.ceil(2 * severity * len(members)))))
    start = int(rng.integers(0, start_horizon + 1))
    sign = float(rng.choice([-1, 1]))
    # Fixed-sized plan draws keep onset/source fixed across severity sweeps and
    # make sensor sets nested as the same scenario becomes more severe.
    member_coupling = rng.uniform(.85, 1., len(members))
    positions = np.sort(rng.permutation(len(members))[:count])
    channels = np.asarray(members)[positions]
    coupling = member_coupling[positions]
    # Derive all streams before any trajectory-length-dependent random draws.
    streams = [np.random.default_rng(rng.integers(0, np.iinfo(np.int64).max)) for _ in families]
    means, stds, scales, stress = _physical_context(reference, profile, regimes)
    metadata.update(domain=domain, channels=[SENSORS[j] for j in channels],
                    onset_observed_offset=start, severity=severity,
                    raw_unit_scales=scales[channels].tolist())
    active = max(0, len(x) - start)
    cols = 3 + channels
    for fault, fault_rng in zip(families, streams):
        details = {"family": fault}
        if fault == "intermittent":
            # A common contact/telemetry fault drops a group in bursts, not
            # independent sensor/cycle cells. Each cycle consumes two draws.
            draws = fault_rng.random((active, 2))
            down, age = True, 0
            hazard = severity / (intermittent_mean_burst * (1 - severity))
            for offset in range(active):
                t = start + offset
                if not down and draws[offset, 0] < min(.95, hazard * (1 + transition_gain * stress[t])):
                    down, age = True, 0
                if down:
                    x[t, cols] = np.nan
                    age += 1
                    if age >= 2 and draws[offset, 1] < 1 / (intermittent_mean_burst - 1):
                        down, age = False, 0
            details.update(mean_burst_cycles=intermittent_mean_burst, base_entry_hazard=hazard,
                           transition_gain=transition_gain)
        elif fault == "outage":
            x[start:, cols] = np.nan
        elif fault == "block":
            span, event = max(5, int(100 * severity)), start
            events = []
            while event < len(x):
                x[event:event + span, cols] = np.nan
                events.append(event)
                event += span + int(fault_rng.integers(max(1, block_gap_cycles // 2), block_gap_cycles + 1))
            details.update(block_cycles=span, gap_max_cycles=block_gap_cycles, event_starts=events)
        elif fault == "noise":
            # Draw a full domain even if only a subset is affected, so a severity
            # sweep retains the same innovations on its existing channels.
            innovations = fault_rng.normal(size=(active, 1 + len(members)))
            state = np.zeros(1 + len(members))
            sigma = severity * scales[channels]
            for offset in range(active):
                state = noise_autocorrelation * state + np.sqrt(1 - noise_autocorrelation ** 2) * innovations[offset]
                error = np.sqrt(.8) * state[0] + np.sqrt(.2) * state[1 + positions]
                t = start + offset
                x[t, cols] += sigma * error * (1 + transition_gain * stress[t]) / stds[t, channels]
            details.update(raw_noise_sigma=sigma.tolist(), autocorrelation=noise_autocorrelation,
                           common_innovation_variance_fraction=.8, transition_gain=transition_gain)
        elif fault in {"bias", "drift"}:
            bound = sign * fault_rng.uniform(.6, 1.) * 2.5 * severity * scales[channels] * coupling
            ramp = (np.minimum(np.arange(active) / drift_ramp_cycles, 1.) if fault == "drift" else np.ones(active))
            x[start:, cols] += ramp[:, None] * bound / stds[start:, channels]
            details.update(raw_offset_bound=bound.tolist())
            if fault == "drift":
                details["ramp_cycles"] = drift_ramp_cycles
        elif fault == "stuck":
            if active:
                # Hold the physical reading. Holding a normalized value would
                # silently change the physical reading when regimes change.
                held = linear_reference[start, cols] * stds[start, channels] + means[start, channels]
                x[start + 1:, cols] = (held - means[start + 1:, channels]) / stds[start + 1:, channels]
                details["held_raw_reading"] = held.tolist()
        metadata["mechanisms"].append(details)
    return result()
