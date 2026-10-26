# Structured sensor errors, protocol v4

## Format-6 representation note

Physical fault equations below operate in the original regime z-score/raw-unit
space. Format 6 wraps this simulation with invertible asinh model features;
it decodes before applying faults and re-encodes only changed readings. It does
not shrink the underlying physical fault or change its channel/onset plan.
Legacy format-5 profiles remain linear. See [RECOVERY_V6.md](RECOVERY_V6.md).

`structured-sensor-faults-v4` replaces unrelated channel corruption and independent
per-cell noise. The simulator is in `grace_rul/corruption.py`; `WindowDataset` applies
it to complete observed trajectories before making windows. The notebook previews
all eight scenarios in physical units and displays their selected domains/channels.

C-MAPSS already includes simulated measurement noise and engine degradation. Its
reference trajectories are not noise-free ground truth. Added errors represent
sensor/measurement or acquisition failures while preserving the original degradation
and RUL targets. Changing an engine component's health would require a separate
engine simulation and consistent new failure-time targets.

Real noise and intermittent contacts can be stochastic. Their temporal memory,
related channels, onset, duration and operating context make the scenarios structured.
The behavioral distinction between bias, additive drift, scaling, noise and hard
failures follows [NASA's sensor-fault modeling study](https://ntrs.nasa.gov/citations/20090033812).
That study does not establish the numerical parameters or channel wiring used here.

## Sensor domains and related mechanisms

Each scenario selects one of these measurement domains. More severe scenarios cover
more members of that domain, rather than unrelated sensors across the engine.

| Domain | Source channels / names |
| --- | --- |
| Temperature | 1–4: T2, T24, T30, T50 |
| Pressure | 5, 6, 7, 11: P2, P15, P30, Ps30 |
| Fan speed | 8, 13: Nf, NRf |
| Core speed | 9, 14: Nc, NRc |
| Coolant flow | 20, 21: W31, W32 |
| Individual derived quantities | 10: epr; 12: phi; 15: BPR; 16: farB; 17: htBleed |

The groups are explicit hypotheses for common measurement/electronics errors,
supported by the variable definitions in [NASA's C-MAPSS model report](https://ntrs.nasa.gov/citations/20080043619).
They do not identify actual shared harnesses, buses or failed hardware. Source
channels 18–19 are demanded speeds, so this sensor-error experiment preserves them,
along with all three operating settings. Single derived quantities are not treated
as a common physical sensor group.

Mixed scenarios use the **same domain, selected channels and onset**, with independent
random streams for each mechanism:

| Scenario | Mechanisms | Assumed error source |
| --- | --- | --- |
| Connector | Correlated noise + intermittent dropout | Unstable contact or acquisition connection |
| Calibration | Fixed bias + additive drift, with a common sign | Offset error followed by calibration creep |
| Acquisition | Stuck reading + recurring missing blocks | Frozen acquisition followed by interrupted delivery |

These are modeling choices, not observed C-MAPSS failure probabilities or extra
validated reproductions of the three papers.

## Amplitudes, timing and equations

For sensor `j`, the fitting-only scaler saves a raw-unit reference scale `s[j]`:

1. Compute each fitting operating regime's `(p90 - p10) / 2.5631031311`.
2. Take the median across fitting regimes.
3. Floor it by the smallest observed positive sensor increment. For a globally
   constant channel, the explicit fallback is `max(1e-8, abs(fitting median) * 1e-4)`.

This robust spread includes natural simulation noise, manufacturing variation and
degradation. It is an **empirical perturbation scale**, not a pure noise estimate,
sensor tolerance, or identified physical fault magnitude. The constant-channel
fallback is an assumption because the source data cannot estimate its noise scale.
Test/validation/calibration engines do not fit this profile; each OOF model fits its
own profile using only its own fitting engines.

All finite errors are constructed in raw units and divided by the current regime's
normalization standard deviation before being added to the model inputs. This
preserves a fixed physical calibration offset across changing operating regimes.
Stuck readings are also held in physical units; holding a normalized number would
imply a different physical reading when the regime changes.

| Family | Behavior |
| --- | --- |
| Intermittent | A shared two-state dropout process, with bursts lasting at least two observed cycles and configured mean four. Base entry hazard is `severity / (mean_burst * (1 - severity))`; observed regime transitions increase entry risk. |
| Outage | Selected channels unavailable from onset onward. |
| Block | Recurring losses of `max(5, int(100 * severity))` cycles; gaps are sampled between half of `block_gap_cycles` and its maximum, default 50. |
| Noise | Added error has AR(1) memory, default `rho = .9`. Gaussian innovations combine 80% common-source variance with 20% channel-specific variance. Baseline raw standard deviation is `severity * s[j]`, increased during observed regime transitions. This is additional faulty noise, not replacement of existing reference noise. |
| Bias | One persistent raw offset with common sign and per-channel scaling; its magnitude is bounded by `2.5 * severity * s[j]`. |
| Drift | An additive raw offset grows linearly from zero at onset to the same magnitude bound over `drift_ramp_cycles`, default 100, then stays fixed. Zero standardized measurements can drift. |
| Stuck | Holds the physical reading at onset for the rest of the observed history. Unchanging reference readings yield no changed-entry labels. |

Operating-transition stress uses distances between the **fitted regime centers**
at consecutive observed cycles, clipped to one, then smoothed causally with weights
0.8 on the previous state and 0.2 on the new transition. The amplitude multiplier is
`1 + fault_transition_gain * stress`, default gain 1. Single-regime datasets do not
interpret small setting jitter as repeated operating transitions. This is a
transition-stress proxy; C-MAPSS cycle snapshots do not establish transient
vibration/temperature dynamics or permit interpreting this AR parameter in seconds.

`severity` is a dimensionless scenario control, **not the percentage of corrupted
cells or missing channels**. Training samples .1, .2 and .3, scaled by the existing
curriculum; full robustness evaluation also uses .4 and .5. Domain selection is
uniform and the selected count is `min(domain size, ceil(2 * severity * domain size))`.
These sampling rates are experiment assumptions. Fault metadata records actual
selected channels, onset, raw scales, offset/noise parameters and block event starts.
For fixed seeds, severity sweeps retain the same domain/onset and nested channel
sets. Noise, bias and drift retain their underlying draws on existing channels,
so increasing magnitude does not silently replace the fault source.

Onsets use absolute observed offsets 0 through `fault_start_horizon` (100), never
fractions of eventual engine lifetime or remaining life. Appending unseen future
observations cannot change an already corrupted prefix when the fitted profile
and seed are fixed. Operating stress uses present/past settings only. Overlapping
windows share one cached fault history. Labels mark only readings actually changed,
and remain supervision/audit information rather than inference inputs.

## Validation and remaining evidence

The regression checks cover domain membership, common-source and temporal noise,
bursty losses, additive drift on zero inputs, raw-unit bias/stuck behavior across
regimes, settings/command preservation, prefix invariance, fitting-only profiles,
and checkpoint identity. See [the validation report](STRUCTURED_FAULT_VALIDATION_2026-10-17.md)
and [recorded checks](STRUCTURED_FAULT_VALIDATION_2026-10-17.json).

With the project dependencies installed, reproduce the data audit without training:

```bash
python -m unittest discover -s tests -p 'test_sensor_faults.py'
python -m unittest discover -s tests -p 'test_corruption.py'
python scripts/audit_sensor_faults.py --output-dir artifacts/sensor_fault_audit_v4
```

The audit writes actual active-endpoint fractions, changed-entry fractions, domain
counts, onset ranges, fitting profiles, data hashes and example mechanism parameters.
It calculates no model accuracy. Add `--severities .1 .2 .3 .4 .5` for the full severity
sweep. This script uses the project's data loader and therefore requires PyTorch,
even though it performs no neural operations.

The numerical parameters need sensitivity analysis or calibration against recorded
sensor failures. No field-fault dataset was supplied. These mechanisms are more
structured than the prior iid perturbations; that does not establish real-world
generalization or improved RUL accuracy. Train and compare under the same v4 protocol,
report active-fault and unaffected cohorts separately, and keep clean performance
within the existing acceptance guard.

Use a **fresh run directory**: v3 weights/checkpoints describe a different training
distribution and cannot resume v4 training. Format-5 tensor shapes stay compatible
for inference on old bundles under their original training semantics. This environment
lacked PyTorch during the recorded 17 October 2026 validation, so neural tests,
retraining and actual accuracy improvement were not verified in that run.
