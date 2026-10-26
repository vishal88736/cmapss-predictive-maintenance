# Engine-wise prediction interval calibration

The predictor still learns `min(raw_RUL, 125)` by default. Capped and raw RUL
remain different target contracts. The interval repair changes uncertainty
calibration; it does not demonstrate improved point prediction accuracy or
teach the predictor a longer target horizon.

## Why the previous interval protocol was insufficient

The earlier format selected one seeded random prefix per held calibration
engine and used interpolated 5th/95th signed-residual percentiles. This did not
include the finite-sample conformal correction. The official test endpoint law
also was not established to match that random-prefix policy. Reusing the same
capped-target offsets to report uncapped coverage mixed two target definitions.

Taking the last row of each complete calibration trajectory would not repair
that problem: the training trajectories run to failure, so all those labels
would equal zero. Official test trajectories stop before failure. A last-row
calibration on complete trajectories would create a degenerate calibration set.

## Default: one simultaneous-prefix score per held engine

The model, normalization, and hyperparameters are fixed before calibration. The
calibration engines must be separate from both fitting and model-selection
engines. The calibration module checks fitting IDs recorded by the scaler; the
caller remains responsible for the separate validation partition.

1. Evaluate every observed prefix of each complete held calibration engine on
   unmodified measurements. No optimizer, preprocessing fit, official test
   inputs, or official test labels enter this step. Quick-run window/batch
   limits are disabled for calibration inference so engines and prefixes cannot
   silently disappear.
2. For each target convention, compute one score per engine:
   `s_i = max_t |y_it - prediction_it|`. Prefixes from the same engine are
   dependent and count together as **one** calibration unit.
3. With `n` independent calibration engines and error level `alpha`, use order
   statistic `k = ceil((n + 1) * (1 - alpha))`, without interpolation. If `k > n`,
   the radius is infinite. For a nominal 90% interval, two quick-run calibration
   engines therefore produce an honestly unbounded interval; 20 engines use
   rank 19 and 21 engines use rank 20.
4. Return `[max(0, prediction - radius), max(0, prediction + radius)]`. The
   nonnegative support restriction preserves coverage for nonnegative RUL.

Under exchangeability of complete engine trajectories, with a model fixed
independently of calibration engines, this construction gives at least
`1 - alpha` marginal probability that all prefixes of a new engine are covered
simultaneously. It consequently accommodates an endpoint selected within that
trajectory. It is conservative because an engine is scored by its worst prefix.
This is a conditional statement about an assumed data law, not an established
90% C-MAPSS or real-fleet guarantee. Differences in fleets, censoring/selection,
trajectory lengths, simulated faults, and OOD inputs can violate that assumption.
The guarantee is marginal over exchangeable engines, not conditional coverage
for each operating regime, lifecycle stage, or particular fitted calibration set.

`calibration_policy="random_prefix"` remains an explicit alternative for old
observation-policy experiments. It computes one seeded prefix score per engine
and uses the same finite-sample correction. Its coverage interpretation requires
the same prefix-sampling policy at deployment and does not resolve unknown
official test censoring. Last-row-only calibration is deliberately unsupported.

## Separate target contracts and saved metadata

The same held engines produce two separately calibrated sets of engine scores:

| Protocol | Target | Meaning |
| --- | --- | --- |
| `capped_125` (or the configured cap) | `min(raw_RUL, cap)` | Default training/evaluation target |
| `official_uncapped` | `raw_RUL` | Raw remaining cycles, using its own residual radius |

A raw-target radius can be large while the capped point model remains inaccurate
at long horizons. Calibrating that radius does not establish uncapped point
prediction quality. Neither clean-reference interval is a guarantee under sensor
faults or distribution shift.

New calibration records include `method`, `policy`, `primary_protocol`, engine
count, endpoint count, rank, per-engine scores, target convention, scope, and
independent protocol entries. Unbounded radii/offsets are saved as JSON `null`
with `unbounded=true`; `interval_bounds` converts this to `[0, infinity]` in
memory. They must not be displayed as zero, a finite interval, or a validated
confidence claim.

`interval_bounds(predictions, calibration, protocol=None)` uses the primary
target contract by default. Passing `protocol="official_uncapped"` selects only
the separately calibrated raw-target contract. Historical artifacts preserve
their original signed-offset bounds; they have no uncapped calibration contract.

Old saved runs remain historical evidence. New calibration requires a fresh
independent calibration step for the exact frozen model; it cannot retroactively
turn old interval metrics into conformal results.
