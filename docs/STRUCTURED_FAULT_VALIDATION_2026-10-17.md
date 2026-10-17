# Structured fault repair validation, 17 October 2026

The notebook and shared dataset now use `structured-sensor-faults-v4`. Faults select
related measurement domains; noise has temporal/common-source structure; dropout
occurs in bursts; calibration drift is additive; stuck readings remain fixed in raw
units across operating regimes. Mixed scenarios share one source/domain/onset.
Fitting-only profiles determine raw-unit amplitude scales. Operating transitions
affect noise amplitude and burst-entry risk using observed settings only.

See [mechanisms and assumptions](STRUCTURED_SENSOR_FAULTS.md) and
[machine-readable evidence, data/source hashes and example parameters](STRUCTURED_FAULT_VALIDATION_2026-10-17.json).

## Checks completed

| Evidence | Result | Boundary |
| --- | --- | --- |
| Pure NumPy mechanism regressions | 11 passed | Actual simulator, no tensor wrappers |
| Dataset/scaler regressions | 11 passed | Actual NumPy/pandas/scikit-learn code; PyTorch array conversions wrapped with NumPy |
| Future-prefix comparisons | 3,400 passed | Fixed fitting profile and seed; single/mixed/random/clean families |
| Real C-MAPSS scenario endpoints | 16,968 passed | Four subsets, eight scenarios, severity .2, three replicates; no predictions |
| Config/checkpoint guards | Passed | Invalid new parameters rejected; v3 protocol and changed simulation parameters cannot resume v4 |
| Source/notebook compilation | 25 Python files, 10 notebook code cells | Syntax only; notebook figures and training not executed |
| Notebook regeneration | Deterministic, 20 cells | Empty execution outputs preserved |

The audit checked unchanged operating settings/speed commands, accurate changed-entry
labels, intact entries, finite bias/drift/noise/stuck values, original official RUL
targets, and fitting-only profile membership. Endpoint counts below include three
replicates and refer to **actual damage within the final observation window**:

| Subset | Total cases per scenario | Active block | Active noise | Active stuck | Active mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| FD001 | 300 | 268 | 268 | 214 | 270 |
| FD002 | 777 | 684 | 684 | 679 | 679 |
| FD003 | 300 | 285 | 285 | 236 | 283 |
| FD004 | 744 | 678 | 678 | 675 | 675 |

Late onsets and stuck sensors matching constant reference measurements legitimately
leave some endpoints unaffected. These cases stay explicit; an entire named fault
scenario is not treated as an entirely damaged cohort.

On the same 1,500-row zero-reference diagnostic with seed 17, onset zero and severity
.3, added-noise lag-one correlation changed from −0.00731 to 0.91264. Additive drift
changed 1,499 sensor entries, where the old multiplicative simulator changed none.
Intermittent loss stayed in one selected domain/channel instead of touching all
21 channels independently. These numbers describe corruption behavior, not accuracy.

## Remaining validation

Direct full-suite execution could not import three test modules because PyTorch is
missing (`ModuleNotFoundError: No module named 'torch'`). No neural training, actual
SHAP/LIME calculation, model accuracy, or improvement over the papers is claimed.
An isolated CPU PyTorch installation was attempted and failed to resolve
`download.pytorch.org`. `nvidia-smi` also could not communicate with the NVIDIA driver.
Matplotlib was also unavailable, so the new notebook preview was compiled but not
rendered. Fault groupings, probabilities, scales and temporal parameters remain
explicit simulation assumptions; recorded sensor-failure data was not supplied.

Use a fresh training directory and run matched v4 comparisons before making accuracy
claims. The existing active/unaffected metrics and clean-performance guard remain
in effect. Historical v2/v3 audit files describe their earlier protocols.
