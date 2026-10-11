# Implementation validation

Validated locally on 8 October 2026 using the project's `.venv`, CPU execution,
and the original C-MAPSS files under `data/`.

## Completed checks

- All ten notebook code cells executed successfully in quick mode across
  FD001–FD004 after the split/GPU update, including
  staged training with actual WGAN-GP updates, three correction folds, empirical
  calibration, corruption evaluations, export/reload, plots, Kernel SHAP, and LIME.
- Thirteen regression checks passed with `python -m unittest discover -s tests -v`.
  These cover all four dataset counts and normalization regimes, engine separation,
  finite faults versus missingness, persistent corruption across overlapping windows,
  exclusion of future input, padding behavior, complete-outage recovery, second
  derivatives for gradient penalty, and ordered empirical intervals.
  Added checks cover the 80% training pool / 20% validation split, independent
  calibration reservation, fold report export, and failure of an explicit CUDA
  request before any run directory is created when no GPU is accessible.
- API checks exercise multipart uploads, pending-model responses, explicit quick
  selection, real predictions, recovered readings, dropout spread, explanation
  additivity, and rejection of legacy model artifacts.
- `npm --prefix website run build` passed. Chart dependencies are split into a
  separate cached bundle. Backend/package compilation and `pip check` passed.
- Default model discovery reports `awaiting_model`; quick artifacts remain excluded.

Saved run outputs and logs were removed during cleanup. Future notebook validation
creates fresh records under `artifacts/validation/` through
`scripts/validate_notebook.py`.

The checked-in notebook now defaults to `QUICK_RUN = False` and
`CONFIG['device'] = 'cuda'`. The validation runner changes these settings only
in its execution namespace to perform short CPU checks; it can use `--device cuda`
on an accessible GPU. Updated all-subset validation used a temporary output
directory, and no saved run results remain under the project's `artifacts/`.

The split counts for seed 42 are:

| Subset | Training pool | Optimizer fitting | Calibration | Validation |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 80 | 72 | 8 | 20 |
| FD002 | 208 | 187 | 21 | 52 |
| FD003 | 80 | 72 | 8 | 20 |
| FD004 | 199 | 179 | 20 | 50 |

Training pool = fitting + calibration. Whole-engine rounding applies.

## Scope of validation

The workspace prohibits local network sockets. A Jupyter kernel could not be
launched here, so notebook cells were executed sequentially in one Python namespace.
The installed Jupyter kernel points at the project's `.venv`. API routing and
multipart handling were exercised through HTTPX's in-process ASGI transport, with
thread dispatch made synchronous only in the test harness. Production endpoints
retain FastAPI's normal worker dispatch. A live server and browser interaction
were not verified in this restricted workspace.

An NVIDIA RTX 4050 and its loaded driver are visible in hardware information,
but PyTorch reports zero accessible CUDA GPUs and `nvidia-smi` fails here.
CUDA training was configured and its unavailable-device behavior was tested;
actual GPU execution was not verified. Check GPU access from the normal terminal
before running the notebook's full CUDA configuration.

Only small quick runs were trained. These establish executable integration and
produce explicitly labeled validation artifacts. Full five-seed training, matched
paper baselines, retrained ablations, real sensor-fault accuracy, and superiority
over the papers remain research experiments. Full-size model dimensions are
implemented but have not been benchmarked. Empirical interval coverage under
corruption or new operating conditions remains an evaluation result, not a guarantee.
