# Model slot

`model_manifest.json` describes the pending model. It contains no weights or predictions.

Run `GRACE_RUL_all_subsets.ipynb` with `QUICK_RUN = False` to produce
`artifacts/<run>/grace_rul_ALL.pt`. The website discovers full format-3 bundles
automatically. You may instead set `GRACE_MODEL_PATH` to the exact bundle path.

Quick runs are execution checks. Automatic website discovery excludes them.
An explicitly selected quick bundle remains labeled **quick validation model**.
