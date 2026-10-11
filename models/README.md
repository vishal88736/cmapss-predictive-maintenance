# Model slot

`model_manifest.json` describes the pending model. It contains no weights or predictions.

Run `GRACE_RUL_all_subsets.ipynb` with `QUICK_RUN = False` to produce
`artifacts/<run>/grace_rul_ALL.pt`. The website discovers full format-5 bundles
automatically. You may instead set `GRACE_MODEL_PATH` to the exact bundle path.

Quick runs are execution checks. Automatic website discovery excludes them.
An explicitly selected quick bundle remains labeled **quick validation model**.

The improved pipeline requires fresh training. Older format-3/4 checkpoints and
bundles are incompatible with its residual recovery and selection buffers.
