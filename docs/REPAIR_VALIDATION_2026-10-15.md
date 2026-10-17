# Notebook repairs and validation - 15 October 2026

These changes repair the evaluation and comparison defects identified in the three-paper review, improve the training objective, and regenerate `GRACE_RUL_all_subsets.ipynb`. They do not establish improved trained-model accuracy. No current trained bundle is available.

## Implemented changes

- **Causal recurring blocks:** temporary sensor outages recur through a trajectory instead of expiring permanently near its beginning. Absolute onset, independent per-fault streams, prefix consistency, and overlapping-window consistency are preserved. `block_gap_cycles` defaults to 50. The protocol is now `persistent-causal-faults-v3`.
- **Actual damage exposure:** prediction records contain altered-entry counts and an active-fault flag. Reports separate active-fault and unaffected RMSE, MAE and NASA score and include actual affected-window/entry fractions. Capped and official uncapped targets remain separate, with cap labels derived from the configuration.
- **Comparable validation faults:** validation and correction use fixed explicit mixed faults. Changing the training clean fraction cannot make validation artificially easier. Selection averages clean RMSE and active-fault RMSE; all-window metrics remain visible.
- **Clean accuracy protection:** joint training adds clean-reference RUL supervision, default weight 0.25. The existing reference pass disables dropout and now supplies predictor gradients where appropriate; consistency targets remain detached. The default clean RMSE tolerance is 1%, measured against the original selected clean predictor across recovery and correction. Stage tolerances do not compound. Earlier admissible checkpoints and zero correction remain eligible.
- **Usable curriculum:** full fault difficulty is reached within ten epochs rather than after the full stage budget, so early stopping does not routinely precede exposure to mixed/full-strength training faults.
- **Fair no-repair training:** corrupted-input predictor training is controlled separately from repair. Disabling recovery retains a predictor-only joint stage. Clean-only CNN-GRU/Transformer references remain explicitly available.
- **Independent baseline searches:** paper adaptations start from shared untuned settings and receive their own six-trial validation searches. The CNN-LSTM/WGAN adaptation keeps a positive adversarial coefficient. Context/rate candidates are method-specific. Test engines and labels are excluded from those searches.
- **Controlled ablations:** local/global variants retain the gated predictor's remaining sensor/reliability interfaces. An inactive branch is neither executed nor supplied through its auxiliary prediction to the corrector. Equal-fusion and no-correction comparisons are included. The no-correction ablation reuses its exact frozen upstream weights; identical selected ablations also avoid redundant fits.
- **Paired active-fault comparisons:** bootstrap comparisons preserve whole engines, seeds and corruption replicates. Active comparisons require a fault to be present in both models' observed contexts. Final acceptance requires both natural-process and active-fault macro improvements, positive confidence bounds, clean preservation and a complete protocol. Quick runs cannot establish superiority.
- **Broader explanation references:** full-mode backgrounds contain 48 references selected across engines, operating regimes and four life-stage bins. Their provenance is saved in `background_audit.csv` and the bundle. Reference matching considers setting history; short references are resampled monotonically rather than repeated cyclically.
- **Actual global explanation workflow:** optional reports explain distinct held-out engines, repeat perturbation seeds, and aggregate engine-weighted sensor importance and variability. SHAP sparsity is no longer forced to ten blocks. SHAP local accuracy, LIME error/fidelity, and absolute-rank agreement are recorded without forcing agreement. These calculations rerun the actual frozen pipeline when weights and dependencies are available.

The existing supervised, observable-input OOF corrector is retained. It is a Trans-D3-inspired adaptation, not a TD3 implementation. Reconstruction weights remain explicit validation-selected coefficients rather than an unconstrained learned multiplier. No true test RUL is an inference feature or optimizer input. Grouped fitting/validation/calibration splits, fitting-only normalization, full epoch budgets and five primary training seeds remain intact.

## Checks completed here

The machine-readable record is [REPAIR_VALIDATION_2026-10-15.json](REPAIR_VALIDATION_2026-10-15.json).

| Check | Result | Scope |
| --- | --- | --- |
| Ten corruption/data tests | Passed | Actual NumPy/pandas/scikit-learn data code; tensor conversions replaced by array wrappers |
| Four additional regression tests | Passed | Actual pure source helpers: checkpoint constraints, configuration validation, paired bootstrap and active-cohort comparison |
| Dataset audit | Passed | All four official subsets; three corruption replicates; severities 0.1-0.5 |
| Background coverage | Passed | 48 distinct fitting engines per subset; all available regimes and four life-stage bins |
| Research orchestration | Passed with explicit mocks | Independent searches, context changes, branch configuration, weight reuse and quick-run claim guards; no neural training |
| Global explanation aggregation | Passed with explicit mocks | Engine weighting and perturbation-seed variability; no actual SHAP/LIME execution |
| Python/notebook source | Passed | Syntax, JSON, code-cell compilation and deterministic regeneration |
| Full test suite | Blocked during imports | `ModuleNotFoundError: No module named 'torch'` |

PyTorch is absent in the execution environment. A CPU-wheel installation was attempted but DNS lookup for `download.pytorch.org` failed. Neural forward/backward, CUDA execution, real SHAP/LIME, checkpoint-resume training and full benchmark accuracy were therefore **not verified**. New neural regression tests are included for running in the project's actual environment. Mocked workflow checks do not validate their neural behavior.

At nominal block severity 0.2, across the three configured replicate seeds:

| Subset | Previously unaffected inputs | After repair: unaffected / total | After repair: unaffected inputs |
| --- | ---: | ---: | ---: |
| FD001 | 53.0% | 37 / 300 | 12.33% |
| FD002 | 57.79% | 110 / 777 | 14.16% |
| FD003 | 57.0% | 16 / 300 | 5.33% |
| FD004 | 64.92% | 76 / 744 | 10.22% |

Remaining unaffected cases are retained as naturally unaffected observations, including early endpoints preceding onset, and are reported separately. They are not silently counted as successful active-fault recovery. These numbers measure introduced damage exposure, not a trained model's accuracy.

## Verify and train in the actual PyTorch environment

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/validate_notebook.py --device cpu --output-dir /tmp/grace-v3-notebook-check
.venv/bin/python scripts/check_cuda.py --training-check
```

Run fresh training after the neural checks pass:

```bash
.venv/bin/python scripts/train_research.py --device cuda --paper-study --explanations
```

Or use the notebook with `QUICK_RUN = False`, `RUN_PAPER_STUDY = True` and `RUN_EXPLANATIONS = True`. Full comparisons and explanations remain optional because of their substantial compute cost. Core training without them schedules up to 104 upstream fits: 24 validation trials and 80 primary/OOF fits.

Pipeline identity is version 4. Existing v2-protocol checkpoints must not resume this training; use a fresh directory. The format-5 tensor layout remains compatible for loading existing inference bundles under their recorded training semantics. Report/background switches can change without retraining upstream weights.

Evaluate all four subsets and five seeds. Inspect clean and corrupted metrics, paired intervals, component selections, reconstruction/distortion and detector metrics. The reported three-fold CV remains upstream CV, and empirical clean calibration intervals do not establish fault/OOD coverage. Improvement over the published papers remains unproven until matched trained results support it.
