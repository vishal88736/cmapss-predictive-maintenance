# GRACE-RUL v2: design proposal

Status: proposal. v1 (`v1.0-assurance-20261110`, bundle `577cf22a`) stays untouched until v2
passes every gate at the end of this document.

## Why redesign at all

These are measurements on the frozen v1 artifact, not assumptions:

| Finding | Evidence |
| --- | --- |
| Recovery carries the persistent-fault gain | stuck 20.07 -> 15.08 RMSE when recovery is enabled; mixed 14.44 -> 13.05 |
| Recovery is inert under broadband noise | noise@0.3: 13.11 with recovery vs 13.19 without |
| The bounded corrector buys nothing | removing it changes RMSE by at most 0.03 in every condition |
| Noise barely hurts the predictor | 12.01 clean -> 12.37 at 20% noise -> 13.68 at 50% |
| FD004 error is tail-dominated | the worst two engines carry ~14% of squared error |
| The model is small and fast | 348,656 parameters, 1.33 MB, 5.08 ms/window, 1,285 engines/s |

The v1 weakness is therefore not capacity. It is that recovery was trained for finite faults
and evaluated against broadband noise, which it does not target.

## Keep / drop / add

| Decision | Component | Reason |
| --- | --- | --- |
| Keep | Availability-masked input (21 sensors + 3 settings) | Required for missing readings |
| Keep | Independent finite-fault detector and validated thresholds | Gates repair; passes audits |
| Keep | Residual BiLSTM recovery with fitting-support bounds | Measured +3.6 to +5.0 on stuck |
| Keep | Local CNN-GRU (30 cycles) and global Transformer (70 cycles) | Core regression |
| Keep | Quality-conditioned fusion | No evidence against it |
| Keep | Engine-wise conformal calibration | Coverage holds; leave the contract alone |
| **Drop** | Bounded OOF corrector | 2,209 parameters, <=0.03 RMSE effect, never positive |
| **Add** | Broadband noise in the training corruption mix | Makes the repair objective cover noise |
| **Add** | Explicit denoising loss on noise-injected entries | Supervises recovery on the actual failure mode |
| **Add** | Tail-robust RUL objective with validation-selected weight | Targets the FD004 tail directly |
| Defer | KAN head | Only if a 2 h A/B clears 2% on validation |

## Training pipeline

1. Split: unchanged engine-disjoint 80/10/10, source-stratified, same `split_seed`.
2. Corruption mix per engine: the eight structured families at severities 0.1-0.3, plus
   broadband Gaussian at 5-50% of the within-regime scale, plus cell dropouts at 0-5%.
   `grace_rul/noise.py` already implements the broadband layer deterministically per engine.
3. Objective, in order of weight to be fixed before the run:
   - capped and raw RUL on corrupted input;
   - reconstruction loss on finite altered entries (original-linear units);
   - **denoising loss** between the repaired window and the clean reference on
     noise-injected entries, which is the new term;
   - intact identity loss, bounded by the fitting-support guard;
   - tail loss over the worst-engine error, weight selected on inner validation only;
   - paired-prefix consistency.
4. Selection: inner 36 selection engines only. No official test read before freeze.
5. Export: same bundle format, same strict loader, same 24-guard audit.

## Evaluation protocol (unchanged, so results stay comparable)

- Official C-MAPSS endpoints, 125-cycle cap and raw contract reported separately.
- Per-subset and pooled, engine-disjoint.
- Noise benchmark 0/5/10/20/30/50% plus dropouts.
- Recovery/detection table by fault family.
- Cost table.
- The 24-guard active-path audit.

## Success gates

v2 replaces v1 only if **all** hold:

1. 24/24 active-path guards pass, including `mixed_rul_FD003` and `stuck_rul_FD003`.
2. Validation clean RMSE is no worse than v1's validation value.
3. Noise RMSE improves by at least 0.2 cycles at both 20% and 50%.
4. The worst-five-engine share of FD004 squared error falls below v1's 17.6%.
5. Single-window latency stays under 7 ms.
6. Strict bundle load passes and calibration is refitted only on the 35 calibration engines.

If any gate fails, v1 remains the reported artifact and v2 is recorded as a rejected
candidate, exactly as the earlier repairs were.

## Budget

| Stage | Estimate |
| --- | --- |
| Implement denoising + tail losses | 1 h |
| Full training (one seed, GPU) | ~6.5 h |
| Guard audit and calibration | ~1 h |
| Evaluation and asset regeneration | ~1 h |

About 9 h end to end, slightly more than the 7-8 h available. The honest options are to
run the implementation plus a shortened fine-tune from `577cf22a` within budget, or to
schedule the from-scratch run when a longer window exists.

## Risks

- **Small headroom.** At 20% noise a perfect denoiser buys only 0.36 cycles, so gate 3 is
  the hardest to clear honestly.
- **Guard regression.** Short fine-tunes on this pipeline previously traded clean accuracy
  for guard compliance (11.94 -> 12.01).
- **One seed.** v2 would inherit the same single-seed limitation; it does not fix v1's
  main methodological weakness.
- **No independent holdout.** Still unavailable; every C-MAPSS engine has been used.

## Non-goals

- Chasing STAR's 10.61 on FD001 with this budget.
- Any evaluation that merges official training and test trajectories, or re-splits at the
  sample level. Those published numbers (TRANS-D3 FD001 1.66; Transformer-KAN-BiLSTM FD001
  3.68) are not comparable and are not targets.
