# Backtest Results

## 2026-10-03 — Baseline comparison

Walk-forward from 2023-08-01, retraining weekly, training only on
matches completed strictly before each prediction date.

| model          |    n | log loss | Brier |
|----------------|------|----------|-------|
| poisson        | 1184 |    1.027 | 0.594 |
| league average | 1190 |    1.076 | 0.651 |
| always home    | 1190 |    1.757 | 0.986 |

**Finding:** the Poisson model beats the league-average baseline by
~4.5% in log loss. Team identity carries real but modest signal;
football outcomes are dominated by variance.

**Caveat:** poisson n is 6 lower — matches involving teams absent from
the training data are skipped, so the comparison is not over an
identical set.


## 2026-10-03 — Time weighting experiment

Exponential decay weighting, `weight = 0.5 ** (days_ago / half_life)`,
tuned by walk-forward backtest from 2023-08-01.

| half-life    | log loss | Brier |
|--------------|----------|-------|
| 60 days      |   1.0487 | 0.602 |
| 120 days     |   1.0245 | 0.590 |
| 180 days     |   1.0192 | 0.587 |
| **270 days** |   1.0178 | 0.586 |
| 365 days     |   1.0183 | 0.587 |
| 540 days     |   1.0198 | 0.588 |
| 730 days     |   1.0212 | 0.589 |
| 1095 days    |   1.0228 | 0.590 |
| unweighted   |   1.0269 | 0.594 |

League-average baseline: 1.0756

**Finding:** clear U-shape with an optimum near 270 days. Weighting
improves log loss by ~0.9% over unweighted. Short half-lives (60 days)
are worse than no weighting at all — too few matches to estimate 40
team parameters. The basin is flat between 180 and 540 days, so the
result is not sensitive to the exact value.

**Interpretation:** a Premier League team's strength stays informative
for roughly one season.

**Caveat:** the half-life was selected on the same backtest used to
report the improvement, so 1.0178 is optimistic. A held-out period
would give an honest estimate.


## 2026-10-03 — Final evaluation (held-out test set)

Protocol: half-life tuned on validation window (2023-08-01 to
2024-08-01), then evaluated once on the held-out test window
(2024-08-01 onwards). Walk-forward throughout, training only on
matches completed strictly before each prediction.

**Validation selected: unweighted** (no time decay). The full-data
tuning run had selected 270 days, but that run tuned and reported on
the same matches.

| model          |   n | log loss | Brier |
|----------------|-----|----------|-------|
| poisson        | 806 |   1.0754 | 0.617 |
| league average | 810 |   1.0853 | 0.658 |
| always home    | 810 |   1.8003 | 1.012 |

**Headline: 0.9% improvement in log loss over the league-average
baseline.**

### The lesson

Tuning and reporting on the same backtest gave 5.4%. A proper
train/validate/test split gave 0.9%. The model was identical; only the
evaluation protocol changed. Roughly six-sevenths of the apparent gain
was selection bias.

### Secondary finding

Time weighting reversed direction between windows. On 2023-24 alone,
unweighted was best and log loss decreased monotonically with half-life.
On the full 2023-26 range, 270 days was best. Plausible cause: early in
the dataset there is little history to draw on, so downweighting old
matches discards signal; later there is enough history that discarding
stale matches helps. Time weighting appears to need depth of history to
pay off.

### Note on comparability

Validation log losses (~0.93) are much lower than test (~1.08). This
reflects 2023-24 being a more predictable season, not a better model.
Log loss is not comparable across different match sets.


## 2026-10-03 — Calibration analysis (test set, n=806)

Expected calibration error: 0.0337

### Predicted vs actual totals

| outcome  | predicted | actual | difference |
|----------|-----------|--------|------------|
| home win |     353.0 |    332 |      -21.0 |
| draw     |     184.4 |    213 |      +28.6 |
| away win |     268.5 |    261 |       -7.5 |

**Finding 1 — draws under-predicted by ~15%.** Consistent with the
Poisson independence assumption: the model treats home and away goals
as independent, but real matches drift toward level scorelines. This is
the defect Dixon-Coles corrects, now evidenced from this data rather
than assumed.

**Finding 2 — systematic overconfidence.** The reliability curve is
flatter than the diagonal: outcomes below ~30% predicted happen more
often than claimed, outcomes above ~30% happen less often. Worst
populated bin is 0.6–0.7, where the model claims 64% and observes 56%.
A separate defect from the draw deficit, likely from treating fitted
coefficients as certain. Fixes would be shrinkage or post-hoc
recalibration, not Dixon-Coles.

**Finding 3 — away wins are well calibrated** (gaps < 0.01 in populated
bins). The error concentrates in home wins and draws.

## 2026-10-03 — Dixon-Coles adjustment: no improvement

Grid search over rho on the validation window (2023-08-01 to 2024-08-01,
n=378), unweighted model, matching CHOSEN_HALF_LIFE.

| rho   | log loss |  Brier | draws predicted | draws actual |
|-------|----------|--------|-----------------|--------------|
| -0.20 |   0.9287 | 0.5474 |            99.0 |           82 |
| -0.10 |   0.9250 | 0.5453 |            91.3 |           82 |
| -0.05 |   0.9240 | 0.5447 |            87.4 |           82 |
|  0.00 |   0.9236 | 0.5445 |            83.6 |           82 |
|  0.05 |   0.9240 | 0.5446 |            79.7 |           82 |
|  0.10 |   0.9250 | 0.5450 |            75.8 |           82 |
|  0.20 |   0.9293 | 0.5469 |            68.1 |           82 |

**Result: rho = 0 is optimal. Dixon-Coles is not adopted.**

**Why the motivating evidence didn't hold.** The correction was
motivated by a ~15% draw deficit observed on the test set. On the
validation window the same model predicts draws almost exactly (83.6
vs 82). The deficit is window-specific, not a structural property of
the model.

**Protocol error, recorded.** The calibration analysis that motivated
this was run on the test set, which should be inspected once for the
final number only. Had rho been non-zero, the resulting "improvement"
would have been contaminated by held-out data. No harm occurred because
validation selected rho = 0, but the diagnosis should have been run on
the validation window. Calibration analysis has since been re-run there.

Implementation retained behind `rho=0.0` (a no-op by default) with tests,
so the option exists and the negative result is reproducible.