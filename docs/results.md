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