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