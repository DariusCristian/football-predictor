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


## 2026-10-03 — Final evaluation (held-out test set) — SUPERSEDED

> **Superseded** by "Final evaluation, frozen test window" below. This
> run used an open-ended test window, so its 810 matches included 50
> live 2026-27 matches from the football-data.org API (48 scored for
> poisson). The set grew on every run and mixed two data sources. Kept
> as a record; do not cite these numbers.

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


## 2026-09-27 — Final evaluation, frozen test window

Test window is now frozen: 2024-08-01 to 2026-06-30, i.e. the 2024-25
and 2025-26 seasons only, all from OpenFootball history. The set no
longer changes between runs. Same model as before (unweighted, rho=0),
walk-forward, training only on matches completed strictly before each
prediction. Run once.

| model          |   n | log loss | Brier |
|----------------|-----|----------|-------|
| poisson        | 758 |   1.0494 | 0.614 |
| league average | 760 |   1.0832 | 0.656 |
| always home    | 760 |   1.7901 | 1.006 |

**Headline: 3.1% improvement in log loss over the league-average
baseline.**

The earlier 0.9% included 50 live 2026-27 matches (48 scored for
poisson). Removing them moved the result substantially: on those 48
matches the model averaged roughly 1.49 log loss against about 1.12 for
the baseline.

**Cause: complete separation in the unregularised GLM.** It is not a
data problem; the API rows match OpenFootball exactly for all 380
2025-26 fixtures, orientation included. A promoted team with few
matches and all-zero goals scored (or conceded) drives its attack (or
defence) coefficient towards −∞, while statsmodels still reports
`converged=True`. The model then assigns probabilities near 1e-10 to
events that can happen. Coventry City, predicted on 2026-09-19 from 4
matches with goals scored [0, 0, 0, 0]: attack coefficient −21.6,
P(Coventry win) ≈ 1e-10. Coventry won; log loss 22.9. That one match
adds ~0.48 to the 48-match average.

**The same failure affects this frozen run.** Burnley v Sunderland,
2025-08-23: Sunderland had conceded 0 in their only training match, so
P(Burnley win) ≈ 8e-11. Burnley won; log loss 23.2. It contributes
~0.031 to the 758-match average of 1.0494, which lowers the headline
improvement by roughly 2.8 points. The 3.1% is a correct run of the
locked model, but it is dominated by one degenerate prediction.

poisson n is 2 lower: the opening matches of Ipswich (2024-25) and
Sunderland (2025-26) are skipped because neither team is in the training
data. On the common 758 matches the improvement is slightly larger.


## 2026-10-03 — Calibration analysis (test set, n=806)

> `data/test_predictions.csv` was overwritten by the frozen-window run
> (n=758), so the n=806 numbers in this section can no longer be
> regenerated from it.

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


## 2026-10-03 — Calibration on validation: earlier findings do not replicate

Same model (unweighted, rho=0) on the validation window (n=378).

| outcome  | predicted | actual | difference |
|----------|-----------|--------|------------|
| home win |     169.8 |    174 |       +4.2 |
| draw     |      83.6 |     82 |       -1.6 |
| away win |     124.7 |    122 |       -2.7 |

ECE: 0.0224 (test set: 0.0337)

**Neither test-set finding replicates.**

The ~15% draw deficit is absent: 83.6 predicted vs 82 actual.

The overconfidence pattern is absent. On the test set, reliability gaps
were positive below 0.3 and negative above, monotonically. Here they
alternate sign, and the two highest bins are positive — the opposite
direction.

**Interpretation.** Both test-set findings were window-specific noise.
With ~100 predictions per bin, the standard error on an observed
frequency near 0.5 is roughly 0.05, larger than most of the gaps that
were interpreted as systematic. A single window is not enough to
establish a model defect; a claimed defect should either replicate
across windows or come with interval estimates.

**Consequence.** No shrinkage or recalibration step is adopted. The
model appears adequately calibrated given the sample sizes available.

## 2026-09-29 — Ridge regularisation: alpha = 1e-5

Addresses the complete-separation failure described under "Final
evaluation, frozen test window". `fit()` now takes `alpha`, a ridge (L2)
penalty on the team and opponent coefficients, via statsmodels'
`fit_regularized(L1_wt=0)`. The intercept and `home` are not penalised
(per-coefficient penalty weights of 0); penalising them pulls the
league-wide goal rate and home advantage towards zero.

### Validation grid

Validation window (2023-08-01 to 2024-08-01), unweighted, rho = 0.
"Near-zero" counts matches where any outcome probability is below 1e-4;
"degenerate" counts fits with a coefficient beyond ±10.

| alpha | n   | log loss | vs alpha=0 | Brier   | max \|coef\| | near-zero | degenerate |
|-------|-----|----------|------------|---------|--------------|-----------|------------|
| 0     | 378 | 0.92364  |            | 0.54447 | 20.98        | 1         | 1          |
| 1e-7  | 378 | 0.92366  | +0.00001   | 0.54448 | 7.43         | 0         | 0          |
| 1e-6  | 378 | 0.92368  | +0.00004   | 0.54449 | 5.43         | 0         | 0          |
| 1e-5  | 378 | 0.92384  | +0.00020   | 0.54458 | 3.54         | 0         | 0          |
| 3e-5  | 378 | 0.92411  | +0.00047   | 0.54474 | 2.71         | 0         | 0          |
| 1e-4  | 378 | 0.92496  | +0.00132   | 0.54525 | 1.87         | 0         | 0          |
| 1e-3  | 378 | 0.93260  | +0.00896   | 0.55037 | 1.05         | 0         | 0          |
| 1e-2  | 378 | 0.95722  | +0.03358   | 0.56732 | 0.56         | 0         | 0          |

Validation alone selects alpha = 0. It contains a single separation
event (Sheffield United's first match of 2023-24, P(away win) ≈ 5e-10),
and that outcome did not occur, so log loss never sees the failure.
Reproduce with `scripts/tune_alpha.py`.

### Known failure cases

Each refit on every match before the fixture date. At alpha = 0 these
reproduce the originally recorded values. The Sunderland case lies in
the frozen test window; only its training set was reconstructed, and
the test window was not re-scored.

Coventry attack: Nottingham v Coventry, 2026-09-19 (4 matches, scored
[0, 0, 0, 0]).

| alpha | Coventry attack | max \|coef\| | P(Coventry win) |
|-------|-----------------|--------------|-----------------|
| 0     | −21.56          | 21.56        | 1.1e-10         |
| 1e-7  | −7.54           | 7.54         | 1.4e-4          |
| 1e-6  | −5.52           | 5.52         | 1.0e-3          |
| 1e-5  | −3.63           | 3.63         | 0.0070          |

Hull defence: Hull v Aston Villa, 2026-09-05 (2 matches, conceded
[0, 0]). The largest coefficient here is Coventry's attack.

| alpha | Hull defence | max \|coef\| | P(Villa win) |
|-------|--------------|--------------|--------------|
| 0     | −20.31       | 20.57        | 3.1e-10      |
| 1e-7  | −5.80        | 6.48         | 6.3e-4       |
| 1e-6  | −3.96        | 4.42         | 3.9e-3       |
| 1e-5  | −2.36        | 2.61         | 0.020        |

Sunderland defence: Burnley v Sunderland, 2025-08-23 (1 match,
conceded [0]).

| alpha | Sunderland defence | max \|coef\| | P(Burnley win) |
|-------|--------------------|--------------|----------------|
| 0     | −20.16             | 20.16        | 8.6e-11        |
| 1e-7  | −6.15              | 6.15         | 1.0e-4         |
| 1e-6  | −4.21              | 4.21         | 7.3e-4         |
| 1e-5  | −2.45              | 2.45         | 0.0044         |

### Decision: alpha = 1e-5

Validation chose alpha = 0, but the unregularised model assigns ~1e-10
to outcomes that occur. That is a correctness failure independent of
any score. 1e-7 only caps the runaway: coefficients of −5.8 to −7.5,
still under 1% of a normal goal rate. 1e-5 brings Hull (−2.36) and
Sunderland (−2.45) into the plausible range and Coventry to −3.63, at a
validation cost of 0.0002 log loss, inside the 0.0005 threshold set in
advance. Stronger values were not chosen: they were justified only by
the three hand-picked failure cases, not by validation.

Set as `DEFAULT_ALPHA` in `poisson.py` and `CHOSEN_ALPHA` in
`scripts/final_evaluation.py`. The test window has not been re-run.

### Limitation

Ridge reduces the problem but does not eliminate it. Coventry still
gets P(win) = 0.7%, which costs ~5.0 log loss when it happens (versus
22.9 unregularised and ~1.0 for a typical match). The principled fix is
a hierarchical prior that shrinks each team towards the league average.
Ridge shrinks towards the reference team instead (Arsenal, the first
level alphabetically), an artifact of the treatment-coded categorical
encoding rather than a modelling choice.

### Penalty scaling

statsmodels minimises −loglike/nobs + alpha/2·|b|², equivalent to an
effective penalty of alpha × training rows. Training grows from 1,520
to 2,260 rows across the validation window, so the strength drifts
×1.487. Immaterial at 1e-5: the effective penalty goes from 0.0152 to
0.0226, an implied prior sd on team coefficients of 8.1 falling to 6.7,
far wider than real coefficients.


## 2026-10-XX — SECOND test run: regularised model

**This is the second run on the held-out test set.** Justification: the
unregularised model assigns probabilities near 1e-10 to outcomes that
occur, a correctness failure visible in validation predictions and live
matches, independent of any test-set result. The fix (alpha) was chosen
on validation. See the regularisation section above.

Frozen window 2024-08-01 to 2026-06-30. Unweighted, rho=0, alpha=1e-5.

| model          |   n | log loss | Brier |
|----------------|-----|----------|-------|
| poisson        | 758 |   1.0260 | 0.614 |
| league average | 760 |   1.0832 | 0.656 |
| always home    | 760 |   1.7901 | 1.006 |

**Headline: 5.3% improvement in log loss over the league-average
baseline.** Supersedes the 3.1% from the first frozen run.

**What changed, and what did not.** The model is no better at predicting
football. Log loss fell 0.023, and the single Burnley v Sunderland
collapse was contributing ~0.031 at alpha=0. Almost the entire gain is
one impossible prediction becoming merely a poor one. 5.3% is a truer
estimate of ordinary performance because it is not distorted by a
pathological fit, not because the model learned anything.

## 2026-09-29 — Promoted teams in production

The first live predictions (10 fixtures, 10–12 October 2026, committed
to the store before kickoff) show the promoted-team problem in the
production model: unweighted, rho=0, alpha=1e-5. No
DegenerateFitWarning fired. Every coefficient stays well within the
±10 limit, so these are not divergences. They are thin samples taken
at face value.

**Hull City: rated the best defence in the league on 5 matches.** The
defence coefficient is −0.099, better than Arsenal (0, the reference)
and Man City (−0.032). It comes from conceding 4 in 5 games. The
committed prediction for Hull v Everton gives Hull 54% to win at home
(draw 29%, Everton 17%).

**Coventry City: attack coefficient −1.93, from 1 goal in 5 games.**
That makes Coventry 5.4% to win at home to Newcastle (committed), and
1.1% to win at Man City in the matrix.

### What ridge did and did not do

Ridge prevented divergence: without it, the known failure cases
give ~1e-10 to outcomes that happen. It did not
solve the underlying problem. Five matches are still taken close to
face value.

Shrinkage goes toward the reference team, not the league average.
That is an artifact of treatment-coded categoricals, not a modelling
choice. With Arsenal as the reference, a promoted team with little
data is pulled toward Arsenal's defence and Arsenal's attack, both
elite, rather than toward a typical side. For Hull that shrinkage
makes the flattering defence number more flattering, not less. For
Coventry it only partly offsets the attack signal.

Centring the coefficients for display (subtracting the league mean)
does not change this. It is a constant shift, so rankings and
predictions are identical. Centred, Hull's defence is −0.438, still
first, ahead of Man City at −0.371. Centring fixes how the numbers
read (0 = league average), not what the model believes.

The principled fix is a hierarchical prior: team coefficients drawn
from a common distribution, so each is shrunk toward the league
average in proportion to how little data it has.

### What the site does about it

The site surfaces sample size rather than hiding it. teams.json
carries matches_in_training and a confidence level (low under 10,
medium 10–25, high above 25). Each matrix.json pairing carries the
lower of its two teams' levels. Hull and Coventry are "low", as are
all 74 pairings involving them.
