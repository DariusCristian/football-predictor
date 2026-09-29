import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from footy.data.training import load_all_matches
from footy.model.backtest import walk_forward
from footy.model.evaluation import validation_window
from footy.model.poisson import DegenerateFitWarning, fit as fit_poisson
from footy.model.scoring import OUTCOMES, score_predictions

HALF_LIFE = None  # matches CHOSEN_HALF_LIFE in scripts/final_evaluation.py
RHO = 0.0  # selected on validation; see scripts/tune_rho.py
ALPHAS = [0, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4, 1e-3, 1e-2]
NEAR_ZERO = 1e-4

matches = load_all_matches()
start, end = validation_window()

print(f"Tuning ridge alpha on validation window {start.date()} to {end.date()}\n")

# statsmodels minimises -loglike/nobs + alpha/2 * |b|^2, i.e.
# -loglike + (alpha * nobs)/2 * |b|^2: the effective ridge penalty is
# alpha * nobs, so it grows as the training set grows.
training_rows: list[int] = []

rows = []
for alpha in ALPHAS:
    max_coef = 0.0
    degenerate_fits = 0

    def make(training, a=alpha):
        global max_coef, degenerate_fits
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DegenerateFitWarning)
            model = fit_poisson(training, half_life_days=HALF_LIFE,
                                reference_date=training["date"].max(),
                                rho=RHO, alpha=a)
        if any(issubclass(w.category, DegenerateFitWarning) for w in caught):
            degenerate_fits += 1
        max_coef = max(max_coef, model._result.params.abs().max())
        training_rows.append(int(model._result.model.nobs))
        return model

    preds = walk_forward(matches, make, start_date=start, end_date=end)
    scores = score_predictions(preds)
    rows.append({
        "alpha": alpha,
        **scores,
        "max_abs_coef": max_coef,
        # matches where any of the three outcome probabilities is < NEAR_ZERO
        "near_zero_preds": int((preds[OUTCOMES].min(axis=1) < NEAR_ZERO).sum()),
        "degenerate_fits": degenerate_fits,
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False, float_format=lambda x: f"{x:.5f}",
                      formatters={"alpha": "{:g}".format}))

if table["n"].nunique() > 1:
    print("\nWARNING: n differs across alpha — some predictions were skipped")

best = table.loc[table["log_loss"].idxmin()]
print(f"\nbest on validation: alpha = {best['alpha']}")

first, last = training_rows[0], training_rows[-1]
finite = table[(table["alpha"] > 0) & (table["max_abs_coef"] <= 10)]
if not finite.empty:
    smallest = finite.iloc[0]
    a = smallest["alpha"]
    print(f"\nsmallest alpha with every coefficient within +-10: {a}")
    print(f"  log loss cost vs alpha=0: "
          f"{smallest['log_loss'] - table.loc[0, 'log_loss']:+.5f}")
    for label, n in (("first retrain", first), ("last retrain", last)):
        penalty = a * n
        print(f"  {label}: {n} rows, effective penalty alpha*nobs = "
              f"{penalty:.4g} (implied prior sd {penalty ** -0.5:.3g})")
    print(f"  drift: x{last / first:.3f}")
