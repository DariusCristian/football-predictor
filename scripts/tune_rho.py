import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from footy.data.training import load_all_matches
from footy.model.backtest import walk_forward
from footy.model.calibration import outcome_frequencies
from footy.model.evaluation import validation_window
from footy.model.poisson import PoissonModel, fit as fit_poisson
from footy.model.scoring import score_predictions

HALF_LIFE = None  # matches CHOSEN_HALF_LIFE in scripts/final_evaluation.py
RHOS = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]

matches = load_all_matches()
start, end = validation_window()

# rho only affects predict(), so fit the GLM once per retrain date and
# reuse it across every rho.
_fits: dict[pd.Timestamp, PoissonModel] = {}


def make(training, rho):
    key = training["date"].max()
    if key not in _fits:
        _fits[key] = fit_poisson(training, half_life_days=HALF_LIFE,
                                 reference_date=key)
    base = _fits[key]
    return PoissonModel(base._result, base.teams, rho=rho)


print(f"Tuning rho on validation window {start.date()} to {end.date()}\n")

rows = []
for rho in RHOS:
    preds = walk_forward(matches, lambda t, r=rho: make(t, r),
                         start_date=start, end_date=end)
    scores = score_predictions(preds)
    freq = outcome_frequencies(preds).set_index("outcome")
    rows.append({
        "rho": rho,
        **scores,
        "draws_predicted": freq.loc["draw", "predicted_total"],
        "draws_actual": freq.loc["draw", "actual_total"],
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# walk_forward skips predictions that raise ValueError, which includes a
# rho that makes a cell negative. Unequal n would mean that happened.
if table["n"].nunique() > 1:
    print("\nWARNING: n differs across rho — some predictions were skipped")

best = table.loc[table["log_loss"].idxmin()]
print(f"\nbest on validation: rho = {best['rho']}")
