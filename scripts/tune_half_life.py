import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from footy.data.training import load_all_matches
from footy.model.backtest import walk_forward
from footy.model.baselines import LeagueAverage
from footy.model.poisson import fit as fit_poisson
from footy.model.scoring import score_predictions

matches = load_all_matches()
START = "2023-08-01"

HALF_LIVES = [60, 120, 180, 270, 365, 540, 730, 1095, None]

rows = []
for half_life in HALF_LIVES:
    def make(training, hl=half_life):
        reference = training["date"].max()
        return fit_poisson(training, half_life_days=hl, reference_date=reference)

    preds = walk_forward(matches, make, start_date=START)
    scores = score_predictions(preds)
    label = "none (equal)" if half_life is None else f"{half_life} days"
    rows.append({"half_life": label, **scores})
    print(f"  {label:<14} log loss {scores['log_loss']:.4f}  (n={scores['n']})")

print()
baseline = score_predictions(
    walk_forward(matches, lambda t: LeagueAverage().fit(t), start_date=START)
)
print(f"league average log loss: {baseline['log_loss']:.4f}")

print()
results = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
print(results.to_string(index=False))