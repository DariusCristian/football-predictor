import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from footy.data.training import load_all_matches
from footy.model.backtest import walk_forward
from footy.model.evaluation import validation_window
from footy.model.poisson import fit as fit_poisson
from footy.model.scoring import score_predictions

matches = load_all_matches()
start, end = validation_window()

HALF_LIVES = [120, 180, 270, 365, 540, 730, None]

print(f"Tuning on validation window {start.date()} to {end.date()}\n")

rows = []
for half_life in HALF_LIVES:
    def make(training, hl=half_life):
        return fit_poisson(training, half_life_days=hl,
                           reference_date=training["date"].max())

    preds = walk_forward(matches, make, start_date=start, end_date=end)
    scores = score_predictions(preds)
    label = "unweighted" if half_life is None else f"{half_life} days"
    rows.append({"half_life": label, "value": half_life, **scores})
    print(f"  {label:<12} log loss {scores['log_loss']:.4f}  (n={scores['n']})")

best = min(rows, key=lambda r: r["log_loss"])
print(f"\nbest on validation: {best['half_life']}")