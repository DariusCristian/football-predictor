import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from footy.config import DATA_DIR
from footy.data.training import load_all_matches
from footy.model.backtest import walk_forward
from footy.model.baselines import AlwaysHomeWin, LeagueAverage
from footy.model.evaluation import test_window
from footy.model.poisson import fit as fit_poisson
from footy.model.scoring import score_predictions

CHOSEN_HALF_LIFE = None  # selected on validation window; do not change
CHOSEN_ALPHA = 1e-5  # ridge penalty; see docs/results.md, "Ridge regularisation"

matches = load_all_matches()
start, end = test_window()

print(f"FINAL EVALUATION — test window from {start.date()}")
print("Run once. Report whatever it says.\n")

predictors = {
    "poisson": lambda t: fit_poisson(t, half_life_days=CHOSEN_HALF_LIFE,
                                     reference_date=t["date"].max(),
                                     alpha=CHOSEN_ALPHA),
    "league_average": lambda t: LeagueAverage().fit(t),
    "always_home": lambda t: AlwaysHomeWin().fit(t),
}

rows = []
for name, make in predictors.items():
    preds = walk_forward(matches, make, start_date=start, end_date=end)
    rows.append({"model": name, **score_predictions(preds)})

    if name == "poisson":
        preds.to_csv(DATA_DIR / "test_predictions.csv", index=False)

results = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
print(results.to_string(index=False))

poisson_loss = results.loc[results["model"] == "poisson", "log_loss"].iloc[0]
baseline_loss = results.loc[results["model"] == "league_average", "log_loss"].iloc[0]
print(f"\nimprovement over league average: "
      f"{(baseline_loss - poisson_loss) / baseline_loss:.1%}")