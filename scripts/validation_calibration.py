import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from footy.config import DATA_DIR
from footy.data.training import load_all_matches
from footy.model.backtest import walk_forward
from footy.model.calibration import (
    expected_calibration_error,
    outcome_frequencies,
    reliability_table,
    to_flat,
)
from footy.model.evaluation import validation_window
from footy.model.poisson import fit as fit_poisson

HALF_LIFE = None  # matches CHOSEN_HALF_LIFE in scripts/final_evaluation.py

matches = load_all_matches()
start, end = validation_window()

print(f"Calibration on VALIDATION window {start.date()} to {end.date()}\n")

preds = walk_forward(
    matches,
    lambda t: fit_poisson(t, half_life_days=HALF_LIFE,
                          reference_date=t["date"].max()),
    start_date=start,
    end_date=end,
)
preds.to_csv(DATA_DIR / "validation_predictions.csv", index=False)

flat = to_flat(preds)

print("=== predicted vs actual totals ===")
print(outcome_frequencies(preds).to_string(index=False))

print(f"\nexpected calibration error: {expected_calibration_error(flat):.4f}")

print("\n=== reliability, all outcomes ===")
print(reliability_table(flat, n_bins=8).to_string(index=False))