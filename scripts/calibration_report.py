import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from footy.config import DATA_DIR
from footy.model.calibration import (
    expected_calibration_error,
    outcome_frequencies,
    reliability_table,
    to_flat,
)

preds = pd.read_csv(DATA_DIR / "test_predictions.csv")
flat = to_flat(preds)

print("=== predicted vs actual totals ===")
print(outcome_frequencies(preds).to_string(index=False))

print(f"\nexpected calibration error: {expected_calibration_error(flat):.4f}")

print("\n=== reliability, all outcomes ===")
print(reliability_table(flat).to_string(index=False))

for outcome in ["home_win", "draw", "away_win"]:
    print(f"\n=== reliability, {outcome} ===")
    print(reliability_table(flat, n_bins=5, outcome=outcome).to_string(index=False))