"""Calibration analysis.

Log loss says how good predictions are overall. Calibration says whether
they mean what they claim: when the model says 30%, does it happen 30%
of the time? A model can have decent log loss and still be
systematically over- or under-confident.
"""

import numpy as np
import pandas as pd

OUTCOMES = ["home_win", "draw", "away_win"]


def to_flat(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, outcome): the probability and whether it happened."""
    rows = []
    for _, row in predictions.iterrows():
        for outcome in OUTCOMES:
            rows.append(
                {
                    "outcome": outcome,
                    "probability": row[outcome],
                    "happened": int(row["outcome"] == outcome),
                }
            )
    return pd.DataFrame(rows)


def reliability_table(
    flat: pd.DataFrame, n_bins: int = 10, outcome: str | None = None
) -> pd.DataFrame:
    """Bucket predictions by probability; compare predicted vs observed."""
    data = flat if outcome is None else flat[flat["outcome"] == outcome]
    edges = np.linspace(0, 1, n_bins + 1)
    data = data.assign(bin=pd.cut(data["probability"], edges, include_lowest=True))

    table = (
        data.groupby("bin", observed=True)
        .agg(
            n=("happened", "size"),
            mean_predicted=("probability", "mean"),
            observed=("happened", "mean"),
        )
        .reset_index()
    )
    table["gap"] = table["observed"] - table["mean_predicted"]
    return table


def expected_calibration_error(flat: pd.DataFrame, n_bins: int = 10) -> float:
    """Weighted mean absolute gap between predicted and observed. 0 is perfect."""
    table = reliability_table(flat, n_bins=n_bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


def outcome_frequencies(predictions: pd.DataFrame) -> pd.DataFrame:
    """Total predicted vs actual count per outcome — catches systematic bias."""
    rows = []
    for outcome in OUTCOMES:
        rows.append(
            {
                "outcome": outcome,
                "predicted_total": float(predictions[outcome].sum()),
                "actual_total": int((predictions["outcome"] == outcome).sum()),
            }
        )
    table = pd.DataFrame(rows)
    table["difference"] = table["actual_total"] - table["predicted_total"]
    return table