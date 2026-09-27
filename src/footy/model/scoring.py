"""Probabilistic scoring.

Accuracy is the wrong metric for probability forecasts: predicting 51%
and 99% look identical under accuracy but are very different claims.
Log loss and Brier score reward calibration, not just being on the
right side of 50%.
"""

import numpy as np
import pandas as pd

OUTCOMES = ["home_win", "draw", "away_win"]
EPS = 1e-15


def actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals == away_goals:
        return "draw"
    return "away_win"


def log_loss(probabilities: dict[str, float], outcome: str) -> float:
    """-log(p) of the outcome that actually happened. Lower is better."""
    p = max(probabilities[outcome], EPS)
    return -float(np.log(p))


def brier_score(probabilities: dict[str, float], outcome: str) -> float:
    """Squared error across all three outcomes. Lower is better."""
    return float(
        sum(
            (probabilities[o] - (1.0 if o == outcome else 0.0)) ** 2
            for o in OUTCOMES
        )
    )


def score_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    """predictions needs columns: home_win, draw, away_win, outcome."""
    losses, briers = [], []
    for _, row in predictions.iterrows():
        probs = {o: row[o] for o in OUTCOMES}
        losses.append(log_loss(probs, row["outcome"]))
        briers.append(brier_score(probs, row["outcome"]))
    return {
        "n": len(predictions),
        "log_loss": float(np.mean(losses)),
        "brier": float(np.mean(briers)),
    }