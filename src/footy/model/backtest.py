"""Walk-forward backtesting.

For each matchday, train only on matches completed strictly before it,
then predict that matchday. This mirrors how the system runs in
production and is the only honest way to estimate future performance.
"""

import pandas as pd

from footy.model.scoring import actual_outcome, score_predictions


def walk_forward(
    matches: pd.DataFrame,
    make_predictor,
    start_date: str,
    retrain_every_days: int = 7,
) -> pd.DataFrame:
    """Predict every match on or after start_date, training as we go.

    make_predictor(training_frame) must return an object with
    .predict(home, away) -> {"home_win", "draw", "away_win"}.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    start = pd.Timestamp(start_date)
    to_predict = matches[matches["date"] >= start]

    rows = []
    model = None
    last_trained = None

    for _, match in to_predict.iterrows():
        cutoff = match["date"]

        needs_retrain = (
            model is None
            or (cutoff - last_trained).days >= retrain_every_days
        )
        if needs_retrain:
            training = matches[matches["date"] < cutoff]
            model = make_predictor(training)
            last_trained = cutoff

        try:
            probs = model.predict(match["home"], match["away"])
        except ValueError:
            continue  # team unseen in training data

        rows.append(
            {
                "date": cutoff,
                "season": match["season"],
                "home": match["home"],
                "away": match["away"],
                "home_win": probs["home_win"],
                "draw": probs["draw"],
                "away_win": probs["away_win"],
                "outcome": actual_outcome(match["home_goals"], match["away_goals"]),
            }
        )

    return pd.DataFrame(rows)


def compare(matches: pd.DataFrame, predictors: dict, start_date: str) -> pd.DataFrame:
    results = []
    for name, make in predictors.items():
        preds = walk_forward(matches, make, start_date)
        scores = score_predictions(preds)
        results.append({"model": name, **scores})
    return pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)