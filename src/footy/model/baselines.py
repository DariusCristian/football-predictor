"""Reference predictors the real model must beat.

Each takes the same training data the model gets, so comparisons are fair.
"""

import pandas as pd


class AlwaysHomeWin:
    """Confidently predicts a home win every time. Deliberately naive."""

    name = "always_home"

    def fit(self, matches: pd.DataFrame) -> "AlwaysHomeWin":
        return self

    def predict(self, home_team: str, away_team: str) -> dict:
        return {"home_win": 0.90, "draw": 0.05, "away_win": 0.05}


class LeagueAverage:
    """Predicts the historical base rates, ignoring who is playing."""

    name = "league_average"

    def __init__(self):
        self.rates = None

    def fit(self, matches: pd.DataFrame) -> "LeagueAverage":
        home = (matches["home_goals"] > matches["away_goals"]).mean()
        draw = (matches["home_goals"] == matches["away_goals"]).mean()
        self.rates = {
            "home_win": float(home),
            "draw": float(draw),
            "away_win": float(1 - home - draw),
        }
        return self

    def predict(self, home_team: str, away_team: str) -> dict:
        if self.rates is None:
            raise RuntimeError("fit() must be called before predict()")
        return dict(self.rates)