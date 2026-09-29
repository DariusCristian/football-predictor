"""The live model: the evaluated configuration, pinned.

Unweighted, rho=0, alpha=1e-5 is what the frozen test window scored
(docs/results.md, second test run). Everything that publishes or stores
predictions fits through here, so the configuration cannot drift.
"""

import warnings

import pandas as pd

from footy.model.poisson import DegenerateFitWarning, PoissonModel, fit

HALF_LIFE_DAYS = None
RHO = 0.0
ALPHA = 1e-5
MODEL_VERSION = "poisson-unweighted-rho0-alpha1e-5"


def fit_production_model(matches: pd.DataFrame) -> tuple[PoissonModel, list[str]]:
    """Fit on matches; return the model and any DegenerateFitWarning messages."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DegenerateFitWarning)
        model = fit(matches, half_life_days=HALF_LIFE_DAYS, rho=RHO, alpha=ALPHA)
    degenerate = []
    for warning in caught:
        if issubclass(warning.category, DegenerateFitWarning):
            degenerate.append(str(warning.message))
        else:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    return model, degenerate


def prediction_row(model: PoissonModel, fixture: dict) -> dict:
    """A store row for one fixture (match_date, season, home, away)."""
    p = model.predict(fixture["home"], fixture["away"])
    home_goals, away_goals = p["most_likely_score"]
    return {
        **fixture,
        "p_home": p["home_win"],
        "p_draw": p["draw"],
        "p_away": p["away_win"],
        "most_likely_home": home_goals,
        "most_likely_away": away_goals,
        "model_version": MODEL_VERSION,
    }
