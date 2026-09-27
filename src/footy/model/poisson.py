"""Poisson regression model for match outcomes.

Each match contributes two rows: one per attacking side. The model learns
an attack strength and a defence strength per team, plus a global home
advantage, and predicts a goal rate for any team pairing.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson


def to_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (attacking team, defending team, venue)."""
    home_rows = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["home"],
            "opponent": matches["away"],
            "home": 1,
            "goals": matches["home_goals"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["away"],
            "opponent": matches["home"],
            "home": 0,
            "goals": matches["away_goals"],
        }
    )
    return pd.concat([home_rows, away_rows], ignore_index=True)


def decay_weights(
    dates: pd.Series,
    half_life_days: float,
    reference_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Exponential decay: a match half_life_days old gets weight 0.5."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    reference = reference_date if reference_date is not None else dates.max()
    days_ago = (reference - dates).dt.days.clip(lower=0)
    return 0.5 ** (days_ago / half_life_days)

class PoissonModel:
    def __init__(self, result, teams: set[str], rho: float = 0.0):
        self._result = result
        self.teams = teams
        self.rho = rho  # Dixon-Coles dependence; 0.0 means plain Poisson

    def expected_goals(self, team: str, opponent: str, at_home: bool) -> float:
        for name in (team, opponent):
            if name not in self.teams:
                raise ValueError(f"Team not in training data: {name!r}")
        row = pd.DataFrame(
            [{"team": team, "opponent": opponent, "home": int(at_home)}]
        )
        return float(self._result.predict(row).iloc[0])

    def predict(self, home_team: str, away_team: str) -> dict:
        home_rate = self.expected_goals(home_team, away_team, at_home=True)
        away_rate = self.expected_goals(away_team, home_team, at_home=False)
        matrix = score_matrix(home_rate, away_rate)
        matrix = dixon_coles_correction(matrix, home_rate, away_rate, self.rho)
        probs = outcome_probabilities(matrix)
        i, j = np.unravel_index(matrix.argmax(), matrix.shape)
        return {
            "home_rate": home_rate,
            "away_rate": away_rate,
            "most_likely_score": (int(i), int(j)),
            **probs,
        }
    
DEFAULT_HALF_LIFE_DAYS = 270  # tuned by backtest; see docs/results.md


def fit(
    matches: pd.DataFrame,
    half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS,
    reference_date: pd.Timestamp | None = None,
    rho: float = 0.0,
) -> PoissonModel:
    """Fit the model. half_life_days=None means all matches weighted equally.

    rho is the Dixon-Coles dependence parameter applied at predict time;
    it does not affect the GLM fit.
    """
    long = to_long_format(matches)

    if half_life_days is None:
        weights = None
    else:
        weights = decay_weights(long["date"], half_life_days, reference_date)

    result = smf.glm(
        formula="goals ~ home + C(team) + C(opponent)",
        data=long,
        family=sm.families.Poisson(),
        freq_weights=weights,
    ).fit()
    teams = set(long["team"])
    return PoissonModel(result, teams, rho=rho)

MAX_GOALS = 10


def score_matrix(home_rate: float, away_rate: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P[i, j] = probability of home scoring i and away scoring j."""
    home_probs = poisson.pmf(np.arange(max_goals + 1), home_rate)
    away_probs = poisson.pmf(np.arange(max_goals + 1), away_rate)
    return np.outer(home_probs, away_probs)


def dixon_coles_correction(
    matrix: np.ndarray, home_rate: float, away_rate: float, rho: float
) -> np.ndarray:
    """Dixon-Coles (1997) low-score adjustment, renormalised to sum to 1.

    Rescales cells (0,0), (0,1), (1,0), (1,1) by tau; all others are
    untouched. Negative rho raises 0-0 and 1-1 (more draws); positive rho
    lowers them.

    Raises ValueError if any tau is negative, which happens when
    rho > 1 / (home_rate * away_rate) or rho < -1 / max(home_rate, away_rate).
    """
    tau = {
        (0, 0): 1 - home_rate * away_rate * rho,
        (0, 1): 1 + home_rate * rho,
        (1, 0): 1 + away_rate * rho,
        (1, 1): 1 - rho,
    }
    if min(tau.values()) < 0:
        raise ValueError(
            f"rho={rho} gives a negative probability at rates "
            f"({home_rate:.3f}, {away_rate:.3f})"
        )
    corrected = matrix.copy()
    for (i, j), factor in tau.items():
        corrected[i, j] *= factor
    return corrected / corrected.sum()


def outcome_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Sum the matrix into home win / draw / away win."""
    total = matrix.sum()
    return {
        "home_win": float(np.tril(matrix, -1).sum() / total),
        "draw": float(np.trace(matrix) / total),
        "away_win": float(np.triu(matrix, 1).sum() / total),
    }

