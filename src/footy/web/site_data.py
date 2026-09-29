"""JSON payloads for the static site.

Pure functions from models and frames to dicts; scripts/build_site.py
does the fetching and writing. Probabilities are published to 4
decimals, rounded as a triple so each still sums to exactly 1.
"""

import numpy as np
import pandas as pd

from footy.model.poisson import PoissonModel
from footy.model.scoring import log_loss
from footy.production import MODEL_VERSION

DECIMALS = 4

# Frozen test window, second run (docs/results.md). Published as-is,
# never recomputed: rerunning the test set is a deliberate act.
BACKTEST_RESULT = {
    "window_start": "2024-08-01",
    "window_end": "2026-06-30",
    "config": "unweighted, rho=0, alpha=1e-5",
    "models": [
        {"model": "poisson", "n": 758, "log_loss": 1.0260},
        {"model": "league_average", "n": 760, "log_loss": 1.0832},
        {"model": "always_home", "n": 760, "log_loss": 1.7901},
    ],
}


def round_probabilities(*probs: float, decimals: int = DECIMALS) -> list[float]:
    """Round so the results still sum to exactly 1 (largest remainder).

    Rounding each value independently can leave a triple off by up to
    1.5 units in the last place.
    """
    scale = 10**decimals
    total = sum(probs)
    exact = np.array(probs) / total * scale
    units = np.floor(exact).astype(int)
    shortfall = scale - units.sum()
    for i in np.argsort(-(exact - units), kind="stable")[:shortfall]:
        units[i] += 1
    return [int(u) / scale for u in units]


def _probs(p_home: float, p_draw: float, p_away: float) -> dict:
    home, draw, away = round_probabilities(p_home, p_draw, p_away)
    return {"home": home, "draw": draw, "away": away}


def predictions_payload(stored: pd.DataFrame, generated_at: str) -> dict:
    """Committed predictions for matches not yet kicked off."""
    upcoming = stored[stored["scored_at"].isna() & (stored["match_date"] >= generated_at)]
    return {
        "generated_at": generated_at,
        "predictions": [
            {
                "kickoff": row.match_date,
                "home": row.home,
                "away": row.away,
                "probabilities": _probs(row.p_home, row.p_draw, row.p_away),
                "most_likely_score": [int(row.most_likely_home), int(row.most_likely_away)],
                "model_version": row.model_version,
                "predicted_at": row.created_at,
            }
            for row in upcoming.itertuples()
        ],
    }


# Training matches per team, upper bounds (exclusive) of each level.
CONFIDENCE_LEVELS = [(10, "low"), (26, "medium")]
CONFIDENCE_ORDER = ["low", "medium", "high"]


def confidence_level(matches_in_training: int) -> str:
    """low under 10 matches, medium 10-25, high above 25."""
    for upper, level in CONFIDENCE_LEVELS:
        if matches_in_training < upper:
            return level
    return "high"


def matches_in_training(training: pd.DataFrame, teams: list[str]) -> dict[str, int]:
    counts = pd.concat([training["home"], training["away"]]).value_counts()
    return {team: int(counts.get(team, 0)) for team in teams}


def matrix_payload(
    model: PoissonModel, training: pd.DataFrame, teams: list[str], generated_at: str
) -> dict:
    """Every ordered (home, away) pairing of teams.

    confidence is the lower of the two teams' levels: a pairing is only
    as well-founded as its thinner side.
    """
    levels = {
        team: confidence_level(n)
        for team, n in matches_in_training(training, teams).items()
    }
    pairings = []
    for home in teams:
        for away in teams:
            if home == away:
                continue
            p = model.predict(home, away)
            pairings.append(
                {
                    "home": home,
                    "away": away,
                    "probabilities": _probs(p["home_win"], p["draw"], p["away_win"]),
                    "expected_goals": {
                        "home": round(p["home_rate"], DECIMALS),
                        "away": round(p["away_rate"], DECIMALS),
                    },
                    "most_likely_score": list(p["most_likely_score"]),
                    "confidence": min(levels[home], levels[away],
                                      key=CONFIDENCE_ORDER.index),
                }
            )
    return {
        "generated_at": generated_at,
        "model_version": MODEL_VERSION,
        "pairings": pairings,
    }


def team_coefficients(model: PoissonModel) -> dict[str, dict[str, float]]:
    """Attack and defence log-rate coefficients per team.

    Treatment coding: the alphabetically first team is the reference, at
    0 for both. Defence is the opponent term, so higher means the team
    concedes more.
    """
    params = model._result.params  # read-only; no public accessor on the model
    return {
        team: {
            "attack": float(params.get(f"C(team)[T.{team}]", 0.0)),
            "defence": float(params.get(f"C(opponent)[T.{team}]", 0.0)),
        }
        for team in model.teams
    }


def _result_letter(goals_for: int, goals_against: int) -> str:
    if goals_for > goals_against:
        return "W"
    if goals_for == goals_against:
        return "D"
    return "L"


def _team_matches(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    """One row per match played by team, from its perspective, oldest first."""
    home = matches[matches["home"] == team]
    away = matches[matches["away"] == team]
    return pd.concat(
        [
            pd.DataFrame({"date": home["date"], "season": home["season"],
                          "goals_for": home["home_goals"],
                          "goals_against": home["away_goals"]}),
            pd.DataFrame({"date": away["date"], "season": away["season"],
                          "goals_for": away["away_goals"],
                          "goals_against": away["home_goals"]}),
        ]
    ).sort_values("date", kind="stable")


def teams_payload(
    model: PoissonModel,
    training: pd.DataFrame,
    teams: list[str],
    current_season: str,
    generated_at: str,
) -> dict:
    coefficients = team_coefficients(model)
    # Display only: centre on the mean over the published teams. A shift,
    # so rankings and predictions are unchanged; 0 now means league average.
    mean_attack = np.mean([coefficients[t]["attack"] for t in teams])
    mean_defence = np.mean([coefficients[t]["defence"] for t in teams])
    entries = []
    for team in teams:
        played = _team_matches(training, team)
        season = played[played["season"] == current_season]
        entry = {
            "name": team,
            "attack": round(coefficients[team]["attack"], DECIMALS),
            "defence": round(coefficients[team]["defence"], DECIMALS),
            "attack_centred": round(coefficients[team]["attack"] - mean_attack, DECIMALS),
            "defence_centred": round(coefficients[team]["defence"] - mean_defence, DECIMALS),
            "matches_in_training": len(played),
            "confidence": confidence_level(len(played)),
            "season_matches": len(season),
        }
        if len(season):  # omitted before a team's first match of the season
            entry["goals_for_per_game"] = round(season["goals_for"].mean(), DECIMALS)
            entry["goals_against_per_game"] = round(season["goals_against"].mean(), DECIMALS)
        entry["last_5"] = [
            _result_letter(m.goals_for, m.goals_against)
            for m in played.tail(5).itertuples()
        ]
        entries.append(entry)
    return {
        "generated_at": generated_at,
        "model_version": MODEL_VERSION,
        "current_season": current_season,
        "notes": {
            "coefficients": (
                "Log goal-rate multipliers relative to the reference team "
                f"({min(model.teams)}, 0 for both). Higher attack scores more; "
                "higher defence concedes more. attack_centred and "
                "defence_centred subtract the mean across the teams listed "
                "here, so 0 is league average."
            ),
            "confidence": (
                "From matches_in_training: low under 10, medium 10-25, high above 25."
            ),
            "last_5": "Oldest first, most recent last.",
        },
        "teams": entries,
    }


def history_payload(matches: pd.DataFrame, generated_at: str) -> dict:
    return {
        "generated_at": generated_at,
        "matches": [
            {
                "season": m.season,
                "date": m.date.strftime("%Y-%m-%d"),
                "home": m.home,
                "away": m.away,
                "home_goals": int(m.home_goals),
                "away_goals": int(m.away_goals),
            }
            for m in matches.itertuples()
        ],
    }


def live_scoreboard(stored: pd.DataFrame) -> dict:
    """Running log loss over scored predictions. n=0 has no log_loss key."""
    scored = stored[stored["scored_at"].notna()]
    losses = [
        log_loss({"home_win": r.p_home, "draw": r.p_draw, "away_win": r.p_away}, r.outcome)
        for r in scored.itertuples()
    ]
    live = {"n": len(losses)}
    if losses:
        live["log_loss"] = round(float(np.mean(losses)), DECIMALS)
    return live


def scoreboard_payload(stored: pd.DataFrame, generated_at: str) -> dict:
    return {
        "generated_at": generated_at,
        "backtest": BACKTEST_RESULT,
        "live": live_scoreboard(stored),
    }
