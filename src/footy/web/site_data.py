"""JSON payloads for the static site.

Pure functions from models and frames to dicts; scripts/build_site.py
does the fetching and writing. Probabilities are published to 4
decimals, rounded as a triple so each still sums to exactly 1.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

from footy.model.poisson import PoissonModel, dixon_coles_correction, score_matrix
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


def slugify(name: str) -> str:
    """'Leeds United' -> 'leeds-united'.

    Lowercase, spaces to hyphens. Any other run of characters outside
    a-z and 0-9 also becomes one hyphen, so a future name with an
    apostrophe or ampersand still gives a clean URL segment; no current
    team name has one.
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def match_slug(home: str, away: str) -> str:
    """'arsenal-leeds-united' for Arsenal at home to Leeds United."""
    return f"{slugify(home)}-{slugify(away)}"


class SlugCollisionError(ValueError):
    pass


def check_unique_slugs(slugs: list[str], what: str) -> None:
    """Raise if two entries share a slug; the site would show the wrong one."""
    repeated = sorted(slug for slug, n in Counter(slugs).items() if n > 1)
    if repeated:
        raise SlugCollisionError(f"duplicate {what} slugs: {', '.join(repeated)}")


def predictions_payload(stored: pd.DataFrame, generated_at: str) -> dict:
    """Committed predictions for matches not yet kicked off."""
    upcoming = stored[stored["scored_at"].isna() & (stored["match_date"] >= generated_at)]
    check_unique_slugs([match_slug(r.home, r.away) for r in upcoming.itertuples()], "prediction")
    return {
        "generated_at": generated_at,
        "predictions": [
            {
                "slug": match_slug(row.home, row.away),
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


# Scorelines published per pairing, likeliest first.
TOP_SCORES = 10


def top_scores(model: PoissonModel, home: str, away: str, k: int = TOP_SCORES) -> list[dict]:
    """The k likeliest scorelines, from the matrix model.predict() sums.

    Built the way PoissonModel.predict builds it (score_matrix, then the
    Dixon-Coles step, which also normalises), so the site displays these
    rather than recomputing Poisson maths in JavaScript. Ties break on
    fewer home goals, then fewer away goals, so output is deterministic.
    """
    home_rate = model.expected_goals(home, away, at_home=True)
    away_rate = model.expected_goals(away, home, at_home=False)
    matrix = dixon_coles_correction(
        score_matrix(home_rate, away_rate), home_rate, away_rate, model.rho
    )
    cells = sorted(np.ndindex(matrix.shape), key=lambda ij: (-matrix[ij], ij))[:k]
    return [
        {"home": int(i), "away": int(j), "p": round(float(matrix[i, j]), DECIMALS)}
        for i, j in cells
    ]


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
                    "slug": match_slug(home, away),
                    "home": home,
                    "away": away,
                    "probabilities": _probs(p["home_win"], p["draw"], p["away_win"]),
                    "expected_goals": {
                        "home": round(p["home_rate"], DECIMALS),
                        "away": round(p["away_rate"], DECIMALS),
                    },
                    "most_likely_score": list(p["most_likely_score"]),
                    "top_scores": top_scores(model, home, away),
                    "confidence": min(levels[home], levels[away],
                                      key=CONFIDENCE_ORDER.index),
                }
            )
    check_unique_slugs([p["slug"] for p in pairings], "pairing")
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
                          "opponent": home["away"], "venue": "home",
                          "goals_for": home["home_goals"],
                          "goals_against": home["away_goals"]}),
            pd.DataFrame({"date": away["date"], "season": away["season"],
                          "opponent": away["home"], "venue": "away",
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
    check_unique_slugs([slugify(t) for t in teams], "team")
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
            "slug": slugify(team),
            "attack": round(coefficients[team]["attack"], DECIMALS),
            "defence": round(coefficients[team]["defence"], DECIMALS),
            "attack_centred": round(coefficients[team]["attack"] - mean_attack, DECIMALS),
            "defence_centred": round(coefficients[team]["defence"] - mean_defence, DECIMALS),
            # exp(centred): goals scored / conceded relative to league average.
            "attack_multiplier": round(
                float(np.exp(coefficients[team]["attack"] - mean_attack)), DECIMALS),
            "defence_multiplier": round(
                float(np.exp(coefficients[team]["defence"] - mean_defence)), DECIMALS),
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
                "here, so 0 is league average. The _multiplier fields are "
                "exp(centred): 1.2 means 20% more goals than league average."
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


def standings_payload(
    matches: pd.DataFrame, teams: list[str], current_season: str, generated_at: str
) -> dict:
    """The current-season league table from completed matches.

    Sorted by points, then goal difference, then goals scored; teams level
    on all three are listed alphabetically and still get distinct
    positions (the league's own head-to-head tiebreaks aren't applied).
    matches_in_training and confidence come from every match in the
    data, as in teams_payload, so promoted teams can be marked.
    """
    season = matches[matches["season"] == current_season]
    names = sorted(set(teams) | set(season["home"]) | set(season["away"]))
    evidence = matches_in_training(matches, names)
    rows = []
    for team in names:
        played = _team_matches(season, team)
        won = int((played["goals_for"] > played["goals_against"]).sum())
        drawn = int((played["goals_for"] == played["goals_against"]).sum())
        lost = int((played["goals_for"] < played["goals_against"]).sum())
        goals_for = int(played["goals_for"].sum())
        goals_against = int(played["goals_against"].sum())
        rows.append({
            "team": team,
            "slug": slugify(team),
            "played": len(played),
            "won": won,
            "drawn": drawn,
            "lost": lost,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_difference": goals_for - goals_against,
            "points": 3 * won + drawn,
            "matches_in_training": evidence[team],
            "confidence": confidence_level(evidence[team]),
        })
    rows.sort(key=lambda r: (-r["points"], -r["goal_difference"], -r["goals_for"], r["team"]))
    return {
        "generated_at": generated_at,
        "season": current_season,
        # No matchday field in the match data: the most matches any team
        # has played. Equal to the matchday when the round is complete.
        "matchday": max((r["played"] for r in rows), default=0),
        "notes": {
            "order": ("Points, then goal difference, then goals scored; "
                      "teams level on all three are listed alphabetically."),
            "matchday": "The most league matches played by any team so far.",
            "confidence": "From matches_in_training, as in teams.json.",
        },
        "standings": [{"position": i, **row} for i, row in enumerate(rows, start=1)],
    }


# Results shown per team on a match page.
RECENT_RESULTS = 10


def _recent_results(before: pd.DataFrame, team: str, k: int = RECENT_RESULTS) -> list[dict]:
    """A team's last k results, most recent first, from its perspective."""
    played = _team_matches(before, team).tail(k).iloc[::-1]
    return [
        {
            "date": m.date.strftime("%Y-%m-%d"),
            "season": m.season,
            "opponent": m.opponent,
            "venue": m.venue,
            "goals_for": int(m.goals_for),
            "goals_against": int(m.goals_against),
            "result": _result_letter(m.goals_for, m.goals_against),
        }
        for m in played.itertuples()
    ]


def _head_to_head(before: pd.DataFrame, home: str, away: str) -> dict:
    """Every earlier meeting at either venue, most recent first.

    result and the totals are from home's perspective: home is the team
    at home in the upcoming fixture, not in each past meeting.
    """
    meetings = before[((before["home"] == home) & (before["away"] == away))
                      | ((before["home"] == away) & (before["away"] == home))]
    meetings = meetings.sort_values("date", ascending=False, kind="stable")
    rows = []
    for m in meetings.itertuples():
        home_for, home_against = ((m.home_goals, m.away_goals) if m.home == home
                                  else (m.away_goals, m.home_goals))
        rows.append({
            "date": m.date.strftime("%Y-%m-%d"),
            "season": m.season,
            "home": m.home,
            "away": m.away,
            "home_goals": int(m.home_goals),
            "away_goals": int(m.away_goals),
            "result": _result_letter(home_for, home_against),
        })
    letters = [r["result"] for r in rows]
    return {
        "perspective": home,
        "played": len(rows),
        "won": letters.count("W"),
        "drawn": letters.count("D"),
        "lost": letters.count("L"),
        "meetings": rows,
    }


def fixture_detail_payload(
    predictions: list[dict], matches: pd.DataFrame, generated_at: str
) -> dict:
    """Recent form and head-to-head for each upcoming fixture, by slug.

    predictions are predictions_payload entries. Only matches dated
    before the kickoff day count, so nothing from the fixture itself or
    after it can appear.
    """
    fixtures = {}
    for p in predictions:
        before = matches[matches["date"] < pd.Timestamp(p["kickoff"][:10])]
        fixtures[p["slug"]] = {
            "home": p["home"],
            "away": p["away"],
            "kickoff": p["kickoff"],
            "recent": {
                "home": _recent_results(before, p["home"]),
                "away": _recent_results(before, p["away"]),
            },
            "head_to_head": _head_to_head(before, p["home"], p["away"]),
        }
    return {
        "generated_at": generated_at,
        "data_starts": matches["date"].min().strftime("%Y-%m-%d") if len(matches) else None,
        "recent_results": RECENT_RESULTS,
        "fixtures": fixtures,
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


def relative_improvement(backtest: dict, model: str = "poisson",
                         baseline: str = "league_average") -> float:
    """Fractional log-loss reduction of model against baseline."""
    loss = {m["model"]: m["log_loss"] for m in backtest["models"]}
    return 1 - loss[model] / loss[baseline]


# Log loss of guessing 1/3 for every outcome: a fixed reference, not a model.
UNIFORM_LOG_LOSS = round(float(np.log(3)), DECIMALS)


def scoreboard_payload(stored: pd.DataFrame, generated_at: str) -> dict:
    return {
        "generated_at": generated_at,
        "backtest": {
            **BACKTEST_RESULT,
            "improvement_over_league_average": round(
                relative_improvement(BACKTEST_RESULT), DECIMALS),
            "uniform_log_loss": UNIFORM_LOG_LOSS,
        },
        "live": live_scoreboard(stored),
    }
