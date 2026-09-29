import math

import numpy as np
import pandas as pd
import pytest

from footy.web.site_data import (
    confidence_level,
    live_scoreboard,
    matrix_payload,
    round_probabilities,
    scoreboard_payload,
    teams_payload,
)
from footy.production import fit_production_model


@pytest.mark.parametrize(
    "probs",
    [(1 / 3, 1 / 3, 1 / 3), (0.12345, 0.23455, 0.642), (0.99995, 0.00004, 0.00001),
     (0.45678, 0.27161, 0.27161)],
)
def test_rounded_triple_sums_to_one(probs):
    rounded = round_probabilities(*probs)
    assert sum(rounded) == pytest.approx(1.0, abs=1e-12)
    for raw, r in zip(probs, rounded):
        assert abs(raw - r) < 1e-4
        assert r == round(r, 4)


def test_independent_rounding_would_not_sum_to_one():
    """The case that motivates round_probabilities."""
    probs = (1 / 3, 1 / 3, 1 / 3)
    assert abs(sum(round(p, 4) for p in probs) - 1) > 1e-6


def _stored(rows):
    columns = ["p_home", "p_draw", "p_away", "outcome", "scored_at"]
    return pd.DataFrame(rows, columns=columns)


def test_live_scoreboard_empty():
    live = live_scoreboard(_stored([]))
    assert live == {"n": 0}


def test_live_scoreboard_log_loss():
    stored = _stored([
        (0.5, 0.3, 0.2, "home_win", "2026-10-11T00:00:00Z"),
        (0.5, 0.3, 0.2, "draw", "2026-10-11T00:00:00Z"),
        (0.5, 0.3, 0.2, None, None),  # unscored, excluded
    ])
    live = live_scoreboard(stored)
    assert live["n"] == 2
    assert live["log_loss"] == round(-(math.log(0.5) + math.log(0.3)) / 2, 4)


def test_scoreboard_sections_separate():
    board = scoreboard_payload(_stored([]), "2026-09-29T00:00:00Z")
    assert board["backtest"]["models"][0] == {"model": "poisson", "n": 758, "log_loss": 1.0260}
    assert board["live"] == {"n": 0}


def _toy_matches():
    rng = np.random.default_rng(0)
    teams = ["A", "B", "C", "D"]
    rows = []
    for rnd in range(6):
        for home in teams:
            for away in teams:
                if home != away:
                    rows.append({"season": "2025-26" if rnd < 3 else "2026-27",
                                 "date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=rnd * 30),
                                 "home": home, "away": away,
                                 "home_goals": int(rng.poisson(1.5)),
                                 "away_goals": int(rng.poisson(1.1))})
    return pd.DataFrame(rows)


def test_matrix_covers_every_ordered_pair():
    model, _ = fit_production_model(_toy_matches())
    pairings = matrix_payload(model, _toy_matches(), ["A", "B", "C", "D"], "t")["pairings"]
    assert len(pairings) == 12
    assert len({(p["home"], p["away"]) for p in pairings}) == 12
    for p in pairings:
        assert sum(p["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_teams_payload_reference_team_is_zero_and_season_filtered():
    matches = _toy_matches()
    model, _ = fit_production_model(matches)
    teams = teams_payload(model, matches, ["A", "B"], "2026-27", "t")["teams"]
    a = teams[0]
    assert (a["attack"], a["defence"]) == (0.0, 0.0)
    assert a["matches_in_training"] == 36
    assert a["season_matches"] == 18
    assert len(a["last_5"]) == 5 and set(a["last_5"]) <= {"W", "D", "L"}


def test_teams_payload_omits_per_game_without_season_matches():
    matches = _toy_matches()
    model, _ = fit_production_model(matches)
    team = teams_payload(model, matches, ["A"], "2027-28", "t")["teams"][0]
    assert team["season_matches"] == 0
    assert "goals_for_per_game" not in team


@pytest.mark.parametrize(
    "n, level",
    [(0, "low"), (9, "low"), (10, "medium"), (25, "medium"), (26, "high"), (195, "high")],
)
def test_confidence_boundaries(n, level):
    assert confidence_level(n) == level


def test_matrix_confidence_is_the_lower_of_the_two():
    matches = _toy_matches()
    # A newcomer with 2 matches: low, against teams with 36+ (high).
    newcomer = pd.DataFrame([
        {"season": "2026-27", "date": pd.Timestamp("2026-03-01"), "home": "E",
         "away": "A", "home_goals": 1, "away_goals": 1},
        {"season": "2026-27", "date": pd.Timestamp("2026-03-08"), "home": "B",
         "away": "E", "home_goals": 2, "away_goals": 0},
    ])
    matches = pd.concat([matches, newcomer], ignore_index=True)
    model, _ = fit_production_model(matches)
    pairings = matrix_payload(model, matches, ["A", "B", "E"], "t")["pairings"]
    for p in pairings:
        expected = "low" if "E" in (p["home"], p["away"]) else "high"
        assert p["confidence"] == expected


def test_centred_coefficients_average_zero_and_preserve_order():
    matches = _toy_matches()
    model, _ = fit_production_model(matches)
    teams = teams_payload(model, matches, ["A", "B", "C", "D"], "2026-27", "t")["teams"]
    for raw, centred in [("attack", "attack_centred"), ("defence", "defence_centred")]:
        assert sum(t[centred] for t in teams) == pytest.approx(0, abs=1e-3)
        diffs = {round(t[raw] - t[centred], 3) for t in teams}
        assert len(diffs) == 1  # a constant shift
