import math

import numpy as np
import pandas as pd
import pytest

from footy.web.site_data import (
    SlugCollisionError,
    check_unique_slugs,
    confidence_level,
    fixture_detail_payload,
    live_scoreboard,
    match_slug,
    matrix_payload,
    predictions_payload,
    relative_improvement,
    round_probabilities,
    scoreboard_payload,
    slugify,
    standings_payload,
    teams_payload,
    top_scores,
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


def test_top_scores_agree_with_predict():
    """top_scores rebuilds predict()'s matrix; pin the two together."""
    model, _ = fit_production_model(_toy_matches())
    cells = 11 * 11
    tolerance = cells * 0.5 * 10**-4  # worst case: every cell rounded to 4 dp
    for home, away in [("A", "B"), ("C", "D"), ("D", "A")]:
        p = model.predict(home, away)
        everything = top_scores(model, home, away, k=cells)
        assert (everything[0]["home"], everything[0]["away"]) == p["most_likely_score"]
        assert sum(c["p"] for c in everything) == pytest.approx(1.0, abs=tolerance)
        home_win = sum(c["p"] for c in everything if c["home"] > c["away"])
        draw = sum(c["p"] for c in everything if c["home"] == c["away"])
        assert home_win == pytest.approx(p["home_win"], abs=tolerance)
        assert draw == pytest.approx(p["draw"], abs=tolerance)


def test_top_scores_sorted_and_sized():
    model, _ = fit_production_model(_toy_matches())
    scores = top_scores(model, "A", "B")
    assert len(scores) == 10
    assert len({(c["home"], c["away"]) for c in scores}) == 10
    ps = [c["p"] for c in scores]
    assert ps == sorted(ps, reverse=True)


def test_matrix_pairings_carry_top_scores():
    model, _ = fit_production_model(_toy_matches())
    pairing = matrix_payload(model, _toy_matches(), ["A", "B"], "t")["pairings"][0]
    assert pairing["top_scores"] == top_scores(model, "A", "B")


def test_multipliers_are_exp_of_centred():
    matches = _toy_matches()
    model, _ = fit_production_model(matches)
    for team in teams_payload(model, matches, ["A", "B", "C", "D"], "2026-27", "t")["teams"]:
        assert team["attack_multiplier"] == pytest.approx(math.exp(team["attack_centred"]), abs=1e-3)
        assert team["defence_multiplier"] == pytest.approx(math.exp(team["defence_centred"]), abs=1e-3)


def test_relative_improvement_matches_headline():
    backtest = {"models": [{"model": "poisson", "log_loss": 1.0260},
                           {"model": "league_average", "log_loss": 1.0832}]}
    assert relative_improvement(backtest) == pytest.approx(0.0528, abs=1e-4)


def test_scoreboard_publishes_improvement_and_uniform_reference():
    backtest = scoreboard_payload(_stored([]), "t")["backtest"]
    assert backtest["improvement_over_league_average"] == pytest.approx(0.0528, abs=1e-4)
    assert backtest["uniform_log_loss"] == pytest.approx(math.log(3), abs=1e-4)


# ---------- Slugs ----------

@pytest.mark.parametrize(
    "name, slug",
    [("Leeds United", "leeds-united"), ("Arsenal", "arsenal"), ("Man City", "man-city"),
     ("Brighton Hove", "brighton-hove"), ("Nott'm Forest", "nott-m-forest"),
     ("Brighton & Hove Albion", "brighton-hove-albion")],
)
def test_slugify(name, slug):
    assert slugify(name) == slug


def test_match_slug_is_home_then_away():
    assert match_slug("Arsenal", "Leeds United") == "arsenal-leeds-united"
    assert match_slug("Leeds United", "Arsenal") == "leeds-united-arsenal"


def test_check_unique_slugs_raises_on_collision():
    # "Man" v "City Hull" and "Man City" v "Hull" join to the same slug:
    # the case the build guards against.
    slugs = [match_slug("Man", "City Hull"), match_slug("Man City", "Hull")]
    with pytest.raises(SlugCollisionError, match="man-city-hull"):
        check_unique_slugs(slugs, "pairing")


def test_payloads_carry_slugs():
    matches = _toy_matches()
    model, _ = fit_production_model(matches)
    pairings = matrix_payload(model, matches, ["A", "B", "C", "D"], "t")["pairings"]
    assert all(p["slug"] == match_slug(p["home"], p["away"]) for p in pairings)
    teams = teams_payload(model, matches, ["A", "B"], "2026-27", "t")["teams"]
    assert [t["slug"] for t in teams] == ["a", "b"]


def _stored_upcoming(rows):
    columns = ["match_date", "home", "away", "p_home", "p_draw", "p_away",
               "most_likely_home", "most_likely_away", "model_version",
               "created_at", "scored_at"]
    return pd.DataFrame([(*r, 0.5, 0.3, 0.2, 1, 0, "v", "2026-10-01T00:00:00Z", None)
                         for r in rows], columns=columns)


def test_predictions_payload_slug():
    stored = _stored_upcoming([("2026-10-10T14:00:00Z", "Arsenal", "Leeds United")])
    pred = predictions_payload(stored, "2026-10-02T00:00:00Z")["predictions"][0]
    assert pred["slug"] == "arsenal-leeds-united"


# ---------- Standings ----------

def _match(date, home, away, hg, ag, season="2026-27"):
    return {"season": season, "date": pd.Timestamp(date), "home": home, "away": away,
            "home_goals": hg, "away_goals": ag}


def test_standings_identities_hold_for_every_team():
    matches = _toy_matches()
    table = standings_payload(matches, ["A", "B", "C", "D"], "2026-27", "t")["standings"]
    assert len(table) == 4
    for row in table:
        assert row["points"] == 3 * row["won"] + row["drawn"]
        assert row["played"] == row["won"] + row["drawn"] + row["lost"]
        assert row["goal_difference"] == row["goals_for"] - row["goals_against"]
    assert sum(r["goals_for"] for r in table) == sum(r["goals_against"] for r in table)
    assert [r["position"] for r in table] == [1, 2, 3, 4]


def test_standings_sorted_by_points_then_goal_difference_then_goals_for():
    matches = pd.DataFrame([
        # A and B both 3 points, GD +1; B scored more. C 3 points, GD +2.
        _match("2026-08-16", "A", "D", 1, 0),
        _match("2026-08-16", "B", "E", 3, 2),
        _match("2026-08-16", "C", "F", 2, 0),
        # G: 4 points from two matches, tops the table.
        _match("2026-08-09", "G", "H", 1, 1),
        _match("2026-08-23", "H", "G", 0, 1),
        # Last season's results don't count.
        _match("2026-03-01", "D", "A", 9, 0, season="2025-26"),
    ])
    payload = standings_payload(matches, list("ABCDEFGH"), "2026-27", "t")
    order = [r["team"] for r in payload["standings"]]
    assert order[:4] == ["G", "C", "B", "A"]
    a = next(r for r in payload["standings"] if r["team"] == "A")
    assert (a["played"], a["won"], a["goals_for"], a["goals_against"]) == (1, 1, 1, 0)
    assert payload["matchday"] == 2
    assert payload["season"] == "2026-27"


def test_standings_include_teams_yet_to_play_and_mark_confidence():
    matches = _toy_matches()
    table = standings_payload(matches, ["A", "B", "C", "D", "Z"], "2026-27", "t")["standings"]
    z = next(r for r in table if r["team"] == "Z")
    assert (z["played"], z["points"], z["position"]) == (0, 0, 5)
    assert z["confidence"] == "low"
    a = next(r for r in table if r["team"] == "A")
    assert (a["matches_in_training"], a["confidence"]) == (36, "high")


def test_standings_empty_season():
    payload = standings_payload(_toy_matches(), ["A", "B"], "2030-31", "t")
    assert payload["matchday"] == 0
    assert all(r["played"] == 0 for r in payload["standings"])


# ---------- Fixture detail ----------

def _history():
    return pd.DataFrame([
        _match("2024-09-01", "A", "B", 2, 0, "2024-25"),
        _match("2025-02-01", "B", "A", 1, 1, "2024-25"),
        _match("2025-09-01", "B", "A", 3, 1, "2025-26"),
        _match("2025-10-01", "A", "C", 0, 1, "2025-26"),
        _match("2026-10-10", "A", "B", 5, 0, "2026-27"),  # the fixture's own day
    ])


def test_head_to_head_from_home_perspective_most_recent_first():
    preds = [{"slug": "a-b", "home": "A", "away": "B", "kickoff": "2026-10-10T14:00:00Z"}]
    h2h = fixture_detail_payload(preds, _history(), "t")["fixtures"]["a-b"]["head_to_head"]
    assert h2h["perspective"] == "A"
    assert [m["date"] for m in h2h["meetings"]] == ["2025-09-01", "2025-02-01", "2024-09-01"]
    assert [m["result"] for m in h2h["meetings"]] == ["L", "D", "W"]
    assert (h2h["played"], h2h["won"], h2h["drawn"], h2h["lost"]) == (3, 1, 1, 1)


def test_head_to_head_empty_when_never_met():
    preds = [{"slug": "b-c", "home": "B", "away": "C", "kickoff": "2026-10-10T14:00:00Z"}]
    h2h = fixture_detail_payload(preds, _history(), "t")["fixtures"]["b-c"]["head_to_head"]
    assert h2h["played"] == 0 and h2h["meetings"] == []


def test_recent_results_capped_newest_first_and_before_kickoff():
    rows = [_match(pd.Timestamp("2025-08-01") + pd.Timedelta(days=7 * i), "A", "B", i % 3, 1)
            for i in range(14)]
    rows.append(_match("2026-10-10", "A", "B", 9, 9))  # kickoff day: excluded
    preds = [{"slug": "b-a", "home": "B", "away": "A", "kickoff": "2026-10-10T14:00:00Z"}]
    recent = fixture_detail_payload(preds, pd.DataFrame(rows), "t")["fixtures"]["b-a"]["recent"]
    assert len(recent["home"]) == 10
    dates = [r["date"] for r in recent["away"]]
    assert dates == sorted(dates, reverse=True)
    assert "2026-10-10" not in dates
    latest = recent["away"][0]  # i = 13: A 1-1 B at home
    assert (latest["opponent"], latest["venue"], latest["goals_for"],
            latest["goals_against"], latest["result"]) == ("B", "home", 1, 1, "D")
    assert recent["home"][0]["venue"] == "away"
