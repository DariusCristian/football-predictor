import pandas as pd
import pytest

from footy.store import (
    PredictionExistsError,
    load_predictions,
    predicted_keys,
    save_predictions,
    score_pending,
)


def _row(home="Arsenal", away="Chelsea", season="2026-27"):
    return {
        "match_date": "2026-10-10T14:00:00Z",
        "season": season,
        "home": home,
        "away": away,
        "p_home": 0.5,
        "p_draw": 0.3,
        "p_away": 0.2,
        "most_likely_home": 1,
        "most_likely_away": 0,
        "model_version": "test",
    }


def _finished(home, away, home_goals, away_goals, season="2026-27"):
    return pd.DataFrame([{"season": season, "home": home, "away": away,
                          "home_goals": home_goals, "away_goals": away_goals}])


@pytest.fixture
def db(tmp_path):
    return tmp_path / "predictions.db"


def test_duplicate_save_raises(db):
    save_predictions([_row()], db_path=db)
    with pytest.raises(PredictionExistsError):
        save_predictions([_row()], db_path=db)


def test_duplicate_does_not_overwrite(db):
    save_predictions([_row()], db_path=db)
    revised = {**_row(), "p_home": 0.9, "p_draw": 0.05, "p_away": 0.05}
    with pytest.raises(PredictionExistsError):
        save_predictions([revised], db_path=db)
    assert load_predictions(db_path=db)["p_home"].tolist() == [0.5]


def test_conflicting_batch_saves_nothing(db):
    save_predictions([_row()], db_path=db)
    with pytest.raises(PredictionExistsError):
        save_predictions([_row("Leeds United", "Fulham"), _row()], db_path=db)
    assert len(load_predictions(db_path=db)) == 1


def test_same_fixture_next_season_is_allowed(db):
    save_predictions([_row()], db_path=db)
    save_predictions([_row(season="2027-28")], db_path=db)
    assert predicted_keys(db_path=db) == {
        ("2026-27", "Arsenal", "Chelsea"), ("2027-28", "Arsenal", "Chelsea")
    }


def test_scoring_fills_the_right_row(db):
    save_predictions([_row(), _row("Chelsea", "Arsenal")], db_path=db)
    assert score_pending(_finished("Chelsea", "Arsenal", 2, 2), db_path=db) == 1

    rows = load_predictions(db_path=db).set_index(["home", "away"])
    scored = rows.loc[("Chelsea", "Arsenal")]
    assert (scored["actual_home_goals"], scored["actual_away_goals"]) == (2, 2)
    assert scored["outcome"] == "draw"
    assert scored["scored_at"] is not None
    assert pd.isna(rows.loc[("Arsenal", "Chelsea"), "scored_at"])


def test_unfinished_match_left_unscored(db):
    save_predictions([_row()], db_path=db)
    assert score_pending(_finished("Leeds United", "Fulham", 1, 0), db_path=db) == 0
    assert score_pending(pd.DataFrame(), db_path=db) == 0
    row = load_predictions(db_path=db).iloc[0]
    assert pd.isna(row["scored_at"]) and pd.isna(row["outcome"])


def test_scoring_matches_on_season(db):
    save_predictions([_row()], db_path=db)
    assert score_pending(_finished("Arsenal", "Chelsea", 1, 0, season="2025-26"),
                         db_path=db) == 0


def test_scored_row_is_not_rescored(db):
    save_predictions([_row()], db_path=db)
    score_pending(_finished("Arsenal", "Chelsea", 1, 0), db_path=db)
    assert score_pending(_finished("Arsenal", "Chelsea", 0, 3), db_path=db) == 0
    assert load_predictions(db_path=db).iloc[0]["outcome"] == "home_win"
