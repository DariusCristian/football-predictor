from footy.data.teams import CANONICAL_TEAMS, from_openfootball, UnknownTeamError
from footy.data.history import load_seasons
import pytest
from footy.data.history import _full_time_score

def test_dict_shape():
    match = {"score": {"ht": [1, 0], "ft": [2, 1]}}
    assert _full_time_score(match) == [2, 1]


def test_list_shape():
    match = {"score": [0, 0]}
    assert _full_time_score(match) == [0, 0]


def test_missing_score():
    assert _full_time_score({}) is None
    assert _full_time_score({"score": None}) is None


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        _full_time_score({"score": "2-1"})

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def test_all_historical_teams_map():
    df = load_seasons(SEASONS)
    names = set(df.home) | set(df.away)
    assert names <= CANONICAL_TEAMS


def test_unknown_team_raises():
    with pytest.raises(UnknownTeamError):
        from_openfootball("Real Madrid CF")


def test_no_duplicate_canonical_collisions():
    """Two different clubs must never collapse into one canonical name."""
    from footy.data.teams import OPENFOOTBALL_TO_CANONICAL
    values = list(OPENFOOTBALL_TO_CANONICAL.values())
    assert len(values) == len(set(values))