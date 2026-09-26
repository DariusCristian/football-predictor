import pandas as pd
import pytest

from footy.data.training import KEY, SCHEMA, load_all_matches, load_training_data


def test_schema_is_stable():
    df = load_all_matches(use_api=False)
    assert list(df.columns) == SCHEMA


def test_no_duplicate_fixtures():
    df = load_all_matches(use_api=False)
    assert not df.duplicated(subset=KEY).any()


def test_each_season_has_380_matches():
    df = load_all_matches(use_api=False)
    counts = df.groupby("season").size()
    assert (counts == 380).all(), counts.to_dict()


def test_cutoff_excludes_later_matches():
    cutoff = pd.Timestamp("2023-01-01")
    df = load_training_data(cutoff, use_api=False)
    assert (df["date"] < cutoff).all()
    assert len(df) > 0


def test_cutoff_is_strict():
    """A match on the cutoff date is excluded, not included."""
    df = load_all_matches(use_api=False)
    a_date = df["date"].iloc[500]
    before = load_training_data(a_date, use_api=False)
    assert (before["date"] < a_date).all()