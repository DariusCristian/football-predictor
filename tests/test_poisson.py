import numpy as np
import pandas as pd
import pytest

from footy.model.poisson import outcome_probabilities, score_matrix


def test_probabilities_sum_to_one():
    matrix = score_matrix(1.5, 1.2)
    probs = outcome_probabilities(matrix)
    assert pytest.approx(sum(probs.values()), abs=1e-9) == 1.0


def test_equal_rates_are_symmetric():
    probs = outcome_probabilities(score_matrix(1.4, 1.4))
    assert pytest.approx(probs["home_win"], abs=1e-9) == probs["away_win"]


def test_stronger_home_team_favoured():
    probs = outcome_probabilities(score_matrix(2.5, 0.8))
    assert probs["home_win"] > probs["away_win"]
    assert probs["home_win"] > 0.5


def test_truncation_loses_little_mass():
    assert score_matrix(2.0, 2.0).sum() > 0.9999

    import pandas as pd

from footy.model.poisson import decay_weights


def _dates(*day_offsets):
    base = pd.Timestamp("2026-01-01")
    return pd.Series([base - pd.Timedelta(days=d) for d in day_offsets])


def test_recent_match_has_weight_one():
    weights = decay_weights(_dates(0), half_life_days=365)
    assert pytest.approx(weights.iloc[0]) == 1.0


def test_one_half_life_halves_the_weight():
    weights = decay_weights(_dates(0, 365), half_life_days=365)
    assert pytest.approx(weights.iloc[1], abs=1e-9) == 0.5


def test_two_half_lives_quarter_the_weight():
    weights = decay_weights(_dates(0, 730), half_life_days=365)
    assert pytest.approx(weights.iloc[1], abs=1e-9) == 0.25


def test_weights_decrease_monotonically():
    weights = decay_weights(_dates(0, 100, 200, 400), half_life_days=200)
    assert list(weights) == sorted(weights, reverse=True)


def test_invalid_half_life_raises():
    with pytest.raises(ValueError):
        decay_weights(_dates(0), half_life_days=0)