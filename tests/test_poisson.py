import numpy as np
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