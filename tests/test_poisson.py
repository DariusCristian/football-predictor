import numpy as np
import pandas as pd
import pytest

from footy.model.poisson import (
    dixon_coles_correction,
    outcome_probabilities,
    score_matrix,
)


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

def _draw_total(matrix):
    return float(np.trace(matrix))


def test_dixon_coles_rho_zero_is_noop():
    matrix = score_matrix(1.5, 1.2)
    corrected = dixon_coles_correction(matrix, 1.5, 1.2, rho=0.0)
    # Only renormalisation of the truncated matrix; no cell reweighted.
    np.testing.assert_allclose(corrected, matrix / matrix.sum(), atol=1e-15)


def test_dixon_coles_sums_to_one():
    for rho in (-0.2, -0.1, 0.05, 0.1):
        corrected = dixon_coles_correction(score_matrix(1.6, 0.9), 1.6, 0.9, rho)
        assert pytest.approx(corrected.sum(), abs=1e-12) == 1.0


def test_dixon_coles_non_negative_within_valid_rho_range():
    rates = np.arange(0.5, 3.01, 0.25)
    for lh in rates:
        for la in rates:
            for rho in np.arange(-0.2, 0.2001, 0.05):
                if rho > 1 / (lh * la):
                    continue  # outside the method's valid range; see below
                corrected = dixon_coles_correction(
                    score_matrix(lh, la), lh, la, rho
                )
                assert (corrected >= 0).all(), (lh, la, rho)


def test_dixon_coles_rejects_rho_that_makes_nil_nil_negative():
    # tau(0,0) = 1 - lh*la*rho < 0 when rho > 1/(lh*la): 1 - 2*3*0.2 = -0.2.
    # Realistic-looking inputs, so this is a real constraint on rho.
    with pytest.raises(ValueError):
        dixon_coles_correction(score_matrix(2.0, 3.0), 2.0, 3.0, rho=0.2)


def test_dixon_coles_negative_rho_increases_draws():
    # With the standard tau, negative rho inflates 0-0 and 1-1.
    matrix = score_matrix(1.4, 1.1)
    base = _draw_total(dixon_coles_correction(matrix, 1.4, 1.1, rho=0.0))
    assert _draw_total(dixon_coles_correction(matrix, 1.4, 1.1, rho=-0.1)) > base
    assert _draw_total(dixon_coles_correction(matrix, 1.4, 1.1, rho=0.1)) < base


def test_dixon_coles_changes_only_low_score_cells():
    matrix = score_matrix(1.4, 1.1)
    base = dixon_coles_correction(matrix, 1.4, 1.1, rho=0.0)
    corrected = dixon_coles_correction(matrix, 1.4, 1.1, rho=-0.1)
    # Renormalisation shifts every cell by a common factor, so compare
    # each cell's ratio to that factor.
    ratio = corrected / base
    scale = ratio[2, 2]
    changed = {(i, j) for i, j in zip(*np.where(~np.isclose(ratio, scale)))}
    assert changed == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert pytest.approx(corrected[2, 2] / scale) == base[2, 2]
