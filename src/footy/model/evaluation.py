"""Train / validation / test protocol.

Splits are chronological, never random: a random split would let the
model learn from matches played after the ones it predicts.

Validation is for tuning. Test is run once, after every choice is
locked, and reported as-is.
"""

import pandas as pd

VALIDATION_START = pd.Timestamp("2023-08-01")
TEST_START = pd.Timestamp("2024-08-01")


def validation_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    return VALIDATION_START, TEST_START


def test_window() -> tuple[pd.Timestamp, pd.Timestamp | None]:
    return TEST_START, None