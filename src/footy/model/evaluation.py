"""Train / validation / test protocol.

Splits are chronological, never random: a random split would let the
model learn from matches played after the ones it predicts.

Validation is for tuning. Test is run once, after every choice is
locked, and reported as-is.
"""

import pandas as pd

VALIDATION_START = pd.Timestamp("2023-08-01")
TEST_START = pd.Timestamp("2024-08-01")
# Frozen at the end of the 2025-26 season so the test set is fixed and
# single-source (OpenFootball history only). An open end would pull in
# live API matches, growing the set on every run.
TEST_END = pd.Timestamp("2026-06-30")


def validation_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    return VALIDATION_START, TEST_START


def test_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    return TEST_START, TEST_END