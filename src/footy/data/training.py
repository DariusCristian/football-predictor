"""Assembling training data.

This module is the single place where past and present data meet, which
makes it the single place data leakage can occur. Every match returned
by load_training_data() was completed strictly before as_of.
"""

import pandas as pd

from footy.data.fetch import finished_matches_frame
from footy.data.history import load_seasons

HISTORICAL_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

KEY = ["season", "home", "away"]
SCHEMA = ["season", "date", "home_raw", "away_raw", "home", "away",
          "home_goals", "away_goals"]


def load_all_matches(use_api: bool = True) -> pd.DataFrame:
    """Every completed match known, from both sources, deduplicated."""
    frames = [load_seasons(HISTORICAL_SEASONS)]
    if use_api:
        live = finished_matches_frame()
        if not live.empty:
            frames.append(live)

    combined = pd.concat(frames, ignore_index=True)
    # History wins on conflict: it's settled, the API may still be correcting.
    combined = combined.drop_duplicates(subset=KEY, keep="first")
    return combined.sort_values("date").reset_index(drop=True)[SCHEMA]


def load_training_data(as_of: pd.Timestamp, use_api: bool = True) -> pd.DataFrame:
    """Matches completed strictly before as_of. The leakage boundary."""
    matches = load_all_matches(use_api=use_api)
    return matches[matches["date"] < pd.Timestamp(as_of)].reset_index(drop=True)