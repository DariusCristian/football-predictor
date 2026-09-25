import json
from pathlib import Path

import pandas as pd
import requests

from footy.config import DATA_DIR

from footy.data.teams import from_openfootball

BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master"


def _cache_path(season: str) -> Path:
    return DATA_DIR / f"openfootball_{season}.json"


def download_season(season: str, force: bool = False) -> dict:
    """Download one season, caching it locally. season looks like '2020-21'."""
    path = _cache_path(season)
    if path.exists() and not force:
        return json.loads(path.read_text())

    response = requests.get(f"{BASE_URL}/{season}/en.1.json", timeout=30)
    response.raise_for_status()
    path.write_text(response.text)
    return response.json()


def _full_time_score(match: dict) -> list[int] | None:
    """Extract [home_goals, away_goals], handling both OpenFootball shapes.

    Older seasons:  "score": {"ht": [..], "ft": [h, a]}
    2025-26 (some): "score": [h, a]
    """
    score = match.get("score")
    if score is None:
        return None
    if isinstance(score, dict):
        return score.get("ft")
    if isinstance(score, list):
        return score
    raise ValueError(f"Unrecognised score shape {type(score).__name__}: {score!r}")


def season_to_frame(season: str) -> pd.DataFrame:
    raw = download_season(season)
    rows = []
    for match in raw["matches"]:
        score = _full_time_score(match)
        if not score or len(score) != 2:
            continue  # unplayed or malformed
        rows.append(
            {
                "season": season,
                "date": match["date"],
                "home_raw": match["team1"],
                "away_raw": match["team2"],
                "home_goals": score[0],
                "away_goals": score[1],
                "home": from_openfootball(match["team1"]),
                "away": from_openfootball(match["team2"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_seasons(seasons: list[str]) -> pd.DataFrame:
    frames = [season_to_frame(s) for s in seasons]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)