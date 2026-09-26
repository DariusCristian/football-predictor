import requests
from footy.config import API_BASE, COMPETITION, FOOTBALL_DATA_TOKEN


class FootballDataError(Exception):
    pass


def _get(path, params=None):
    if not FOOTBALL_DATA_TOKEN:
        raise FootballDataError("FOOTBALL_DATA_TOKEN is not set")

    response = requests.get(
        f"{API_BASE}{path}",
        headers={"X-Auth-Token": FOOTBALL_DATA_TOKEN},
        params=params or {},
        timeout=15,
    )
    if response.status_code != 200:
        raise FootballDataError(f"{response.status_code}: {response.text[:200]}")
    return response.json()


def fetch_matches(status=None, matchday=None):
    params = {}
    if status:
        params["status"] = status
    if matchday:
        params["matchday"] = matchday
    return _get(f"/competitions/{COMPETITION}/matches", params)["matches"]

import pandas as pd

from footy.data.teams import from_api


def _season_label(utc_date: str) -> str:
    """'2026-10-10' -> '2026-27'. Seasons start in July."""
    year, month = int(utc_date[:4]), int(utc_date[5:7])
    start = year if month >= 7 else year - 1
    return f"{start}-{str(start + 1)[2:]}"


def finished_matches_frame() -> pd.DataFrame:
    """Completed matches from the current season, in the shared schema."""
    rows = []
    for match in fetch_matches(status="FINISHED"):
        score = match["score"]["fullTime"]
        if score["home"] is None or score["away"] is None:
            continue
        rows.append(
            {
                "season": _season_label(match["utcDate"]),
                "date": match["utcDate"][:10],
                "home_raw": match["homeTeam"]["shortName"],
                "away_raw": match["awayTeam"]["shortName"],
                "home": from_api(match["homeTeam"]["shortName"]),
                "away": from_api(match["awayTeam"]["shortName"]),
                "home_goals": score["home"],
                "away_goals": score["away"],
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame