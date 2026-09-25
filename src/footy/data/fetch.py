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