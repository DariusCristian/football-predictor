import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
from footy.data.history import download_season

for season in ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]:
    try:
        raw = download_season(season)
    except Exception as exc:
        print(f"{season}: FAILED {exc}")
        continue
    matches = raw.get("matches", [])
    print(f"\n=== {season}: {len(matches)} matches ===")
    if matches:
        print(json.dumps(matches[0], indent=2)[:400])