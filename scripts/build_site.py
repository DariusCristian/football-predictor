import sys
from pathlib import Path

# The editable-install .pth is unreliable here: iCloud marks .venv files hidden
# and Python 3.13 skips hidden .pth files. Put src/ on the path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json

from footy.config import PROJECT_ROOT
from footy.data.fetch import current_season, current_season_teams
from footy.data.training import load_all_matches
from footy.production import fit_production_model
from footy.store import load_predictions, utc_now
from footy.web.site_data import (
    fixture_detail_payload,
    history_payload,
    matrix_payload,
    predictions_payload,
    scoreboard_payload,
    standings_payload,
    teams_payload,
)

OUT_DIR = PROJECT_ROOT / "site" / "data"
HISTORY_SIZE_LIMIT = 400 * 1024

generated_at = utc_now()
matches = load_all_matches()
teams = current_season_teams()
stored = load_predictions()

model, degenerate = fit_production_model(matches)
for message in degenerate:
    print(f"WARNING DegenerateFitWarning: {message}")

predictions = predictions_payload(stored, generated_at)
season = current_season()
payloads = {
    "predictions.json": predictions,
    "matrix.json": matrix_payload(model, matches, teams, generated_at),
    "teams.json": teams_payload(model, matches, teams, season, generated_at),
    "history.json": history_payload(matches, generated_at),
    "scoreboard.json": scoreboard_payload(stored, generated_at),
    "standings.json": standings_payload(matches, teams, season, generated_at),
    "fixture_detail.json": fixture_detail_payload(
        predictions["predictions"], matches, generated_at),
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, payload in payloads.items():
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    size = len(text.encode())
    if name == "history.json" and size > HISTORY_SIZE_LIMIT:
        print(f"NOT WRITTEN {name}: {size / 1024:.1f} KB exceeds "
              f"{HISTORY_SIZE_LIMIT // 1024} KB; trim before shipping")
        continue
    (OUT_DIR / name).write_text(text)
    print(f"wrote {name:<20} {size / 1024:7.1f} KB")

print(f"\n{len(degenerate)} DegenerateFitWarning(s)")
