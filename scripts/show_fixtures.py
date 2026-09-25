import sys
from pathlib import Path

# The editable-install .pth is unreliable here: iCloud marks .venv files hidden
# and Python 3.13 skips hidden .pth files. Put src/ on the path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from footy.data.fetch import fetch_matches

for match in fetch_matches(status="SCHEDULED")[:10]:
    home = match["homeTeam"]["shortName"]
    away = match["awayTeam"]["shortName"]
    print(f"MD{match['matchday']:>2}  {match['utcDate'][:10]}  {home} vs {away}")