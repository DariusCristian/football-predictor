import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.data.fetch import fetch_matches

names = set()
for status in ("SCHEDULED", "FINISHED"):
    for match in fetch_matches(status=status):
        names.add(match["homeTeam"]["shortName"])
        names.add(match["awayTeam"]["shortName"])

for name in sorted(names):
    print(name)
print(f"\n{len(names)} teams")