import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.data.history import load_seasons

seasons = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
df = load_seasons(seasons)

print(df.shape)
print(df.head())
print()
print(df.groupby("season").size())
print()
for name in sorted(set(df.home_raw) | set(df.away_raw)):
    print(name)