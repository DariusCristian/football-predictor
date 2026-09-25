import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
from collections import Counter
from footy.data.history import download_season

for season in ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]:
    raw = download_season(season)
    shapes = Counter()
    examples = {}
    for match in raw["matches"]:
        score = match.get("score")
        kind = type(score).__name__
        shapes[kind] += 1
        if kind not in examples:
            examples[kind] = match
    print(f"\n=== {season} ===")
    for kind, count in shapes.items():
        print(f"  {kind}: {count}")
    for kind, match in examples.items():
        if kind != "dict":
            print(f"  example of {kind}:")
            print(json.dumps(match, indent=2)[:400])