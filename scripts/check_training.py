# scripts/check_training.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from footy.data.training import load_all_matches

df = load_all_matches()
print(df.groupby("season").size())
print(f"\ntotal: {len(df)}")
print(df.tail())