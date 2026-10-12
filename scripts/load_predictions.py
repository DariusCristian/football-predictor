import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from footy.store import score_pending, load_predictions

updated = score_pending()
print(f"scored {updated} predictions\n")

df = load_predictions()
scored = df[df["outcome"].notna()]

if scored.empty:
    print("nothing scored yet")
    raise SystemExit

losses = []
for _, r in scored.iterrows():
    probs = {"home_win": r["p_home"], "draw": r["p_draw"], "away_win": r["p_away"]}
    hit = probs[r["outcome"]]
    losses.append(-np.log(max(hit, 1e-15)))
    print(f"{r['home']:<16} {int(r['actual_home_goals'])}-{int(r['actual_away_goals'])} "
          f"{r['away']:<16}  model gave {hit:.0%}")

print(f"\nn = {len(scored)}   mean log loss = {np.mean(losses):.4f}")
print("(backtest was 1.0260; nine matches is far too few to compare)")