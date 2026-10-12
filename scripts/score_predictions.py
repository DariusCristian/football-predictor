import sys
from pathlib import Path

# The editable-install .pth is unreliable here: iCloud marks .venv files hidden
# and Python 3.13 skips hidden .pth files. Put src/ on the path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from footy.model.scoring import log_loss
from footy.store import load_predictions, score_pending

BACKTEST_LOG_LOSS = 1.0260

updated = score_pending()
print(f"scored {updated} newly finished predictions\n")

predictions = load_predictions()
scored = predictions[predictions["scored_at"].notna()]

if scored.empty:
    print("No predictions have been scored yet; no matches have finished.")
    raise SystemExit

losses = []
for row in scored.itertuples():
    probs = {"home_win": row.p_home, "draw": row.p_draw, "away_win": row.p_away}
    losses.append(log_loss(probs, row.outcome))
    print(
        f"{row.home:<16} {int(row.actual_home_goals)}-{int(row.actual_away_goals)} "
        f"{row.away:<16}  model gave {probs[row.outcome]:.0%} to what happened"
    )

print(f"\nn = {len(scored)}   mean log loss = {np.mean(losses):.4f}")
print(
    f"(backtest was {BACKTEST_LOG_LOSS:.4f}; {len(scored)} matches is far "
    "too small a sample to compare)"
)
