import sys
from pathlib import Path

# The editable-install .pth is unreliable here: iCloud marks .venv files hidden
# and Python 3.13 skips hidden .pth files. Put src/ on the path explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from footy.data.fetch import upcoming_fixtures
from footy.data.training import load_all_matches
from footy.production import MODEL_VERSION, fit_production_model, prediction_row
from footy.store import predicted_keys, save_predictions

WITHIN_DAYS = 14

fixtures = upcoming_fixtures(within_days=WITHIN_DAYS)
existing = predicted_keys()
new = [f for f in fixtures if (f["season"], f["home"], f["away"]) not in existing]
skipped = [f for f in fixtures if (f["season"], f["home"], f["away"]) in existing]

print(f"{len(fixtures)} fixtures in the next {WITHIN_DAYS} days; "
      f"{len(new)} new, {len(skipped)} already predicted\n")

if new:
    matches = load_all_matches()
    model, degenerate = fit_production_model(matches)
    for message in degenerate:
        print(f"WARNING DegenerateFitWarning: {message}")
    print(f"Fitted {MODEL_VERSION} on {len(matches)} completed matches "
          f"(latest {matches['date'].max().date()})\n")

    rows = [prediction_row(model, f) for f in new]
    save_predictions(rows)
    print("Saved:")
    for r in rows:
        print(f"  {r['match_date']}  {r['home']:>15} v {r['away']:<15} "
              f"H {r['p_home']:.3f}  D {r['p_draw']:.3f}  A {r['p_away']:.3f}  "
              f"({r['most_likely_home']}-{r['most_likely_away']})")

if skipped:
    print("Skipped (already predicted):")
    for f in skipped:
        print(f"  {f['match_date']}  {f['home']:>15} v {f['away']}")
