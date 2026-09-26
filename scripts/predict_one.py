
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.data.training import load_all_matches
from footy.model.poisson import fit

model = fit(load_all_matches())

for home, away in [("Liverpool", "Man City"), ("Arsenal", "Leeds United"),
                   ("Everton", "Man City")]:
    p = model.predict(home, away)
    print(f"\n{home} vs {away}")
    print(f"  rates: {p['home_rate']:.2f} - {p['away_rate']:.2f}")
    print(f"  most likely: {p['most_likely_score'][0]}-{p['most_likely_score'][1]}")
    print(f"  home {p['home_win']:.0%}  draw {p['draw']:.0%}  away {p['away_win']:.0%}")