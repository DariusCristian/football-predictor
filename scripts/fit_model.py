import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from footy.data.training import load_all_matches
from footy.model.poisson import fit

matches = load_all_matches()
model = fit(matches)

print(model._result.summary())

print("\n--- home advantage ---")
print(f"coefficient: {model._result.params['home']:.4f}")
print(f"multiplier:  {np.exp(model._result.params['home']):.3f}")

print("\n--- attack strength (higher = scores more) ---")
attack = {
    name.split("T.")[1].rstrip("]"): value
    for name, value in model._result.params.items()
    if name.startswith("C(team)")
}
for team, value in sorted(attack.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {team:<18} {value:+.3f}")