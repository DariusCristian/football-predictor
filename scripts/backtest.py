import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footy.data.training import load_all_matches
from footy.model.backtest import compare
from footy.model.baselines import AlwaysHomeWin, LeagueAverage
from footy.model.poisson import fit as fit_poisson

matches = load_all_matches()

predictors = {
    "always_home": lambda training: AlwaysHomeWin().fit(training),
    "league_average": lambda training: LeagueAverage().fit(training),
    "poisson": fit_poisson,
}

print("Running walk-forward backtest from 2023-08-01...")
print("(this takes a minute — the Poisson model refits weekly)\n")

results = compare(matches, predictors, start_date="2023-08-01")
print(results.to_string(index=False))