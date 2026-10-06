# Premier League Match Predictor

Predicts Premier League match outcomes as probabilities, commits each prediction before kickoff, and scores it afterwards against baselines.

**Live site:** https://dariuscristian.github.io/football-predictor/

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-GLM-4B8BBE)
![pytest](https://img.shields.io/badge/tests-86_passing-0A9EDC?logo=pytest&logoColor=white)
![GitHub Pages](https://img.shields.io/badge/deployed-GitHub_Pages-222222?logo=github)

---

## Results

Evaluated on a frozen held-out test window (the 2024-25 and 2025-26 seasons), using walk-forward prediction: every match is predicted by a model trained only on matches completed strictly before it.

| Model | n | Log loss | Brier |
|---|---|---|---|
| **Poisson GLM (ridge, α = 1e-5)** | 758 | **1.0260** | **0.614** |
| League-average baseline | 760 | 1.0832 | 0.656 |
| Always home win | 760 | 1.7901 | 1.006 |

**5.3% lower log loss than the league-average baseline.** Football outcomes are dominated by variance, so a modest margin over a well-chosen baseline is the realistic target. The model outputs probabilities, never single picks, and is scored with proper scoring rules.

The Poisson model scores 2 fewer matches because the opening games of two promoted teams are skipped: neither team appears in the training data at that point.

## How it works

1. **Data.** 1,900 historical matches (2021-22 to 2025-26) from OpenFootball, combined with live fixtures and results from the football-data.org API. Where the sources overlap, the settled historical record wins. Team names from both sources are mapped into one canonical set, and an unknown name raises an error rather than being silently dropped.
2. **Model.** A Poisson regression (statsmodels GLM) estimates each team's attack and defence strength plus a home advantage. Expected goals for both sides give a full scoreline probability matrix, which is summed into home win / draw / away win probabilities.
3. **Regularisation.** A small ridge penalty on team coefficients prevents complete separation: without it, a newly promoted team with zero goals in its first few matches gets a coefficient near −∞, and real outcomes are assigned probabilities around 1e-10.
4. **Predictions store.** Upcoming fixtures are predicted and committed before kickoff, then scored once results arrive, so live performance can't be adjusted after the fact. Each prediction carries a confidence level based on how many matches each team has in the training data.
5. **Site.** A static site generated from JSON and deployed to GitHub Pages by GitHub Actions, with views for upcoming fixtures, match detail (score matrix, recent form, head-to-head), the league table, team ratings, the prediction record, and the method.

## Evaluation protocol

- **Train / validate / test split.** Hyperparameters were tuned on a validation window (2023-24), and the test window was used only for final numbers.
- **No leakage.** Walk-forward refitting; the model never sees matches played on or after the prediction date.
- **Baselines.** Every result is reported against league-average and always-home baselines.
- **Every experiment is logged**, including negative results, in [`docs/results.md`](docs/results.md).

### What the experiments showed

- **Selection bias is real and large.** Tuning time-decay weighting and reporting on the same backtest suggested a 5.4% gain. A proper held-out split showed most of that was selection bias, and validation chose no time weighting at all.
- **Dixon-Coles correction: not adopted.** Validation selected ρ = 0. The draw under-prediction that motivated it did not replicate outside the test window. The implementation is kept behind a no-op default so the negative result is reproducible.
- **Calibration findings did not replicate.** Apparent overconfidence on one window vanished on another; with ~100 predictions per bin, most gaps were within noise.
- **Ridge fixes a correctness failure, not a skill gap.** Almost all of the improvement from regularisation comes from one impossible prediction becoming merely a poor one. The model is no better at football; the number is simply no longer distorted by a pathological fit.
- **Expected goals (xG): investigated, not implemented.** No source offered Premier League per-match xG from 2021-22 onwards under terms permitting automated access. The investigation is documented in [`docs/data-sources.md`](docs/data-sources.md).

## Project structure

```
src/footy/
├── data/           # API client, historical loader, team-name mapping, training sets
├── model/          # Poisson model, baselines, backtest, scoring, calibration
├── web/            # site data generation
├── config.py       # settings and environment
├── production.py   # the locked production model configuration
└── store.py        # predictions committed before kickoff
scripts/            # backtests, tuning, final evaluation, predictions, site build
tests/              # pytest suite
site/               # static site deployed to GitHub Pages
docs/               # results log, decisions log, data-source investigation
```

## Getting started

Requires Python 3.13 and a free API token from [football-data.org](https://www.football-data.org/).

```bash
git clone https://github.com/DariusCristian/football-predictor.git
cd football-predictor
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "FOOTBALL_DATA_TOKEN=your_token" > .env
```

Run the tests (86 tests, covering configuration, historical data parsing, training
