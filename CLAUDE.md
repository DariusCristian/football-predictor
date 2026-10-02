# Football Match Predictor

Predicts Premier League match outcomes as probabilities, records
predictions before matches, and scores them afterwards against baselines.

## Conventions
- Python 3.13, code in src/footy; no PYTHONPATH or install needed
  - pytest finds src/ via [tool.pytest.ini_options] pythonpath
  - every script in scripts/ must start with the sys.path bootstrap
    (see scripts/show_fixtures.py). Don't rely on `pip install -e .`: the repo
    is in iCloud-synced ~/Documents, iCloud marks .venv files hidden, and
    Python 3.13 skips hidden .pth files, so the editable install breaks
    minutes after installing.
- Layers: data (fetch/clean), model (fit/predict), web (serve)
- Model outputs probabilities, never single predictions
- Never train on matches that occurred after the match being predicted
- Secrets in .env only, never committed

## Working style
- Explain design choices before implementing
- Write tests alongside code, especially for probability math

## Deploying
- The static site in site/ is built with `python scripts/build_site.py`,
  which writes site/data/*.json
- The generated JSON is committed (not gitignored) so GitHub Pages can serve it
- Pushing to main deploys site/ via .github/workflows/deploy.yml; the
  workflow runs no Python, so rebuild and commit the JSON before pushing
