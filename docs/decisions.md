# Decisions & Problems Log

## 2026-09-19 — Session 1: Project setup

### What was built
Layered Python package (`src/footy` with `data`, `model`, `web`),
football-data.org API client with explicit error handling, pytest
configured, dependencies pinned, secrets in `.env`.

### Problems encountered

**Interrupted venv creation left a broken environment.**
`python3 -m venv` was cancelled during the pip bootstrap step, leaving a
`.venv` with no `activate` script. Fix: `rm -rf .venv` and recreate,
letting it finish. Lesson: that step is silent for ~30s; silence is not
a hang.

**Placeholder values pasted literally.**
Both `FOOTBALL_DATA_TOKEN=your_token_here` and
`ANTHROPIC_API_KEY=sk-ant-...` were saved as-is. The first caused a
`400 invalid token` from the API; the second caused Claude Code to send
a bogus key and 401 on every request, despite login succeeding.
Diagnosis trick: check the length, not the value —
`python -c "...print(len(t))"` returned 15, exactly
`len("your_token_here")`.
Lesson: an env var that exists is not an env var that is correct.

**`ANTHROPIC_API_KEY` took precedence over subscription auth.**
Claude Code prefers the env var over an interactive login, so
`/login` reported success and the next call still 401'd. Also survived
in already-open shells after the dotfile was edited — `unset` in the
live shell was needed, not just `source`.

**Editable install did not register.**
`pip install -e .` succeeded and wrote a correct `.pth` file pointing at
`src/`, but `sys.path` never picked it up, so `import footy` failed
intermittently. `pip install -e . --config-settings editable_mode=compat`
worked where the default did not. Root cause never fully identified.
Resolution: scripts bootstrap `sys.path` explicitly; pytest uses
`pythonpath = ["src"]` in `pyproject.toml`. Both reliable.

### Decisions
- Canonical names follow the API's `shortName` style, since they are
  what the UI will display.
- `_get()` raises on non-200 rather than returning empty data, so
  failures are loud.

---

## 2026-09-26 — Session 2: Historical data

### What was built
OpenFootball loader with local caching, team name normalization across
two sources, 1,900 matches across five complete seasons (2021-22 to
2025-26), 8 passing tests.

### Problems encountered

**Inconsistent JSON schema within one source.**
OpenFootball encodes scores as `{"ht": [..], "ft": [h, a]}` in most
seasons, but 27 of 380 matches in 2025-26 use a bare list `[h, a]`.
Inspecting the first record of each season showed nothing wrong — the
variant was buried mid-file. Found by counting `type(score)` across all
1,900 records.
Resolution: `_full_time_score()` handles both shapes explicitly and
**raises** on any third shape rather than skipping the match. Silently
dropping records would corrupt the model invisibly.

**Team names disagree across sources, and within one source.**
OpenFootball mostly suffixes " FC" but uses "AFC Bournemouth" (prefix)
and "Sunderland AFC" (different suffix), so no mechanical rule works.
The API says "Nottingham", "Brighton Hove", "Man United" where
OpenFootball says "Nottingham Forest FC", "Brighton & Hove Albion FC",
"Manchester United FC".
Resolution: two explicit dicts, both mapping into one canonical name,
never source-to-source. Unknown names raise `UnknownTeamError` naming
the file to edit.

**Test file overwritten instead of appended.**
Four score-parsing tests were silently lost when new tests were added.
Caught by noticing the collected count dropped from 8 to 4.
Lesson: read `git diff` before committing; the count in pytest output
is a cheap canary.

### Decisions
- Raw source names kept as `home_raw` / `away_raw` alongside canonical
  `home` / `away`, so debugging never requires re-fetching.
- Seasons cached to `data/` after first download — re-runs are offline
  and the source isn't hammered.
- A test asserts every historical team name maps to a known canonical
  name; another asserts no two clubs collapse into the same canonical
  name.

### Open questions
- Dedup key for combining history and live API data. `(date, home, away)`
  is fragile because postponed matches change date.
- Whether the current season comes from the API or from OpenFootball
  once it publishes.