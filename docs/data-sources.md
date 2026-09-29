# Data Sources

## 2026-09-29 — Expected goals (xG): investigated, not implemented

### What xG is and why it was the most promising remaining improvement

Expected goals (xG) is the sum, over a team's shots in a match, of each
shot's estimated probability of being scored, given its location, type
and situation. A match's team xG measures the quality of chances created,
not the goals that resulted from them.

The model currently learns each team's attack and defence only from
goals scored and conceded. Every other change tried so far (time
weighting, Dixon-Coles, ridge) altered the model's functional form or
fitting; xG would change the information it learns from. Per match, xG
is a less noisy measure of team strength than goals, because it does
not depend on whether a handful of chances happened to go in.

It would likely have reduced the promoted-team failure described in
results.md ("Final evaluation, frozen test window" and "Ridge
regularisation"). That failure arises when a team with 1-4 training
matches has all-zero goals scored or conceded, and the GLM drives its
coefficient towards −∞. Team xG is continuous and practically never
zero for a full match, so a team that scored [0, 0, 0, 0] would still
show the chances it created. Ridge (alpha = 1e-5) caps the failure but
does not remove it: Coventry still gets P(win) = 0.7%.

### Why it was not implemented

No source was found that provides Premier League per-match team xG for
2021-22 onwards under terms that permit reuse or automated access.

- **FBref** removed xG and other advanced data site-wide, including
  historical archives, on 2026-01-20, after its data provider (Opta)
  ended the agreement, citing a violation.
- **Understat**'s robots.txt, fetched 2026-09-29 (26 bytes,
  `Last-Modified: Mon, 13 Jul 2020`), is in full:

  ```
  User-agent: *
  Disallow: /
  ```

  This covers every path the `soccerdata` package (v1.9.1) requests:
  `/` (session cookies), `/getStatData`, `/getLeagueData/EPL/{season}`
  and `/getMatchData/{id}`. soccerdata never fetches or checks
  robots.txt. Its Understat requests go directly through the session,
  with no delay between them (`rate_limit` defaults to 0 and is not set
  for Understat). No terms-of-use page was found: `/terms`,
  `/terms-of-use`, `/terms-of-service`, `/terms-and-conditions`, `/tos`,
  `/about`, `/privacy`, `/privacy-policy`, `/legal`, `/contact`,
  `/rules`, `/disclaimer` and `/faq` all return 404. The only contact
  given is support@understat.com.
- **Third-party datasets** re-host data from these same two sources, or
  do not say where their xG comes from. Using them would route around
  the restrictions above rather than resolve them. A licence label added
  by an uploader cannot grant rights in data the uploader did not own.
- **StatsBomb open data** covers the Premier League for 2003-04 and
  2015-16 only.
- **Paid APIs** that license xG for automated access exist, but are out
  of scope for this project.

### Candidates checked

| Candidate | Licence | Coverage | Per-match team xG | Access | Why rejected |
|---|---|---|---|---|---|
| Understat | None published; robots.txt `Disallow: /` | 2014-15 to present, incl. 2026-27 in progress | Yes | Website JSON endpoints (e.g. via soccerdata) | robots.txt disallows all automated access |
| FBref | Sports Reference terms; xG was Opta-licensed | xG removed 2026-01-20, including archives | Previously | Website | Data no longer published; provider licence ended |
| StatsBomb open data | StatsBomb Public Data User Agreement (8 Sep 2023) | Premier League 2003-04 and 2015-16 only | Derivable from shot events | GitHub | No coverage of 2021-22 onwards |
| Kaggle: codytipton/understat-data | "Unknown" | 2014-15 to 2024-25 | Yes | Kaggle download | A scrape of Understat |
| Kaggle: armin2080/premier-league-matches-dataset-2021-to-2025 | MIT (uploader's label) | 2021 to 2025; no 2025-26 | Yes | Kaggle download | FBref match-log layout (`gf`, `ga`, `xg`, `poss`, `formation`), so Opta data via FBref |
| Kaggle: furkanark/premier-league-2024-2025-data | CC BY-SA 4.0 (uploader's label) | 2024-25 only | Yes | Kaggle download | Described as "scraped from FBref"; one season |
| GitHub/Zenodo: RezaGooner/english-premier-league-match-dataset | CC BY-NC 4.0 (compiler's label) | xG from 2018-19 (partial), complete to 2025-26; no 2026-27 | Yes, 1-2 decimals | CSV on GitHub | xG provenance not documented (compiled from 11v11, FBref, Sky Sports, premierleague.com, StatMuse, Transfermarkt, ESPN); values differ from Understat |
| Sportmonks, TheStatsAPI | Commercial contract | Current and historical | Yes | API | Paid; out of scope |

The StatsBomb agreement also forbids users to "edit, distort, distribute,
reproduce, sell or in any way provide the data to any external or third
party" (1.2.1) and to "commercially exploit the data or any analysis
derived from the use of the Service" (1.2.2).

None of the free candidates covers the current 2026-27 season except
Understat. Predicting upcoming matches needs xG from matches already
played this season, so a historical-only source would not have been
enough on its own.

### Per-season xG drift

Before the robots.txt decision, the five Understat seasons were fetched
once and joined to OpenFootball in a scratch environment. That fetch was
about 11 requests, and the data has since been deleted. All 1,900
matches joined on (season, home, away), with zero goal mismatches and no
date more than 2 days apart.

| Season | n | Mean xG | Mean goals | Gap | Gap % | SE of gap | Gap / SE | corr(home xG, home goals) |
|---|---|---|---|---|---|---|---|---|
| 2021-22 | 380 | 2.839 | 2.818 | +0.020 | 0.7% | 0.067 | 0.3 | 0.681 |
| 2022-23 | 380 | 2.965 | 2.853 | +0.112 | 3.9% | 0.075 | 1.5 | 0.636 |
| 2023-24 | 380 | 3.381 | 3.279 | +0.102 | 3.1% | 0.076 | 1.3 | 0.593 |
| 2024-25 | 380 | 3.202 | 2.934 | +0.268 | 9.1% | 0.071 | 3.8 | 0.616 |
| 2025-26 | 380 | 3.059 | 2.750 | +0.309 | 11.2% | 0.072 | 4.3 | 0.537 |
| All | 1,900 | 3.089 | 2.927 | +0.162 | 5.5% | | | 0.616 |

Mean xG and goals are per match (both teams). The gap's standard error
comes from the per-match difference (xG − goals).

The gap is within noise for 2021-22 to 2023-24. It then jumps to 3.8 and
4.3 standard errors in 2024-25 and 2025-26, and 2025-26 has the weakest
xG-goals correlation. This suggests the provider recalibrated its xG
model from 2024-25. The data cannot confirm this: two seasons of
below-expected finishing would look the same. If a revision was applied,
it was not back-filled, because the earlier seasons show no gap.

The drift is not specific to Understat. In the RezaGooner dataset, whose
xG comes from a different, undocumented source, mean xG / mean goals
ranges from 0.87 to 0.97 by season across 2021-22 to 2025-26.

**Implication for any future xG work.** Walk-forward trains on older
seasons and predicts newer ones, so the model would learn an xG-goals
relationship from one era and apply it to another. xG would need
per-season scaling, e.g. each season's xG multiplied by that season's
mean goals / mean xG, computed only from matches completed before the
prediction date. For the season in progress, shrink the ratio towards
the prior season's while few matches have been played.

A single pooled scale factor applied to every row would measure nothing.
If xG enters the model on the log scale, the factor adds a constant that
the intercept absorbs; if it enters linearly, its coefficient rescales
to compensate. Either way predictions are unchanged, and a validation
comparison would show no effect for that reason alone.

### What would unblock this

- Written permission from Understat (support@understat.com) for
  automated access
- A licensed provider (a paid API), if the project's scope changes
- A source that publishes Premier League match xG explicitly for reuse,
  covering 2021-22 onwards and the current season

### Sources

- Understat robots.txt: https://understat.com/robots.txt
- StatsBomb open data and user agreement: https://github.com/statsbomb/open-data
- FBref removal of advanced data: https://awfulannouncing.com/soccer/sports-reference-pulls-advanced-data-agreement-violation-dispute.html
- Kaggle: https://www.kaggle.com/datasets/codytipton/understat-data,
  https://www.kaggle.com/datasets/armin2080/premier-league-matches-dataset-2021-to-2025,
  https://www.kaggle.com/datasets/furkanark/premier-league-2024-2025-data
- RezaGooner dataset: https://github.com/RezaGooner/english-premier-league-match-dataset
- Sportmonks: https://www.sportmonks.com/football-api/premier-league-api/xg-expected-goals/
- TheStatsAPI: https://www.thestatsapi.com/football/xg
