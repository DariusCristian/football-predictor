/* Premier League probabilities: static site.
 *
 * Displays numbers published by scripts/build_site.py; it never computes
 * probabilities. Rounding to whole numbers out of 100 and laying out
 * axes are the only arithmetic here.
 */
"use strict";

const DATA_FILES = ["predictions", "matrix", "teams", "history", "scoreboard"];

// Which files each section cannot work without.
const SECTIONS = {
  fixtures: { label: "Fixtures", needs: ["predictions", "matrix", "teams"] },
  explorer: { label: "Head-to-head", needs: ["matrix", "teams"] },
  teams: { label: "Teams", needs: ["teams"] },
  history: { label: "Team history", needs: ["teams", "history"] },
  record: { label: "Record", needs: ["scoreboard"] },
  method: { label: "Method", needs: [] },
};

// Shape checks: a file that parses but lacks its main field is a failure too.
const REQUIRED_KEY = {
  predictions: "predictions",
  matrix: "pairings",
  teams: "teams",
  history: "matches",
  scoreboard: "backtest",
};

const SEASON_LENGTH = 38;
const TZ = "Europe/London";
const KINDS = ["home", "draw", "away"];
const NUMBER_WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six",
  "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"];

/* ---------- Small DOM helpers ---------- */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  setAttrs(node, attrs);
  node.append(...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function setAttrs(node, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
}

const num = (text, cls = "") => el("span", { class: `num ${cls}`.trim(), text });
const hidden = (text) => el("span", { class: "visually-hidden", text });

/* ---------- Formatting ---------- */

// Whole numbers out of 100 that still add to 100 (largest remainder).
function outOf100(probs) {
  const exact = probs.map((p) => p * 100);
  const units = exact.map(Math.floor);
  const shortfall = 100 - units.reduce((a, b) => a + b, 0);
  exact
    .map((x, i) => [x - units[i], i])
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .slice(0, shortfall)
    .forEach(([, i]) => { units[i] += 1; });
  return units;
}

const countLabel = (count, p) => (count === 0 && p > 0 ? "<1" : String(count));
const per100 = (p) => { const n = Math.round(p * 100); return n === 0 && p > 0 ? "<1" : String(n); };
const times = (x) => x.toFixed(2);

function fmtDate(iso, opts) {
  return new Intl.DateTimeFormat("en-GB", { timeZone: TZ, ...opts }).format(new Date(iso));
}
const fmtKickoff = (iso) =>
  `${fmtDate(iso, { weekday: "short", day: "numeric", month: "short" })} · ${fmtDate(iso, { hour: "2-digit", minute: "2-digit", hour12: false })}`;
const fmtStamp = (iso) =>
  `${fmtDate(iso, { day: "numeric", month: "long", year: "numeric" })} at ${fmtDate(iso, { hour: "2-digit", minute: "2-digit", hour12: false })} UK time`;
const fmtDay = (iso) => fmtDate(iso, { day: "numeric", month: "long", year: "numeric" });
const fmtShortDate = (ymd) =>
  new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })
    .format(new Date(`${ymd}T00:00:00Z`));

function fmtRange(isoA, isoB) {
  const part = (iso, o) => fmtDate(iso, o);
  const [dA, mA, yA] = [part(isoA, { day: "numeric" }), part(isoA, { month: "long" }), part(isoA, { year: "numeric" })];
  const [dB, mB, yB] = [part(isoB, { day: "numeric" }), part(isoB, { month: "long" }), part(isoB, { year: "numeric" })];
  if (dA === dB && mA === mB && yA === yB) return `${dA} ${mA} ${yA}`;
  if (mA === mB && yA === yB) return `${dA}–${dB} ${mA} ${yA}`;
  if (yA === yB) return `${dA} ${mA} – ${dB} ${mB} ${yA}`;
  return `${dA} ${mA} ${yA} – ${dB} ${mB} ${yB}`;
}

const countWord = (n) => (n < NUMBER_WORDS.length ? NUMBER_WORDS[n] : String(n));
const plural = (n, one, many) => (n === 1 ? one : many);

/* ---------- Loading ---------- */

async function loadFile(name) {
  const path = `data/${name}.json`;
  let response;
  try {
    response = await fetch(path, { cache: "no-cache" });
  } catch (err) {
    if (location.protocol === "file:") {
      return { ok: false, reason: "the page was opened straight from disk (file://), and browsers block loading data files that way" };
    }
    return { ok: false, reason: `the request failed before any response arrived (${err.message})` };
  }
  if (!response.ok) {
    return { ok: false, reason: `the server answered HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ""}` };
  }
  let data;
  try {
    data = await response.json();
  } catch {
    return { ok: false, reason: "the file arrived but isn't valid JSON" };
  }
  if (!data || !(REQUIRED_KEY[name] in data)) {
    return { ok: false, reason: `the file is missing its "${REQUIRED_KEY[name]}" field` };
  }
  return { ok: true, data };
}

async function loadAll() {
  const results = await Promise.all(DATA_FILES.map(loadFile));
  return Object.fromEntries(DATA_FILES.map((name, i) => [name, results[i]]));
}

function missingFor(loaded, needs) {
  return needs.filter((name) => !loaded[name].ok);
}

function sectionError(container, loaded, missing, what) {
  const items = missing.map((name) =>
    el("li", {}, el("code", { text: `data/${name}.json` }), `: ${loaded[name].reason}.`));
  container.replaceChildren(el("div", { class: "section-error", role: "note" },
    el("p", {}, `${what} needs data that couldn't be loaded:`),
    el("ul", {}, items)));
}

function renderLoadBanner(loaded) {
  const failed = DATA_FILES.filter((n) => !loaded[n].ok);
  const banner = document.getElementById("load-errors");
  if (!failed.length) return;
  const working = Object.values(SECTIONS)
    .filter((s) => !s.needs.some((n) => failed.includes(n)))
    .map((s) => s.label);
  const parts = [
    el("p", {}, el("strong", { text: failed.length === DATA_FILES.length
      ? "None of the five data files loaded."
      : `${countWord(failed.length)} of the five data files didn't load.` })),
    el("ul", {}, failed.map((n) => el("li", {}, el("code", { text: `data/${n}.json` }), `: ${loaded[n].reason}.`))),
  ];
  if (location.protocol === "file:") {
    parts.push(el("p", {}, "To fix this, serve the folder and open it over http: run ",
      el("code", { text: "python3 -m http.server -d site" }),
      " from the project root, then visit ", el("code", { text: "http://localhost:8000" }), "."));
  }
  parts.push(el("p", {}, working.length
    ? `Still working: ${working.join(", ")}.`
    : "No section can be shown without them."));
  banner.replaceChildren(...parts);
  banner.hidden = false;
}

/* ---------- Shared context built from the data ---------- */

function buildContext(loaded) {
  const ctx = {};
  if (loaded.teams.ok) {
    const teams = loaded.teams.data.teams;
    ctx.teams = teams;
    ctx.team = new Map(teams.map((t) => [t.name, t]));
    const n = teams.length;
    const attackOrder = [...teams].sort((a, b) => b.attack_centred - a.attack_centred).map((t) => t.name);
    const defenceOrder = [...teams].sort((a, b) => a.defence_centred - b.defence_centred).map((t) => t.name);
    const ordinal = ["", "second-", "third-"];
    ctx.standing = (name) => {
      const out = [];
      const a = attackOrder.indexOf(name);
      const d = defenceOrder.indexOf(name);
      if (a > -1 && a < 3) out.push({ kind: "attack", text: `${ordinal[a]}strongest attack` });
      if (a > -1 && a >= n - 3) out.push({ kind: "attack", text: `${ordinal[n - 1 - a]}weakest attack` });
      if (d > -1 && d < 3) out.push({ kind: "defence", text: `${ordinal[d]}best defence` });
      if (d > -1 && d >= n - 3) out.push({ kind: "defence", text: `${ordinal[n - 1 - d]}worst defence` });
      return out;
    };
  }
  if (loaded.matrix.ok) {
    ctx.pairing = new Map(loaded.matrix.data.pairings.map((p) => [`${p.home}|${p.away}`, p]));
  }
  return ctx;
}

const isThin = (level) => level === "low" || level === "medium";

function thinTeams(ctx, ...names) {
  return names.map((n) => ctx.team.get(n)).filter((t) => t && isThin(t.confidence));
}

/* A sentence about why a team's numbers are fragile, built from the data
 * so it stays true when the data is regenerated. */
function evidenceNote(ctx, team) {
  const n = team.matches_in_training;
  let text = `${team.name}: ${n} Premier League ${plural(n, "match", "matches")} in the training data`;
  text += team.season_matches === n && n > 0 ? ", all from this season." : ".";
  const standing = ctx.standing(team.name);
  if (standing.length) {
    const bits = standing.map((s) => s.kind === "attack"
      ? `the league's ${s.text} (${times(team.attack_multiplier)} times the league-average goals scored)`
      : `the league's ${s.text} (${times(team.defence_multiplier)} times the league-average goals conceded)`);
    text += ` The model rates them ${bits.join(" and ")}.`;
    if (team.season_matches > 0 && "goals_for_per_game" in team) {
      text += ` This season they have scored ${team.goals_for_per_game.toFixed(1)} and conceded ${team.goals_against_per_game.toFixed(1)} a game.`;
    }
  }
  text += team.confidence === "low"
    ? " Treat predictions involving them as rough guesses."
    : " Read predictions involving them with some caution.";
  return text;
}

function noteBlock(ctx, teams) {
  return teams.map((t) => el("p", { class: "note" },
    el("span", { class: "dagger", "aria-hidden": "true", text: "†" }), evidenceNote(ctx, t)));
}

/* ---------- Visual pieces ---------- */

function teamLabel(ctx, name) {
  const t = ctx.team && ctx.team.get(name);
  const thin = t && isThin(t.confidence);
  return [
    name,
    thin ? el("span", { class: "dagger", "aria-hidden": "true", text: "†" }) : null,
    thin ? hidden(` (${t.confidence} confidence: ${t.matches_in_training} matches in the data)`) : null,
  ];
}

// One block per full season before this one, one tick per match since.
function gauge(team) {
  const n = team.matches_in_training;
  const current = team.season_matches;
  const prior = Math.max(0, n - current);
  const seasons = Math.floor(prior / SEASON_LENGTH);
  const ticks = n - seasons * SEASON_LENGTH;
  const label = seasons
    ? `${n} Premier League matches in the training data: ${seasons} full ${plural(seasons, "season", "seasons")} and ${ticks} more`
    : `${n} Premier League ${plural(n, "match", "matches")} in the training data`;
  const g = el("span", { class: "gauge", "aria-hidden": "true" });
  for (let i = 0; i < seasons; i++) g.append(el("span", { class: "g-season" }));
  for (let i = 0; i < ticks; i++) g.append(el("span", { class: "g-match" }));
  return el("span", { class: "gauge-wrap", title: label },
    g, el("span", { class: "g-n num", "aria-hidden": "true", text: `${n} ${plural(n, "match", "matches")}` }),
    hidden(label));
}

/* 100 ticks, grouped in tens: if this match were played 100 times.
 * Thin evidence keeps the same ticks but masks them with hatching. */
function strip(probs, confidence, home, away) {
  const p = [probs.home, probs.draw, probs.away];
  const counts = outOf100(p);
  const labels = counts.map((c, i) => countLabel(c, p[i]));
  const conf = isThin(confidence) ? confidence : "high";
  const aria = `If played 100 times: ${home} win ${labels[0]}, draw ${labels[1]}, ${away} win ${labels[2]}`
    + (conf === "high" ? "." : `. ${conf[0].toUpperCase()}${conf.slice(1)} confidence: thin evidence.`);

  const svg = svgEl("svg", { viewBox: "0 0 1000 40", class: `strip c-${conf}`, role: "img", "aria-label": aria, preserveAspectRatio: "xMinYMid meet" });
  const solid = svgEl("g", conf === "high" ? {} : { mask: `url(#mask-${conf})` });
  const outline = conf === "high" ? null : svgEl("g");
  let i = 0;
  counts.forEach((count, k) => {
    for (let j = 0; j < count; j++, i++) {
      // 99 periods + 9 group gaps + one tick width = 1000.
      const x = i * 9.79 + Math.floor(i / 10) * 2.6;
      solid.append(svgEl("rect", { x, y: 0, width: 6.4, height: 40, class: `t-${KINDS[k]}` }));
      if (outline) outline.append(svgEl("rect", { x: x + 0.55, y: 0.55, width: 5.3, height: 38.9, class: `o-${KINDS[k]}` }));
    }
  });
  svg.append(solid);
  if (outline) svg.append(outline);

  // Draw label sits over the middle of its segment, kept inside the strip.
  const drawCentre = Math.min(84, Math.max(16, counts[0] + counts[1] / 2));
  const nums = el("div", { class: "nums", "aria-hidden": "true" },
    el("span", { class: "n-home", text: labels[0] }),
    el("span", { class: "n-draw", style: `left:${drawCentre}%`, text: labels[1] }),
    el("span", { class: "n-away", text: labels[2] }));
  return [svg, nums];
}

function confidenceText(level) {
  return isThin(level) ? el("span", { class: "conf", text: ` · ${level} confidence` }) : null;
}

/* ---------- Fixtures (the top of the page) ---------- */

function renderFixtures(loaded, ctx) {
  const list = document.getElementById("fixture-list");
  const intro = document.getElementById("intro");
  const missing = missingFor(loaded, SECTIONS.fixtures.needs);
  if (missing.length) {
    intro.replaceChildren();
    sectionError(list, loaded, missing, "This round's fixtures");
    return;
  }
  const preds = [...loaded.predictions.data.predictions]
    .sort((a, b) => a.kickoff.localeCompare(b.kickoff) || a.home.localeCompare(b.home));

  const headline = document.getElementById("fixtures-h");
  if (!preds.length) {
    headline.textContent = "Next round";
    intro.replaceChildren(el("p", {}, "No upcoming predictions have been recorded yet. They are written before kickoff, usually early in the week of the round."));
    return;
  }

  headline.textContent = `Next round: ${fmtRange(preds[0].kickoff, preds[preds.length - 1].kickoff)}`;
  const stamps = [...new Set(preds.map((p) => p.predicted_at))].sort();
  const recorded = stamps.length === 1
    ? `recorded on ${fmtStamp(stamps[0])}`
    : `recorded between ${fmtStamp(stamps[0])} and ${fmtStamp(stamps[stamps.length - 1])}`;
  const allBefore = preds.every((p) => p.predicted_at < p.kickoff);

  const introParts = [
    el("p", {},
      el("strong", { text: `${countWord(preds.length)} ${plural(preds.length, "prediction", "predictions")}, ${recorded}` }),
      allBefore ? ", before any of these matches kicked off." : ".",
      " Read each number as times in 100: if the match were played 100 times, how often each result would happen."),
  ];
  if (loaded.scoreboard.ok) {
    const bt = loaded.scoreboard.data.backtest;
    const model = bt.models.find((m) => m.model === "poisson");
    if (model && "improvement_over_league_average" in bt) {
      introParts.push(el("p", {},
        `Tested on ${model.n} past matches it had never seen, the model scored ${(bt.improvement_over_league_average * 100).toFixed(1)}% better than assuming every match is a league-average one (by log loss, where lower is better). That is a small edge; football is mostly noise. `,
        el("a", { href: "#record", text: "How this was tested" }), "."));
    }
  }
  if (preds.every((p) => new Date(p.kickoff) < new Date())) {
    introParts.push(el("p", {}, el("em", { text: "All of these matches have now kicked off; the next round's predictions haven't been published yet." })));
  }
  intro.replaceChildren(...introParts);

  const key = document.getElementById("key");
  key.replaceChildren(
    el("span", {}, el("span", { class: "swatch swatch-home", "aria-hidden": "true" }), "home win"),
    el("span", {}, el("span", { class: "swatch swatch-draw", "aria-hidden": "true" }), "draw"),
    el("span", {}, el("span", { class: "swatch swatch-away", "aria-hidden": "true" }), "away win"),
    el("span", {}, el("span", { class: "swatch swatch-hatch", "aria-hidden": "true" }), "† hatched: a team with few matches in the data"),
    el("span", {}, el("span", { class: "g-season", "aria-hidden": "true" }), "one full season of matches,",
      el("span", { class: "g-match", "aria-hidden": "true" }), "one match"),
  );
  key.hidden = false;

  list.replaceChildren(...preds.map((pred) => {
    const pairing = ctx.pairing.get(`${pred.home}|${pred.away}`);
    const level = pairing ? pairing.confidence : "low";
    const homeT = ctx.team.get(pred.home);
    const awayT = ctx.team.get(pred.away);
    const thin = thinTeams(ctx, pred.home, pred.away);
    const top = pairing && pairing.top_scores && pairing.top_scores[0];
    const [svg, nums] = strip(pred.probabilities, level, pred.home, pred.away);

    const main = el("div", { class: "col-main" },
      el("p", { class: "kickoff" }, el("time", { datetime: pred.kickoff, text: fmtKickoff(pred.kickoff) }), hidden(" UK time")),
      el("h3", { class: "pair" },
        el("span", { class: "home-name" }, teamLabel(ctx, pred.home)),
        hidden(" v "),
        el("span", { class: "away-name" }, teamLabel(ctx, pred.away))),
      el("div", { class: "gauges" },
        homeT ? el("div", { class: "side-home gauge-wrap" }, gauge(homeT)) : el("span"),
        awayT ? el("div", { class: "side-away gauge-wrap" }, gauge(awayT)) : el("span")),
      svg, nums,
      el("p", { class: "score-line" },
        `Likeliest score ${pred.most_likely_score[0]}–${pred.most_likely_score[1]}`,
        top && top.home === pred.most_likely_score[0] && top.away === pred.most_likely_score[1]
          ? [": ", num(per100(top.p)), " in 100"] : null,
        confidenceText(level),
        ". ",
        el("a", { href: "#explorer", "data-home": pred.home, "data-away": pred.away, class: "pair-link",
          text: "Scorelines and past meetings" }),
        hidden(` for ${pred.home} v ${pred.away}`)));

    return el("article", { class: `fixture layout c-${level}` },
      main, el("aside", { class: "col-note" }, noteBlock(ctx, thin)));
  }));

  list.addEventListener("click", (e) => {
    const link = e.target.closest(".pair-link");
    if (link && window.selectPair) window.selectPair(link.dataset.home, link.dataset.away);
  });
}

/* ---------- Head-to-head explorer ---------- */

function renderExplorer(loaded, ctx) {
  const out = document.getElementById("pair-out");
  const missing = missingFor(loaded, SECTIONS.explorer.needs);
  if (missing.length) { sectionError(out, loaded, missing, "The head-to-head explorer"); return; }

  const form = document.getElementById("pair-form");
  const homeSel = document.getElementById("pick-home");
  const awaySel = document.getElementById("pick-away");
  const names = ctx.teams.map((t) => t.name).sort();
  const option = (n) => el("option", { value: n }, n, isThin(ctx.team.get(n).confidence) ? " †" : "");
  homeSel.replaceChildren(...names.map(option));
  awaySel.replaceChildren(...names.map(option));

  const first = loaded.predictions.ok && loaded.predictions.data.predictions[0];
  homeSel.value = first ? first.home : names[0];
  awaySel.value = first ? first.away : names[1];
  form.hidden = false;

  const draw = () => {
    const home = homeSel.value;
    const away = awaySel.value;
    if (home === away) {
      out.replaceChildren(el("div", { class: "layout" }, el("div", { class: "col-main" },
        el("p", {}, "Pick two different teams."))));
      return;
    }
    out.replaceChildren(renderPair(loaded, ctx, home, away));
  };
  homeSel.addEventListener("change", draw);
  awaySel.addEventListener("change", draw);
  document.getElementById("swap").addEventListener("click", () => {
    [homeSel.value, awaySel.value] = [awaySel.value, homeSel.value];
    draw();
  });
  window.selectPair = (home, away) => { homeSel.value = home; awaySel.value = away; draw(); };
  draw();
}

function renderPair(loaded, ctx, home, away) {
  const pairing = ctx.pairing.get(`${home}|${away}`);
  if (!pairing) {
    return el("div", { class: "section-error" }, el("p", { text: `matrix.json has no entry for ${home} v ${away}.` }));
  }
  const level = pairing.confidence;
  const [svg, nums] = strip(pairing.probabilities, level, home, away);
  const top = pairing.top_scores || [];

  const main = el("div", { class: "col-main big" },
    el("h3", { class: "pair" },
      el("span", { class: "home-name" }, teamLabel(ctx, home)),
      hidden(" at home to "),
      el("span", { class: "away-name" }, teamLabel(ctx, away))),
    el("div", { class: "gauges" },
      el("div", { class: "side-home gauge-wrap" }, gauge(ctx.team.get(home))),
      el("div", { class: "side-away gauge-wrap" }, gauge(ctx.team.get(away)))),
    svg, nums,
    isThin(level) ? el("p", { class: "score-line" }, el("span", { class: "conf", text: `${level[0].toUpperCase()}${level.slice(1)} confidence` }), ": see the note on thin evidence.") : null,
    el("p", { class: "xg" }, "On average the model expects ",
      `${home} to score `, num(pairing.expected_goals.home.toFixed(2)),
      ` and ${away} `, num(pairing.expected_goals.away.toFixed(2)), "."),
    top.length ? scoreGrid(top, home, away, level) : null,
    meetings(loaded, home, away));

  return el("div", { class: "layout" }, main,
    el("aside", { class: "col-note" }, noteBlock(ctx, thinTeams(ctx, home, away))));
}

function scoreGrid(top, home, away, level) {
  const maxGoals = Math.max(5, ...top.map((s) => Math.max(s.home, s.away)));
  const byCell = new Map(top.map((s) => [`${s.home}-${s.away}`, s]));
  const pMax = Math.max(...top.map((s) => s.p));

  const head = el("tr", {}, el("th", { scope: "col" }, hidden(`${home} goals down, ${away} goals across`)));
  for (let j = 0; j <= maxGoals; j++) head.append(el("th", { scope: "col", text: String(j) }));

  const body = [];
  for (let i = 0; i <= maxGoals; i++) {
    const row = el("tr", {}, el("th", { scope: "row", text: String(i) }));
    for (let j = 0; j <= maxGoals; j++) {
      const s = byCell.get(`${i}-${j}`);
      const kind = i > j ? "home" : i === j ? "draw" : "away";
      const drawCls = i === j ? " is-draw" : "";
      if (!s) {
        row.append(el("td", { class: `rest${drawCls}` }, hidden(`${i}–${j}: not among the ten likeliest`)));
        continue;
      }
      row.append(el("td", { class: `top${drawCls}` },
        el("span", { class: "cell-n", "aria-hidden": "true", text: per100(s.p) }),
        hidden(`${i}–${j}: ${per100(s.p)} in 100`),
        el("span", { class: `fill fill-${kind}`, style: `height:${Math.max(4, (s.p / pMax) * 46)}%`, "aria-hidden": "true" })));
    }
    body.push(row);
  }

  const best = top[0];
  return el("div", {},
    el("h4", { class: "visually-hidden", text: "Scorelines" }),
    el("p", {}, `Likeliest score ${best.home}–${best.away}: `, num(per100(best.p)), " in 100. Every scoreline below is one the model thinks could easily happen; no single one is likely."),
    el("div", { class: `grid-wrap${isThin(level) ? " thin" : ""}` },
      el("div", { class: "axis-top", "aria-hidden": "true", text: `${away} goals` }),
      el("div", { class: "axis-side", "aria-hidden": "true", text: `${home} goals` }),
      el("table", { class: "score-grid" },
        el("caption", {}, `The ${top.length} likeliest scorelines, in matches per 100. Green: ${home} win; grey: draw; raspberry: ${away} win. Blank cells are less likely than the ${top.length === 10 ? "tenth" : "last shown"}.`),
        el("thead", {}, head), el("tbody", {}, body))));
}

function meetings(loaded, home, away) {
  const wrap = el("div", { class: "meetings" }, el("h4", { text: `Past meetings since 2021-22` }));
  if (!loaded.history.ok) {
    wrap.append(el("div", { class: "section-error" }, el("p", {},
      "Past meetings need ", el("code", { text: "data/history.json" }), `, which couldn't be loaded: ${loaded.history.reason}.`)));
    return wrap;
  }
  const games = loaded.history.data.matches
    .filter((m) => (m.home === home && m.away === away) || (m.home === away && m.away === home))
    .sort((a, b) => b.date.localeCompare(a.date));
  if (!games.length) {
    wrap.append(el("p", { text: `No Premier League meetings between ${home} and ${away} in the data, which starts in August 2021.` }));
    return wrap;
  }
  let wins = 0; let draws = 0; let losses = 0;
  for (const m of games) {
    const hf = m.home === home ? m.home_goals : m.away_goals;
    const af = m.home === home ? m.away_goals : m.home_goals;
    if (hf > af) wins++; else if (hf === af) draws++; else losses++;
  }
  wrap.append(el("p", {},
    `${games.length} ${plural(games.length, "meeting", "meetings")}: ${home} won ${wins}, ${draws} ${plural(draws, "draw", "draws")}, ${away} won ${losses}. `
    + "Venue matters, so check who was at home."));
  wrap.append(el("table", { class: "data" },
    el("thead", {}, el("tr", {},
      el("th", { scope: "col", text: "Date" }),
      el("th", { scope: "col", text: "Home" }),
      el("th", { scope: "col", text: "Score" }),
      el("th", { scope: "col", text: "Away" }))),
    el("tbody", {}, games.map((m) => el("tr", {},
      el("td", { class: "num", text: fmtShortDate(m.date) }),
      el("td", { text: m.home }),
      el("td", { class: "num", text: `${m.home_goals}–${m.away_goals}` }),
      el("td", { text: m.away }))))));
  return wrap;
}

/* ---------- Team table ---------- */

const SORTS = {
  name: { label: "Team", cmp: (a, b) => a.name.localeCompare(b.name), dir: "A to Z" },
  evidence: { label: "Matches", cmp: (a, b) => b.matches_in_training - a.matches_in_training || a.name.localeCompare(b.name), dir: "most first" },
  attack: { label: "Attack", cmp: (a, b) => b.attack_centred - a.attack_centred, dir: "strongest first" },
  defence: { label: "Defence", cmp: (a, b) => a.defence_centred - b.defence_centred, dir: "best first" },
};

function renderTeams(loaded, ctx) {
  const out = document.getElementById("team-table");
  const missing = missingFor(loaded, SECTIONS.teams.needs);
  if (missing.length) { sectionError(out, loaded, missing, "The team table"); return; }

  const teams = ctx.teams;
  // Shared log axis for attack and defence: centred coefficients are logs,
  // so equal distances are equal ratios. Ticks are labelled as multipliers.
  const values = teams.flatMap((t) => [t.attack_centred, t.defence_centred]);
  const lo = Math.min(Math.log(0.25), ...values) - 0.08;
  const hi = Math.max(Math.log(2), ...values) + 0.08;
  const pos = (v) => ((v - lo) / (hi - lo)) * 100;
  const ticks = [0.25, 0.5, 1, 2];

  let sortKey = "attack";

  const track = (value, mult, thin, what) => el("div", { class: "rating" },
    el("div", { class: "track", "aria-hidden": "true" },
      el("span", { class: "baseline" }),
      ticks.filter((t) => t !== 1).map((t) => el("span", { class: "tick", style: `left:${pos(Math.log(t))}%` })),
      el("span", { class: "zero", style: `left:${pos(0)}%` }),
      el("span", { class: `dot${thin ? " hollow" : ""}`, style: `left:${pos(value)}%` })),
    el("span", { class: "num", text: `×${times(mult)}` }),
    hidden(` ${what}`));

  const axis = () => el("div", { class: "axis-scale", "aria-hidden": "true" },
    ticks.filter((t) => t >= 0.5).map((t) => el("span", { style: `left:${pos(Math.log(t))}%`, text: `×${t}` })));

  const sortButton = (key) => el("button", {
    type: "button", class: "sort-button", "aria-pressed": String(sortKey === key),
    onclick: () => { sortKey = key; draw(); },
  }, SORTS[key].label, sortKey === key ? el("span", { class: "sort-dir", text: ` (${SORTS[key].dir})` }) : null);

  // Numeric order: attack and matches run high to low; defence low to high.
  const ARIA_SORT = { name: "ascending", evidence: "descending", attack: "descending", defence: "ascending" };
  const th = (key, extra) => el("th", { scope: "col", "aria-sort": sortKey === key ? ARIA_SORT[key] : null },
    sortButton(key), extra);

  const draw = () => {
    const rows = [...teams].sort(SORTS[sortKey].cmp);
    const thin = rows.filter((t) => isThin(t.confidence));
    const table = el("table", { class: "data team-table" },
      el("caption", { class: "visually-hidden", text: `Team ratings, sorted by ${SORTS[sortKey].label.toLowerCase()}, ${SORTS[sortKey].dir}` }),
      el("thead", {}, el("tr", {},
        th("name"),
        th("evidence", el("span", { class: "axis-note", text: "in the training data" })),
        th("attack", [el("span", { class: "axis-note", text: "goals scored vs league average" }), axis()]),
        th("defence", [el("span", { class: "axis-note", text: "goals conceded vs league average; fewer is better" }), axis()]),
        el("th", { scope: "col" }, "Form", el("span", { class: "axis-note", text: "oldest to latest" })))),
      el("tbody", {}, rows.map((t) => {
        const isT = isThin(t.confidence);
        return el("tr", {},
          el("td", {}, el("button", { type: "button", class: "team-name-button", onclick: () => window.showTeam && window.showTeam(t.name, true) },
            t.name), isT ? el("span", { class: "dagger", "aria-hidden": "true", text: "†" }) : null,
            isT ? hidden(` (${t.confidence} confidence)`) : null),
          el("td", { "data-label": "Matches" }, gauge(t)),
          el("td", { class: "rating-cell", "data-label": "Attack" }, track(t.attack_centred, t.attack_multiplier, isT, "times league-average goals scored")),
          el("td", { class: "rating-cell", "data-label": "Defence" }, track(t.defence_centred, t.defence_multiplier, isT, "times league-average goals conceded")),
          el("td", { "data-label": "Form" }, el("span", { class: "form", "aria-label": `Last ${t.last_5.length}, oldest to latest: ${t.last_5.join(", ")}`, role: "img" },
            t.last_5.map((r) => el("span", { class: `f-${r}`, text: r })))));
      })));

    const mobileSort = el("div", { class: "mobile-sort pair-form" },
      el("label", {}, "Sort by", el("select", {
        onchange: (e) => { sortKey = e.target.value; draw(); },
      }, Object.entries(SORTS).map(([k, s]) => el("option", { value: k, selected: k === sortKey }, `${s.label} (${s.dir})`)))));

    out.replaceChildren(el("div", { class: "layout" },
      el("div", { class: "col-main" }, mobileSort, table),
      el("aside", { class: "col-note" },
        el("p", { class: "note" }, "Hollow dots mark ratings built on thin evidence. They are shown, not hidden, because the model uses them."),
        noteBlock(ctx, thin))));
  };
  draw();
}

/* ---------- Team history ---------- */

function renderHistory(loaded, ctx) {
  const out = document.getElementById("history-out");
  const missing = missingFor(loaded, SECTIONS.history.needs);
  if (missing.length) { sectionError(out, loaded, missing, "Team history"); return; }

  const form = document.getElementById("history-form");
  const sel = document.getElementById("pick-team");
  const names = ctx.teams.map((t) => t.name).sort();
  sel.replaceChildren(...names.map((n) => el("option", { value: n }, n, isThin(ctx.team.get(n).confidence) ? " †" : "")));
  form.hidden = false;
  const current = loaded.teams.data.current_season;
  const matches = loaded.history.data.matches;

  const draw = () => {
    const name = sel.value;
    const t = ctx.team.get(name);
    const played = matches.filter((m) => m.home === name || m.away === name);
    const bySeason = new Map();
    for (const m of played) {
      if (!bySeason.has(m.season)) bySeason.set(m.season, []);
      bySeason.get(m.season).push(m);
    }
    const seasons = [...bySeason.keys()].sort().reverse();

    const summary = [el("p", {}, gauge(t))];
    if (t.season_matches > 0 && "goals_for_per_game" in t) {
      summary.push(el("p", {}, `${current} so far: ${t.season_matches} played, scoring `,
        num(t.goals_for_per_game.toFixed(1)), " and conceding ", num(t.goals_against_per_game.toFixed(1)), " a game."));
    }
    if (isThin(t.confidence)) {
      summary.push(el("p", { class: "note" }, el("span", { class: "dagger", "aria-hidden": "true", text: "†" }), evidenceNote(ctx, t)));
    }
    if (!seasons.length) summary.push(el("p", { text: `No Premier League matches for ${name} in the data.` }));

    const blocks = seasons.map((season) => {
      const games = bySeason.get(season).sort((a, b) => b.date.localeCompare(a.date));
      let w = 0; let d = 0; let l = 0;
      const rows = games.map((m) => {
        const atHome = m.home === name;
        const gf = atHome ? m.home_goals : m.away_goals;
        const ga = atHome ? m.away_goals : m.home_goals;
        const res = gf > ga ? "W" : gf === ga ? "D" : "L";
        if (res === "W") w++; else if (res === "D") d++; else l++;
        return el("tr", {},
          el("td", { class: "num", text: fmtShortDate(m.date) }),
          el("td", { text: atHome ? m.away : m.home }),
          el("td", { text: atHome ? "home" : "away" }),
          el("td", { class: "num", text: `${gf}–${ga}` }),
          el("td", { class: `res f-${res}`, text: res }));
      });
      return el("details", { open: season === seasons[0] ? true : null },
        el("summary", {}, el("strong", { text: season }), `: played ${games.length}, won ${w}, drew ${d}, lost ${l}`,
          season === current ? " (this season)" : ""),
        el("table", { class: "data" },
          el("thead", {}, el("tr", {},
            el("th", { scope: "col", text: "Date" }), el("th", { scope: "col", text: "Opponent" }),
            el("th", { scope: "col", text: "Venue" }), el("th", { scope: "col", text: `Score (${name} first)` }),
            el("th", { scope: "col", text: "Result" }))),
          el("tbody", {}, rows)));
    });

    out.replaceChildren(el("h3", {}, teamLabel(ctx, name)), ...summary, ...blocks);
  };
  sel.addEventListener("change", draw);
  window.showTeam = (name, jump) => {
    sel.value = name;
    draw();
    if (jump) {
      const heading = document.getElementById("history-h");
      heading.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      heading.focus({ preventScroll: true });
    }
  };
  const first = loaded.predictions.ok && loaded.predictions.data.predictions[0];
  sel.value = first ? first.home : names[0];
  draw();
}

/* ---------- Record: backtest and live, kept apart ---------- */

const MODEL_NAMES = {
  poisson: "this model",
  league_average: "league average",
  always_home: "always 90% home win",
};

function renderRecord(loaded, ctx) {
  const out = document.getElementById("record-out");
  const missing = missingFor(loaded, SECTIONS.record.needs);
  if (missing.length) { sectionError(out, loaded, missing, "The record"); return; }
  const { backtest, live } = loaded.scoreboard.data;
  out.replaceChildren(backtestBlock(backtest), liveBlock(loaded, ctx, live));
}

function rulerMark(value, lo, hi, label, side, cls = "") {
  const pct = ((value - lo) / (hi - lo)) * 100;
  const anchor = pct < 12 ? "anchor-start" : pct > 88 ? "anchor-end" : "anchor-mid";
  return [
    el("span", { class: `mark ${cls}`, style: `left:${pct}%` }),
    el("span", { class: `label ${side} ${anchor}`, style: `left:${pct}%` }, label),
  ];
}

function backtestBlock(bt) {
  const loss = Object.fromEntries(bt.models.map((m) => [m.model, m]));
  const model = loss.poisson;
  const base = loss.league_average;
  const home = loss.always_home;
  const uniform = bt.uniform_log_loss;
  const lab = (name, value) => [name, el("br"), num(value.toFixed(3))];

  // Full scale, 0 (a perfect forecaster) to past the worst baseline.
  const fullHi = Math.ceil((home.log_loss + 0.1) * 10) / 10;
  const closeLo = Math.min(model.log_loss, base.log_loss, uniform) - 0.025;
  const closeHi = Math.max(model.log_loss, base.log_loss, uniform) + 0.025;
  const pct = (v) => (v / fullHi) * 100;

  const full = el("div", { class: "ruler", role: "img",
    "aria-label": `Log loss, lower is better. Perfect 0; this model ${model.log_loss.toFixed(3)}; league average ${base.log_loss.toFixed(3)}; one third each ${uniform.toFixed(3)}; always 90% home win ${home.log_loss.toFixed(3)}.` },
    rulerMark(0, 0, fullHi, lab("perfect", 0), "above"),
    el("span", { class: "bracket", style: `left:${pct(closeLo)}%;width:${pct(closeHi) - pct(closeLo)}%` }),
    el("span", { class: "label above anchor-mid", style: `left:${(pct(closeLo) + pct(closeHi)) / 2}%` }, "enlarged below"),
    rulerMark(home.log_loss, 0, fullHi, lab(MODEL_NAMES.always_home, home.log_loss), "above"));

  const z = (v) => ((v - closeLo) / (closeHi - closeLo)) * 100;
  const zoom = el("div", { class: "ruler ruler-zoom", role: "img",
    "aria-label": `Enlarged: this model ${model.log_loss.toFixed(3)}, league average ${base.log_loss.toFixed(3)}, one third each ${uniform.toFixed(3)}.` },
    el("span", { class: "gap", style: `left:${z(model.log_loss)}%;width:${z(base.log_loss) - z(model.log_loss)}%` }),
    rulerMark(model.log_loss, closeLo, closeHi, lab(MODEL_NAMES.poisson, model.log_loss), "above", "m-model"),
    rulerMark(base.log_loss, closeLo, closeHi, lab(MODEL_NAMES.league_average, base.log_loss), "below"),
    rulerMark(uniform, closeLo, closeHi, lab("one third each", uniform), "above"));

  const magnification = Math.round(fullHi / (closeHi - closeLo));

  const main = el("div", { class: "col-main" },
    el("h3", { class: "subhead" }, "Backtest ", el("span", { class: "tag", text: "tested on the past" })),
    el("p", {}, `${fmtShortDate(bt.window_start)} to ${fmtShortDate(bt.window_end)}, the 2024-25 and 2025-26 seasons. Each match was predicted using only matches played before it, then scored. `,
      `${model.n} matches for the model, ${base.n} for the baselines.`),
    el("p", {}, "Log loss, lower is better:"),
    full,
    zoom,
    el("p", { class: "zoom-caption", text: `The second line is the same scale enlarged about ${magnification} times.` }),
    el("p", {}, `The model's log loss is ${(bt.improvement_over_league_average * 100).toFixed(1)}% lower than the league-average baseline's. `
      + "The baseline itself is barely better than guessing one third for each result, which shows how much of football is noise."),
    el("p", { class: "mono", style: "font-size:0.8rem", text: `Configuration: ${bt.config}` }));

  const notes = el("aside", { class: "col-note" },
    el("p", { class: "note" }, "The test window was opened twice. The first run (3.1%) was dominated by one impossible prediction. Ridge regularisation, chosen on separate validation data, fixed that, and the test window was run once more. ",
      el("a", { href: "#method", text: "The full story" }), "."),
    el("p", { class: "note", text: `The model is scored on ${base.n - model.n} fewer matches than the baselines. A newly promoted team's first match can't be predicted when the team has no matches in the data yet (Ipswich in 2024-25, Sunderland in 2025-26).` }));

  return el("div", { class: "layout record-block" }, main, notes);
}

function liveBlock(loaded, ctx, live) {
  const main = el("div", { class: "col-main" },
    el("h3", { class: "subhead", style: "margin-top:2.5rem" }, "Live ", el("span", { class: "tag", text: "tested on the future" })));
  const notes = el("aside", { class: "col-note" });

  if (live.n === 0) {
    main.append(el("p", { class: "live-stat" }, "No live predictions have been scored yet."));
  } else {
    main.append(el("p", { class: "live-stat" },
      `Scored so far: `, num(String(live.n)), ` ${plural(live.n, "match", "matches")}, average log loss `, num(live.log_loss.toFixed(3)), "."));
    notes.append(el("p", { class: "note", text: `With ${live.n} ${plural(live.n, "match", "matches")}, one surprising result moves this a lot. Until it covers a few hundred matches, the backtest is the better guide.` }));
  }

  const ledgerOk = loaded.predictions.ok && loaded.matrix.ok && loaded.teams.ok;
  if (!ledgerOk) {
    const failed = ["predictions", "matrix", "teams"].filter((n) => !loaded[n].ok);
    main.append(el("div", { class: "section-error" }, el("p", {},
      "The list of locked predictions needs ", failed.map((n, i) => [i ? ", " : "", el("code", { text: `data/${n}.json` })]),
      `, which couldn't be loaded.`)));
    return el("div", { class: "layout" }, main, notes);
  }

  const preds = [...loaded.predictions.data.predictions].sort((a, b) => a.kickoff.localeCompare(b.kickoff));
  if (!preds.length) {
    main.append(el("p", { text: "There are no locked predictions waiting for a result right now." }));
    return el("div", { class: "layout" }, main, notes);
  }

  const first = preds[0];
  const last = preds[preds.length - 1];
  main.append(el("p", {},
    live.n === 0
      ? `The ledger opens on ${fmtDay(first.kickoff)}. These ${preds.length} predictions were locked before kickoff and will each be scored once the match is played. The first to be scored is ${first.home} v ${first.away}. Come back after ${fmtDay(last.kickoff)} to see how they did.`
      : `Next to be scored: ${preds.length} locked ${plural(preds.length, "prediction", "predictions")}, starting with ${first.home} v ${first.away} on ${fmtDay(first.kickoff)}.`));

  main.append(el("ol", { class: "ledger", "aria-label": "Locked predictions awaiting results" }, preds.map((p) => {
    const pairing = ctx.pairing.get(`${p.home}|${p.away}`);
    const level = pairing ? pairing.confidence : "low";
    const pr = [p.probabilities.home, p.probabilities.draw, p.probabilities.away];
    const counts = outOf100(pr);
    const labels = counts.map((c, i) => countLabel(c, pr[i]));
    return el("li", {},
      el("time", { class: "when", datetime: p.kickoff, text: fmtKickoff(p.kickoff) }),
      el("span", { class: "teams" }, teamLabel(ctx, p.home), " v ", teamLabel(ctx, p.away)),
      el("span", { class: "probs" },
        el("span", { "aria-hidden": "true" },
          el("span", { class: "p-home", text: labels[0] }), " · ",
          el("span", { class: "p-draw", text: labels[1] }), " · ",
          el("span", { class: "p-away", text: labels[2] })),
        hidden(`${p.home} win ${labels[0]} in 100, draw ${labels[1]}, ${p.away} win ${labels[2]}.`),
        isThin(level) ? el("span", { class: "conf", text: ` ${level} confidence` }) : null),
      el("span", { class: "result", text: "Result: to come" }));
  })));

  notes.append(el("p", { class: "note", text: "Each prediction is written to the store before kickoff and never edited. When a result comes in, its slot is scored against what was written." }));
  return el("div", { class: "layout" }, main, notes);
}

/* ---------- Method: the one data-driven sentence ---------- */

function renderMethodNow(loaded, ctx) {
  const target = document.getElementById("method-now");
  if (!loaded.teams.ok) { target.remove(); return; }
  const thin = ctx.teams.filter((t) => t.confidence === "low");
  if (!thin.length) {
    target.textContent = "Right now no team in the league has fewer than 10 matches in the data.";
    return;
  }
  const parts = thin.map((t) => {
    const s = ctx.standing(t.name);
    const where = s.length ? `is rated the league's ${s.map((x) => x.text).join(" and ")}` : "has ratings resting on very little";
    let text = `${t.name}, on ${t.matches_in_training} ${plural(t.matches_in_training, "match", "matches")}, ${where}`;
    const fixture = loaded.predictions.ok && loaded.predictions.data.predictions.find((p) => p.home === t.name || p.away === t.name);
    if (fixture) {
      const pr = [fixture.probabilities.home, fixture.probabilities.draw, fixture.probabilities.away];
      const c = outOf100(pr).map((x, i) => countLabel(x, pr[i]));
      const side = fixture.home === t.name ? 0 : 2;
      const opp = fixture.home === t.name ? `at home to ${fixture.away}` : `away at ${fixture.home}`;
      text += `; that gives them ${c[side]} in 100 to win ${opp}`;
    }
    return text;
  });
  target.textContent = `Right now: ${parts.join(". ")}.`;
}

/* ---------- Footer ---------- */

function renderFooter(loaded) {
  const stamps = DATA_FILES.filter((n) => loaded[n].ok).map((n) => loaded[n].data.generated_at).filter(Boolean).sort();
  const version = (loaded.matrix.ok && loaded.matrix.data.model_version) || (loaded.teams.ok && loaded.teams.data.model_version);
  const foot = document.getElementById("footer-meta");
  foot.replaceChildren(
    "Probabilities, not tips. ",
    stamps.length ? `Data generated ${fmtStamp(stamps[stamps.length - 1])}. ` : "",
    version ? ["Model ", el("span", { class: "num", text: version }), "."] : "");
}

/* ---------- Boot ---------- */

async function main() {
  const loaded = await loadAll();
  renderLoadBanner(loaded);
  const ctx = buildContext(loaded);
  const steps = [
    ["fixtures", renderFixtures, "fixture-list"],
    ["explorer", renderExplorer, "pair-out"],
    ["teams", renderTeams, "team-table"],
    ["history", renderHistory, "history-out"],
    ["record", renderRecord, "record-out"],
    ["method", renderMethodNow, "method-now"],
  ];
  for (const [key, fn, target] of steps) {
    try {
      fn(loaded, ctx);
    } catch (err) {
      // One section's bug shouldn't blank the rest of the page.
      console.error(err);
      const node = document.getElementById(target);
      if (node) {
        node.replaceChildren(el("div", { class: "section-error" },
          el("p", { text: `${SECTIONS[key].label} couldn't be drawn: ${err.message}. The data loaded, so this is a bug in the page.` })));
      }
    }
  }
  renderFooter(loaded);
}

main();
