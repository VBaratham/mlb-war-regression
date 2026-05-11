// Static frontend for mlb_war_regression. Loads pre-computed CSVs that the
// cronjob in the repo refreshes, so no backend needed.
//
// DATA_BASE: where the data/events/ directory lives, relative to this page.
// Default assumes the webapp is served from the repo root (e.g. via GitHub
// Pages) at <root>/webapp/. To host elsewhere, point DATA_BASE at a
// raw.githubusercontent.com URL for the events dir.
const DATA_BASE = "../data/events/";

// Retrosheet uses team codes that differ from the common modern abbrevs
// (NYA = Yankees, NYN = Mets, etc.). Translate at display time so the
// underlying data keeps its retro identifiers.
const TEAM_DISPLAY = {
  NYA: "NYY",   // Yankees
  NYN: "NYM",   // Mets
  ANA: "LAA",   // Angels
  CHA: "CWS",   // White Sox
  CHN: "CHC",   // Cubs
  LAN: "LAD",   // Dodgers
  SDN: "SDP",   // Padres
  SFN: "SFG",   // Giants
  TBA: "TBR",   // Rays
  KCA: "KCR",   // Royals
  SLN: "STL",   // Cardinals
  WAS: "WSN",   // Nationals
};

function displayTeam(t) {
  return TEAM_DISPLAY[t] || t;
}

const SNAPSHOT_TOP_N_DEFAULT = 10;
const TABLE_PAGE = 200;

const WAR_SORT_KEYS = new Set([
  "total_war", "off_war", "pit_war", "fld_war",
  "war_per_season", "peak_season_war",
]);
const CAREER_ONLY_KEYS = new Set(["war_per_season", "peak_season_war"]);

const state = {
  manifest: null,
  view: null,          // 'all_time' | 'current' | 'season'
  data: null,          // current leaderboard rows (the table)
  snapshots: null,     // { 'YYYY-MM-DD': rows[] } once loaded
  seasonWar: null,     // long-format season_war rows once loaded (lazy)
  careerByPlayer: null,  // player_id -> career_seasons_sum row (lazy cache)
  seasonYear: null,    // active year for the single-season view
  filters: { search: "", positions: new Set(), minInn: 500 },
  sort: { key: "total_war", dir: "desc" },
  rankKey: "total_war",   // tracks the most recently picked WAR column;
                          // the "#" column reflects rank by THIS key so
                          // sorting by name (etc.) still shows WAR rank.
  visibleLimit: TABLE_PAGE,
  topN: SNAPSHOT_TOP_N_DEFAULT,
};

async function loadCSV(path) {
  const res = await fetch(DATA_BASE + path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`failed to load ${path}: ${res.status}`);
  const text = await res.text();
  const parsed = Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  if (parsed.errors.length) console.warn("csv parse warnings:", parsed.errors);
  return parsed.data;
}

async function loadManifest() {
  const r = await fetch(DATA_BASE + "manifest.json", { cache: "no-cache" });
  if (!r.ok) throw new Error(`manifest.json missing (${r.status})`);
  return r.json();
}

function populateViewSelector() {
  const sel = document.getElementById("view");
  sel.innerHTML = "";
  if (state.manifest.all_time) {
    sel.append(new Option(state.manifest.all_time.label, "all_time"));
  }
  if (state.manifest.all_time_single_fit) {
    sel.append(new Option(
      state.manifest.all_time_single_fit.label,
      "all_time_single_fit"
    ));
  }
  if (state.manifest.season_index) {
    const seasons = [...state.manifest.season_index.seasons].sort((a, b) => b - a);
    const cs = state.manifest.current_season;
    const currentYear = cs ? cs.season : null;
    const latestSnapDate = (cs && cs.snapshots && cs.snapshots.length)
      ? cs.snapshots[cs.snapshots.length - 1].date
      : null;
    seasons.forEach(y => {
      let label = String(y);
      if (y === currentYear && latestSnapDate) {
        const md = latestSnapDate.slice(5).replace("-", "/");  // "MM/DD"
        label = `${y} (through ${md})`;
      }
      sel.append(new Option(label, `season:${y}`));
    });
  }
  if (!sel.options.length) {
    sel.append(new Option("(no data)", ""));
    sel.disabled = true;
  }
}

async function ensureSeasonWarLoaded() {
  if (state.seasonWar) return;
  if (!state.manifest.season_index) return;
  state.seasonWar = await loadCSV(state.manifest.season_index.file);
}

async function ensureCareerLoaded() {
  if (state.careerByPlayer) return;
  if (!state.manifest.all_time) return;
  const rows = await loadCSV(state.manifest.all_time.leaderboard);
  state.careerByPlayer = new Map(rows.map(r => [r.player_id, r]));
}

async function switchView(viewKey) {
  const banner = document.getElementById("caveat-banner");
  banner.hidden = true;
  banner.innerHTML = "";

  // Season views hide the career-only WAR columns (war_per_season, peak).
  // If one of them is the active rank/sort key, fall back to total_war so
  // the # column and sort indicator don't reference a hidden column.
  const seasonView = viewKey.startsWith("season:");
  if (seasonView) {
    if (CAREER_ONLY_KEYS.has(state.rankKey)) state.rankKey = "total_war";
    if (CAREER_ONLY_KEYS.has(state.sort.key)) {
      state.sort.key = "total_war";
      state.sort.dir = "desc";
    }
  }

  if (viewKey === "all_time" || viewKey === "all_time_single_fit") {
    state.view = viewKey;
    state.seasonYear = null;
    const node = state.manifest[viewKey];
    state.data = await loadCSV(node.leaderboard);
    state.snapshots = null;
    if (viewKey === "all_time_single_fit") {
      banner.innerHTML =
        "<strong>Caveat:</strong> this view comes from a single all-time " +
        "ridge fit. It has known cross-era bias for pitchers (some HOFers " +
        "like Gaylord Perry and Nolan Ryan come out near zero or negative) " +
        "because half-inning credit-sharing and season/pitcher confounds " +
        "make individual pitcher coefficients hard to identify across eras. " +
        "The default <em>All-time</em> view sums each player's per-season " +
        "WAR (same convention as Fangraphs/B-Ref career WAR) and avoids " +
        "this issue.";
      banner.hidden = false;
    }
  } else if (viewKey.startsWith("season:")) {
    state.view = "season";
    state.seasonYear = parseInt(viewKey.split(":")[1]);
    await ensureSeasonWarLoaded();
    state.data = state.seasonWar.filter(r => Number(r.season) === state.seasonYear);

    // Snapshots only exist for the in-progress current season; if we're
    // viewing that year, fetch them so the WAR-over-time chart shows up.
    const cs = state.manifest.current_season;
    if (cs && cs.season === state.seasonYear && cs.snapshots && cs.snapshots.length) {
      const entries = await Promise.all(cs.snapshots.map(async s => {
        try {
          return [s.date, await loadCSV(s.file)];
        } catch (e) {
          console.warn(`failed to load snapshot ${s.date}: ${e}`);
          return [s.date, null];
        }
      }));
      state.snapshots = Object.fromEntries(entries.filter(([_, v]) => v));
    } else {
      state.snapshots = null;
    }
  }

  const chartVisible = state.view === "season"
    && state.snapshots
    && Object.keys(state.snapshots).length >= 1;
  document.getElementById("chart-section").hidden = !chartVisible;

  document.body.dataset.view = state.view;

  // Sensible default min-innings: all-time wants a higher floor so the table
  // isn't dominated by 19th-century cup-of-coffee guys; single seasons need
  // a much lower floor since players accumulate few innings.
  const isAllTime = state.view === "all_time" || state.view === "all_time_single_fit";
  const target = isAllTime ? 1500 : 100;
  state.filters.minInn = target;
  document.getElementById("min-inn").value = target;

  resetVisible();
  render();
}

function resetVisible() {
  state.visibleLimit = TABLE_PAGE;
}

function hookControls() {
  const $ = id => document.getElementById(id);
  $("view").addEventListener("change", e => switchView(e.target.value));
  $("search").addEventListener("input", e => {
    state.filters.search = e.target.value.toLowerCase();
    resetVisible();
    render();
  });
  document.querySelectorAll("#pos-group input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", () => {
      const sel = new Set();
      document.querySelectorAll("#pos-group input:checked").forEach(c => sel.add(c.value));
      state.filters.positions = sel;
      resetVisible();
      render();
    });
  });
  $("min-inn").addEventListener("input", e => {
    state.filters.minInn = parseInt(e.target.value) || 0;
    resetVisible();
    render();
  });
  document.querySelectorAll("#leaderboard th[data-sort]").forEach(th => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      const def = th.dataset.default || "asc";
      if (state.sort.key === key) {
        state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
      } else {
        state.sort.key = key;
        state.sort.dir = def;
      }
      // Remember the most-recent WAR-column sort so the # column keeps
      // reporting WAR rank even when the user switches to sort by name.
      if (WAR_SORT_KEYS.has(key)) state.rankKey = key;
      resetVisible();
      render();
    });
  });
  $("topn").addEventListener("input", e => { state.topN = Math.max(1, parseInt(e.target.value) || SNAPSHOT_TOP_N_DEFAULT); renderChart(); });
  $("load-more").addEventListener("click", () => {
    state.visibleLimit += TABLE_PAGE;
    render();
  });
  document.querySelector("#leaderboard tbody").addEventListener("click", e => {
    const tr = e.target.closest("tr[data-player-id]");
    if (tr) openPlayerDetail(tr.dataset.playerId);
  });
  document.querySelectorAll("#player-modal [data-close]").forEach(el => {
    el.addEventListener("click", closePlayerDetail);
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && !document.getElementById("player-modal").hidden) {
      closePlayerDetail();
    }
  });
}

function totalInnings(r) {
  return (r.off_innings || 0) + (r.pit_innings || 0) + (r.fld_innings || 0);
}

// Apply position + min-innings + sort -- but NOT the name search. Used by
// renderTable to compute each row's rank in the leaderboard before the
// name filter narrows it down, so a search for "Mays" still shows his
// true overall rank.
function rankedRows() {
  const { positions, minInn } = state.filters;
  const rows = state.data.filter(r => {
    if (positions.size && !positions.has(r.pos)) return false;
    if (totalInnings(r) < minInn) return false;
    return true;
  });
  const { key, dir } = state.sort;
  const mult = dir === "asc" ? 1 : -1;
  const lookup = key === "total_innings"
    ? totalInnings
    : (r => r[key]);
  rows.sort((a, b) => {
    const av = lookup(a), bv = lookup(b);
    const aNum = typeof av === "number" && !isNaN(av);
    const bNum = typeof bv === "number" && !isNaN(bv);
    if (aNum && bNum) return (av - bv) * mult;
    // Push missing values to the end regardless of direction.
    if (av == null || av === "" || (typeof av === "number" && isNaN(av))) return 1;
    if (bv == null || bv === "" || (typeof bv === "number" && isNaN(bv))) return -1;
    return String(av).localeCompare(String(bv)) * mult;
  });
  return rows;
}

function fmt(v, digits = 2) {
  if (v == null || v === "" || isNaN(v)) return "";
  return Number(v).toFixed(digits);
}

async function openPlayerDetail(playerId) {
  await ensureSeasonWarLoaded();
  if (!state.seasonWar) return;
  const rows = state.seasonWar
    .filter(r => r.player_id === playerId)
    .sort((a, b) => Number(a.season) - Number(b.season));
  if (!rows.length) return;
  // Modal header uses CAREER-level metadata (modal team, full chronological
  // teams list, total seasons). Look it up in career_seasons_sum regardless
  // of which view triggered the modal -- the row from state.data is per-
  // season in season views.
  await ensureCareerLoaded();
  const career = (state.careerByPlayer && state.careerByPlayer.get(playerId))
    || state.data.find(r => r.player_id === playerId)
    || rows[rows.length - 1];

  const modal = document.getElementById("player-modal");
  modal.hidden = false;
  document.getElementById("player-modal-title").textContent =
    career.name || rows[0].name || playerId;
  const teamsChrono = (career.teams || career.team || "").split("|").filter(Boolean);
  const modalTeam = career.team || "";
  const teamsOrdered = modalTeam && teamsChrono.includes(modalTeam)
    ? [modalTeam, ...teamsChrono.filter(t => t !== modalTeam)]
    : teamsChrono;
  const modalDisplay = displayTeam(modalTeam);
  const seen = new Set();
  const teamLabel = teamsOrdered.map(displayTeam).filter(d => {
    if (seen.has(d)) return false;
    seen.add(d);
    return true;
  }).map(d => d === modalDisplay
    ? `<strong>${escapeHtml(d)}</strong>`
    : escapeHtml(d)).join(", ");
  document.getElementById("player-modal-sub").innerHTML =
    `${escapeHtml(career.pos || "")} &middot; ${teamLabel} &middot; ` +
    `${rows.length} season${rows.length > 1 ? "s" : ""} ` +
    `(${rows[0].season}–${rows[rows.length - 1].season})`;

  const tbody = document.querySelector("#player-seasons tbody");
  let cumulative = 0;
  const cumYears = [], cumWar = [];
  tbody.innerHTML = rows.map(r => {
    cumulative += Number(r.total_war) || 0;
    cumYears.push(String(r.season));
    cumWar.push(cumulative);
    // Per-season team(s). r.teams is pipe-joined; multi-team rows mean a
    // mid-season trade. Translate retro codes to common abbrevs.
    const seasonTeams = String(r.teams || r.team || "")
      .split("|").filter(Boolean).map(displayTeam).join(", ");
    return `<tr>
      <td class="num">${r.season}</td>
      <td>${escapeHtml(seasonTeams)}</td>
      <td class="num">${fmt(r.total_war)}</td>
      <td class="num">${fmt(r.off_war)}</td>
      <td class="num">${fmt(r.pit_war)}</td>
      <td class="num">${fmt(r.fld_war)}</td>
      <td class="num">${Number(r.off_innings || 0).toLocaleString()}</td>
      <td class="num">${Number(r.pit_innings || 0).toLocaleString()}</td>
      <td class="num">${Number(r.fld_innings || 0).toLocaleString()}</td>
    </tr>`;
  }).join("");

  const css = getComputedStyle(document.documentElement);
  const fg = css.getPropertyValue("--fg").trim() || "#1a1a1a";
  const bg = css.getPropertyValue("--bg").trim() || "#ffffff";
  const accent = css.getPropertyValue("--accent").trim() || "#1f4a80";
  const gridc = css.getPropertyValue("--border").trim() || "#d8d8d8";

  // Align the y=0 baseline on both axes. Without explicit ranges, Plotly
  // auto-fits each axis independently, so cumulative WAR (typically all
  // positive) gets zero at the bottom while season WAR (mixed signs) puts
  // zero a third of the way up. Compute matching fractions-below-zero.
  const seasonVals = rows.map(r => Number(r.total_war) || 0);
  const PAD = 1.05;
  const sMin = Math.min(0, ...seasonVals) * PAD;
  const sMax = Math.max(0, ...seasonVals) * PAD;
  const cMin = Math.min(0, ...cumWar) * PAD;
  const cMax = Math.max(0, ...cumWar) * PAD;
  const sSpan = sMax - sMin || 1;
  const cSpan = cMax - cMin || 1;
  // Fraction of each axis dedicated to negative space; align on the larger.
  const fS = -sMin / sSpan;
  const fC = -cMin / cSpan;
  const f = Math.max(fS, fC);
  const yRange  = sMax > 0 ? [-f / (1 - f) * sMax, sMax] : [sMin, 0];
  const y2Range = cMax > 0 ? [-f / (1 - f) * cMax, cMax] : [cMin, 0];

  Plotly.react("player-chart", [
    {
      x: rows.map(r => String(r.season)),
      y: seasonVals,
      type: "bar",
      name: "Season WAR",
      marker: { color: seasonVals.map(v => v >= 0 ? accent : "#c44") },
      hovertemplate: "%{x}: %{y:.2f} WAR<extra></extra>",
    },
    {
      x: cumYears,
      y: cumWar,
      type: "scatter",
      mode: "lines+markers",
      name: "Cumulative",
      yaxis: "y2",
      line: { color: fg },
      hovertemplate: "%{x}: %{y:.1f} cumulative WAR<extra></extra>",
    },
  ], {
    margin: { t: 20, l: 50, r: 50, b: 60 },
    paper_bgcolor: bg,
    plot_bgcolor: bg,
    font: { color: fg },
    xaxis: { title: "", type: "category", gridcolor: gridc, linecolor: gridc },
    yaxis: { title: "season WAR", range: yRange, gridcolor: gridc, linecolor: gridc, zerolinecolor: gridc },
    yaxis2: { title: "cumulative WAR", range: y2Range, overlaying: "y", side: "right", showgrid: false, linecolor: gridc },
    legend: { orientation: "h", y: -0.25 },
    barmode: "relative",
  }, { responsive: true, displaylogo: false });
}

function closePlayerDetail() {
  document.getElementById("player-modal").hidden = true;
}

const TEAMS_VISIBLE = 2;

function renderTeams(r) {
  const modal = r.team || "";
  const chronologicalRaw = (r.teams || modal || "").split("|").filter(Boolean);
  if (!chronologicalRaw.length) return "";
  // Put the modal team first, then the rest in chronological order. Dedupe
  // after translating retro codes -- a franchise with both retro (ANA) and
  // statsapi (LAA) codings collapses to one display entry.
  const orderedRaw = modal && chronologicalRaw.includes(modal)
    ? [modal, ...chronologicalRaw.filter(t => t !== modal)]
    : chronologicalRaw;
  const seen = new Set();
  const ordered = [];
  for (const t of orderedRaw) {
    const d = displayTeam(t);
    if (!seen.has(d)) { seen.add(d); ordered.push({ raw: t, display: d }); }
  }
  const visible = ordered.slice(0, TEAMS_VISIBLE);
  const hiddenCount = ordered.length - visible.length;
  const modalDisplay = displayTeam(modal);
  const inline = visible.map(({ display }, i) => {
    const html = escapeHtml(display);
    return i === 0 && display === modalDisplay ? `<strong>${html}</strong>` : html;
  }).join(", ");
  const more = hiddenCount > 0 ? `, +${hiddenCount}` : "";
  const fullList = ordered.map(o => o.display).join(", ");
  return `<span class="teams" title="${escapeHtml(fullList)}">${inline}${more}</span>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderSortIndicators() {
  document.querySelectorAll("#leaderboard th[data-sort]").forEach(th => {
    th.classList.remove("sort-asc", "sort-desc", "rank-col");
    if (th.dataset.sort === state.sort.key) {
      th.classList.add(state.sort.dir === "asc" ? "sort-asc" : "sort-desc");
    }
    if (th.dataset.sort === state.rankKey) {
      th.classList.add("rank-col");
    }
  });
}

function renderTable() {
  renderSortIndicators();
  const ranked = rankedRows();
  // Compute ranks against the last-selected WAR column (state.rankKey),
  // not the current sort. So the "#" column reports e.g. "rank by off_war"
  // even when the user re-sorts the table by name.
  const rankKey = state.rankKey;
  const byRankKey = [...ranked].sort(
    (a, b) => (Number(b[rankKey]) || -Infinity) - (Number(a[rankKey]) || -Infinity)
  );
  const ranks = new Map();
  byRankKey.forEach((r, i) => ranks.set(r.player_id, i + 1));
  const search = state.filters.search;
  const rows = search
    ? ranked.filter(r => (r.name || "").toLowerCase().includes(search))
    : ranked;
  const tbody = document.querySelector("#leaderboard tbody");
  const shown = Math.min(rows.length, state.visibleLimit);
  const rankCls = k => k === rankKey ? "num rank-col" : "num";
  tbody.innerHTML = rows.slice(0, shown).map(r => `
    <tr data-player-id="${escapeHtml(r.player_id || "")}">
      <td class="num">${ranks.get(r.player_id)}</td>
      <td>${escapeHtml(r.name || r.player_id || "")}</td>
      <td>${escapeHtml(r.pos || "")}</td>
      <td>${renderTeams(r)}</td>
      <td class="${rankCls("total_war")}">${fmt(r.total_war)}</td>
      <td class="${rankCls("off_war")}">${fmt(r.off_war)}</td>
      <td class="${rankCls("pit_war")}">${fmt(r.pit_war)}</td>
      <td class="${rankCls("fld_war")}">${fmt(r.fld_war)}</td>
      <td class="${rankCls("war_per_season")} career-only">${fmt(r.war_per_season)}</td>
      <td class="${rankCls("peak_season_war")} career-only">${fmt(r.peak_season_war)}</td>
      <td class="num">${totalInnings(r).toLocaleString()}</td>
      <td class="num years-col">${r.first_year || ""}</td>
      <td class="num years-col">${r.last_year_played || r.last_year || ""}</td>
    </tr>`).join("");

  let title;
  if (state.view === "all_time") title = state.manifest.all_time.label;
  else if (state.view === "all_time_single_fit")
    title = state.manifest.all_time_single_fit.label;
  else if (state.view === "season") title = `${state.seasonYear} season`;
  else title = "";
  document.getElementById("table-title").textContent =
    `${title} — ${rows.length.toLocaleString()} qualifiers ` +
    `(showing ${shown.toLocaleString()})`;

  const btn = document.getElementById("load-more");
  if (rows.length > shown) {
    const remaining = rows.length - shown;
    btn.textContent = `Load ${Math.min(remaining, TABLE_PAGE).toLocaleString()} more`;
    btn.hidden = false;
  } else {
    btn.hidden = true;
  }
}

function renderChart() {
  if (!state.snapshots) return;
  // Chart uses the same rows the table shows, including the name search.
  const search = state.filters.search;
  let ranked = rankedRows();
  if (search) ranked = ranked.filter(r => (r.name || "").toLowerCase().includes(search));
  const top = ranked.slice(0, state.topN);
  const dates = Object.keys(state.snapshots).sort();
  if (!dates.length) return;

  // Build an index of player_id -> per-date total_war
  // Strip the year from displayed tick labels (all from one season) to
  // free up vertical space below the chart so the legend doesn't sit on
  // top of the axis. Keep the full ISO date in hover.
  const xLabels = dates.map(d => d.length === 10 ? d.slice(5) : d);
  const traces = top.map(p => {
    const y = dates.map(d => {
      const row = state.snapshots[d].find(r => r.player_id === p.player_id);
      return row ? Number(row.total_war) : null;
    });
    return {
      x: xLabels,
      y,
      mode: dates.length > 1 ? "lines+markers" : "markers",
      type: "scatter",
      name: p.name || p.player_id,
      customdata: dates,
      hovertemplate: "<b>%{fullData.name}</b><br>%{customdata}: %{y:.2f} WAR<extra></extra>",
    };
  });

  const css = getComputedStyle(document.documentElement);
  const fg = css.getPropertyValue("--fg").trim() || "#1a1a1a";
  const bg = css.getPropertyValue("--bg").trim() || "#ffffff";
  const gridc = css.getPropertyValue("--border").trim() || "#d8d8d8";
  Plotly.react("chart", traces, {
    margin: { t: 20, l: 50, r: 20, b: 100 },
    paper_bgcolor: bg,
    plot_bgcolor: bg,
    font: { color: fg },
    xaxis: { title: "", type: "category", gridcolor: gridc, linecolor: gridc, zerolinecolor: gridc },
    yaxis: { title: "cumulative WAR", gridcolor: gridc, linecolor: gridc, zerolinecolor: gridc },
    legend: { orientation: "h", y: -0.35 },
    hovermode: "closest",
  }, { responsive: true, displaylogo: false });

  const note = document.getElementById("chart-note");
  if (dates.length === 1) {
    note.textContent = "Only one snapshot so far; line will fill in as the cron " +
      "publishes daily updates.";
  } else {
    note.textContent = `${dates.length} snapshots (${dates[0]} → ${dates.at(-1)}).`;
  }
}

function render() {
  renderTable();
  if (state.view === "season" && state.snapshots) renderChart();
}

function applyTheme(theme) {
  // theme: "light" | "dark" | null (= follow system)
  const root = document.documentElement;
  if (theme) root.setAttribute("data-theme", theme);
  else root.removeAttribute("data-theme");
  const btn = document.getElementById("theme-toggle");
  if (btn) btn.textContent = effectiveTheme() === "dark" ? "☀" : "🌙";
  // Re-style any rendered Plotly chart for the new background.
  if (state.snapshots && document.getElementById("chart-section") && !document.getElementById("chart-section").hidden) {
    renderChart();
  }
}

function effectiveTheme() {
  const stored = document.documentElement.getAttribute("data-theme");
  if (stored) return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved === "dark" || saved === "light") applyTheme(saved);
  else applyTheme(null);
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    localStorage.setItem("theme", next);
    applyTheme(next);
  });
}

async function init() {
  try {
    initTheme();
    state.manifest = await loadManifest();
    document.getElementById("generated").textContent =
      `updated ${state.manifest.generated_at}`;
    populateViewSelector();
    hookControls();
    const sel = document.getElementById("view");
    if (sel.options.length && !sel.disabled) {
      await switchView(sel.options[0].value);
    }
  } catch (e) {
    document.body.insertAdjacentHTML(
      "beforeend",
      `<pre class="error">${escapeHtml(e.message || e)}</pre>`,
    );
  }
}

document.addEventListener("DOMContentLoaded", init);
