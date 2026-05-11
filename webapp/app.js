// Static frontend for mlb_war_regression. Loads pre-computed CSVs that the
// cronjob in the repo refreshes, so no backend needed.
//
// DATA_BASE: where the data/events/ directory lives, relative to this page.
// Default assumes the webapp is served from the repo root (e.g. via GitHub
// Pages) at <root>/webapp/. To host elsewhere, point DATA_BASE at a
// raw.githubusercontent.com URL for the events dir.
const DATA_BASE = "../data/events/";

const SNAPSHOT_TOP_N_DEFAULT = 10;
const TABLE_PAGE = 200;

const state = {
  manifest: null,
  view: null,          // 'all_time' | 'current'
  data: null,          // current leaderboard rows
  snapshots: null,     // { 'YYYY-MM-DD': rows[] } once loaded
  filters: { search: "", positions: new Set(), minInn: 500 },
  sort: { key: "total_war", dir: "desc" },
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
  if (state.manifest.current_season) {
    sel.append(new Option(state.manifest.current_season.label, "current"));
  }
  if (!sel.options.length) {
    sel.append(new Option("(no data)", ""));
    sel.disabled = true;
  }
}

async function switchView(view) {
  state.view = view;
  const node = view === "all_time"
    ? state.manifest.all_time
    : state.manifest.current_season;
  if (!node) return;

  state.data = await loadCSV(node.leaderboard);

  // Current-season chart needs the per-date snapshot CSVs.
  if (view === "current" && node.snapshots && node.snapshots.length) {
    const entries = await Promise.all(node.snapshots.map(async s => {
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

  const chartVisible = view === "current"
    && state.snapshots
    && Object.keys(state.snapshots).length >= 1;
  document.getElementById("chart-section").hidden = !chartVisible;

  // Tweak default min-innings per view: all-time benefits from a higher
  // floor (so the table isn't dominated by 19th-century cup-of-coffee guys).
  if (view === "all_time" && state.filters.minInn < 1500) {
    state.filters.minInn = 1500;
    document.getElementById("min-inn").value = 1500;
  } else if (view === "current" && state.filters.minInn > 200) {
    state.filters.minInn = 50;
    document.getElementById("min-inn").value = 50;
  }

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
      resetVisible();
      render();
    });
  });
  $("topn").addEventListener("input", e => { state.topN = Math.max(1, parseInt(e.target.value) || SNAPSHOT_TOP_N_DEFAULT); renderChart(); });
  $("load-more").addEventListener("click", () => {
    state.visibleLimit += TABLE_PAGE;
    render();
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

function fmt(v, digits = 1) {
  if (v == null || v === "" || isNaN(v)) return "";
  return Number(v).toFixed(digits);
}

function renderTeams(r) {
  const modal = r.team || "";
  const list = (r.teams || modal || "").split("|").filter(Boolean);
  if (!list.length) return "";
  return list
    .map(t => t === modal
      ? `<strong>${escapeHtml(t)}</strong>`
      : escapeHtml(t))
    .join(", ");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderSortIndicators() {
  document.querySelectorAll("#leaderboard th[data-sort]").forEach(th => {
    th.classList.remove("sort-asc", "sort-desc");
    if (th.dataset.sort === state.sort.key) {
      th.classList.add(state.sort.dir === "asc" ? "sort-asc" : "sort-desc");
    }
  });
}

function renderTable() {
  renderSortIndicators();
  const ranked = rankedRows();
  const search = state.filters.search;
  // Stamp each row with its pre-search rank, then narrow by name search.
  const ranks = new Map();
  ranked.forEach((r, i) => ranks.set(r.player_id, i + 1));
  const rows = search
    ? ranked.filter(r => (r.name || "").toLowerCase().includes(search))
    : ranked;
  const tbody = document.querySelector("#leaderboard tbody");
  const shown = Math.min(rows.length, state.visibleLimit);
  tbody.innerHTML = rows.slice(0, shown).map(r => `
    <tr>
      <td class="num">${ranks.get(r.player_id)}</td>
      <td>${escapeHtml(r.name || r.player_id || "")}</td>
      <td>${escapeHtml(r.pos || "")}</td>
      <td>${renderTeams(r)}</td>
      <td class="num">${fmt(r.total_war)}</td>
      <td class="num">${fmt(r.off_war)}</td>
      <td class="num">${fmt(r.pit_war)}</td>
      <td class="num">${fmt(r.fld_war)}</td>
      <td class="num">${totalInnings(r).toLocaleString()}</td>
      <td class="num">${r.first_year || ""}</td>
      <td class="num">${r.last_year_played || r.last_year || ""}</td>
    </tr>`).join("");

  const title = state.view === "all_time"
    ? state.manifest.all_time.label
    : state.manifest.current_season.label;
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
  const traces = top.map(p => {
    const y = dates.map(d => {
      const row = state.snapshots[d].find(r => r.player_id === p.player_id);
      return row ? Number(row.total_war) : null;
    });
    return {
      x: dates,
      y,
      mode: dates.length > 1 ? "lines+markers" : "markers",
      type: "scatter",
      name: p.name || p.player_id,
      hovertemplate: "<b>%{fullData.name}</b><br>%{x}: %{y:.2f} WAR<extra></extra>",
    };
  });

  Plotly.react("chart", traces, {
    margin: { t: 20, l: 50, r: 20, b: 50 },
    xaxis: { title: "date", type: "category" },
    yaxis: { title: "cumulative WAR" },
    legend: { orientation: "h", y: -0.2 },
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
  if (state.view === "current" && state.snapshots) renderChart();
}

async function init() {
  try {
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
