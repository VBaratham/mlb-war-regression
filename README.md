# MLB WAR by direct regression

A wins-above-replacement-style baseball metric computed by **direct ridge regression** of half-inning run differential on player presence — not assembled bottom-up from box-score components like Fangraphs / Baseball-Reference WAR.

The methodology is regularized adjusted plus-minus (RAPM, well-known from basketball) adapted to baseball. See [REPORT.md](REPORT.md) for the full writeup, including all design choices, calibration, and caveats.

**[Browse the leaderboards interactively →](https://vbaratham.github.io/mlb-war-regression/webapp/)** — all-time, current season with WAR-over-time chart, every individual season since 1910, and click-through to per-player career detail. Refreshed daily.

## Top 15 all-time (career-WAR = sum of per-season fits)

| # | Player | total_war | off | pit | fld | seasons |
|--|--|--|--|--|--|--|
| 1 | Willie Mays | 63.8 | 39.7 | — | 24.2 | 22 |
| 2 | Walter Johnson | 59.0 | -8.2 | 67.2 | — | 18 |
| 3 | Roger Clemens | 54.8 | -0.7 | 55.6 | — | 24 |
| 4 | Tom Seaver | 52.4 | -8.3 | 60.8 | — | 20 |
| 5 | Eddie Murray | 51.7 | 47.7 | — | 4.0 | 21 |
| 6 | Cal Ripken | 49.2 | 41.8 | — | 7.4 | 21 |
| 7 | Brooks Robinson | 49.2 | 44.3 | — | 4.9 | 23 |
| 8 | Frank Robinson | 49.0 | 43.1 | — | 5.9 | 21 |
| 9 | Hank Aaron | 48.3 | 48.6 | — | -0.3 | 23 |
| 10 | Dave Winfield | 47.8 | 46.0 | — | 1.8 | 22 |
| 11 | Lou Gehrig | 47.4 | 41.5 | — | 5.9 | 17 |
| 12 | Pete Alexander | 45.9 | -7.1 | 52.9 | — | 20 |
| 13 | Gil Hodges | 44.2 | 36.3 | — | 7.9 | 18 |
| 14 | Babe Ruth | 43.2 | 30.0 | 9.5 | 3.7 | 22 |
| 15 | Lefty Grove | 42.3 | -4.4 | 46.7 | — | 17 |

The headline career number sums each player's per-season WAR — one ridge fit per season, then aggregated. This matches the Fangraphs/B-Ref convention and avoids the cross-era identifiability issues of fitting one model across all 116 seasons jointly (see [REPORT.md](REPORT.md) §cross-era and §centering). The all-time single-fit version is still available in the webapp under "All-time (single-fit)" with a caveat banner.

## Data sources

The pipeline auto-routes per year:

- **Retrosheet** (`data/raw/<YYYY>/`) — fully detailed historical event files. Used wherever a season's zip exists on disk. `download_all.sh` walks 1910 through the current year and fetches everything Retrosheet has published, then clears any superseded statsapi feed cache for those years.
- **MLB Stats API** (`statsapi.mlb.com`) — used for any year that doesn't have Retrosheet data yet (typically just the current in-progress season). Per-game `feed/live` responses are cached under `data/cache/feeds/<year>/`.
- **Chadwick Bureau register** — MLBAM ↔ Retrosheet player-ID crosswalk so the two sources can be unioned in a single fit. Cached locally on first use.

`build_dataset.py --years 1910-2026 --tag all` HEADs `retrosheet.org` for any year not on disk; if available it downloads on the fly, otherwise it falls back to statsapi. Output: `half_innings_<tag>.parquet` (the row-per-half-inning table the fits consume), `game_park_<tag>.csv`, and `rosters_<tag>.csv`.

## How to reproduce

Requires `chadwick` (Retrosheet's `cwevent` C tool, `brew install chadwick`) and Python with `pandas`, `scipy`, `scikit-learn`.

```bash
# 1. Download every available Retrosheet season (walks 1910 → current year)
bash download_all.sh

# 2. Build the half-innings dataset for the years you want (~5 min on cold cache)
python3 build_dataset.py --years 1910-2026 --tag all

# 3. All-time single-fit (~2 min) -- produces coefficients_all{,_enriched}.{csv,parquet}
python3 fit_ridge_all.py --tag all
python3 make_views.py    --tag all

# 4. Per-season fits (~3 min for 1910-2026) -- produces season_war_all.csv and
#    the per-season-sum career table that the webapp uses as its headline.
python3 fit_per_season.py --tag all --extra-tags 2026

# 5. Regenerate the webapp manifest
python3 snapshot.py --update-manifest-only
```

Output leaderboards under `data/events/`:

| File | What |
|---|---|
| `career_seasons_sum_<tag>.csv` | **Headline career WAR**, derived from the per-season fits. |
| `coefficients_<tag>.csv` / `_enriched.csv` | Single all-time fit with per-season/peak-rate/per-position views. |
| `season_war_<tag>.csv` | Long-format (season × player) WAR table. ~97K rows for 1910-2026. |
| `park_effects_<tag>.csv` | Per-park run environment (single-fit output). |
| `manifest.json` | Index the webapp reads to discover available views. |

## Daily refresh

`refresh.sh` is the cron entry point. It pulls the repo, runs `build_dataset` for the current season (statsapi), refits the current-season models, updates the year's row inside `season_war_all`, regenerates the manifest, and pushes the changed CSVs. Suggested crontab (also checked in as `crontab`):

```cron
PATH=/opt/anaconda3/bin:/usr/local/bin:/usr/bin:/bin
0 6 * * * /Users/vbaratham/claude/mlb_war_regression/refresh.sh >> /Users/vbaratham/claude/mlb_war_regression/refresh.log 2>&1
```

Install with `crontab /path/to/repo/crontab`. On macOS, grant `/usr/sbin/cron` Full Disk Access in System Settings → Privacy & Security.

## Webapp

Static HTML/JS in `webapp/`. No backend. Loads CSVs from `data/events/` via relative paths (or a configurable `DATA_BASE` URL in `app.js` for hosts other than GitHub Pages). Three view modes:

- **All-time** (per-season-sum): the headline.
- **All-time (single-fit)**: legacy single ridge fit across all seasons, with a banner explaining the cross-era caveat.
- **Each individual season** (1910 → current): per-season leaderboard. The current season's entry shows "<year> (through MM/DD)" with a WAR-over-time chart driven by daily snapshots.

Clicking any player row opens a modal with a season-by-season WAR table and a bar chart (per-season WAR) + line (cumulative). Other features: filter by name, multi-position chip filter, click-to-sort columns, dark mode, modern team abbreviations (NYY/NYM/LAD/etc. translated from Retrosheet's NYA/NYN/LAN).

## Files

| File | Purpose |
|---|---|
| `download_all.sh` | Walk 1910..current year, fetch any Retrosheet zips not on disk. |
| `loaders/common.py` | Source-agnostic per-event → half-inning aggregation. |
| `loaders/retro.py` | Retrosheet path (cwevent + park + roster parsing + auto-fetch). |
| `loaders/statsapi.py` | MLB Stats API path (game feed/live + roster API). |
| `loaders/crosswalk.py` | MLBAM ↔ Retrosheet ID mapping via Chadwick register. |
| `build_dataset.py` | Top-level builder; auto-routes years to retro or statsapi. |
| `fit_ridge_all.py` | Single all-time ridge fit (player + season FE + park FE + home). |
| `fit_per_season.py` | One ridge fit per season; emits long-format `season_war`. |
| `make_views.py` | Per-season-rate / peak-rate / position-z views from the single-fit table. |
| `snapshot.py` | Daily snapshot of current-season coefs + manifest.json regeneration. |
| `refresh.sh` | Cron entry point. |
| `webapp/` | Static frontend. |
| `REPORT.md` | Full methodology, calibration, and the cross-era discussion. |
