# mlb_war_regression — static webapp

Single-page static site that reads pre-computed CSV results from this repo
and renders them interactively. Two views:

- **All-time leaderboard** — filter/sort the historical fit.
- **Current season** — same leaderboard for the in-progress year, plus a
  "WAR over time" line chart of the top-N players' cumulative WAR across
  daily snapshots.

No backend. The page fetches:
- `../data/events/manifest.json` — what's available
- `../data/events/coefficients_<tag>_enriched.csv` — leaderboard for a view
- `../data/events/snapshots/<tag>/<date>.csv` — per-day current-season checkpoints

## Local viewing

`fetch()` of relative paths doesn't work from `file://` in most browsers, so
serve the repo root over HTTP:

```bash
cd /path/to/mlb_war_regression
python3 -m http.server 8000
# then open http://localhost:8000/webapp/
```

## Hosting

The simplest deploy is **GitHub Pages** from the repo root, branch `main`:
the app reads `../data/events/` and everything is already in the right place.
After enabling Pages in the repo settings, the site is live at
`https://<user>.github.io/mlb-war-regression/webapp/`.

Hosting elsewhere (Netlify, S3, etc.) works too — either mirror `data/events/`
alongside `webapp/`, or edit `DATA_BASE` in `app.js` to a full
`raw.githubusercontent.com` URL:

```js
const DATA_BASE =
  "https://raw.githubusercontent.com/VBaratham/mlb-war-regression/main/data/events/";
```

## Data refresh

A cronjob in this repo (see `../refresh.sh`) runs daily:

1. Refits the current season (`build_dataset.py` + `fit_ridge_all.py` + `make_views.py`).
2. Writes a dated snapshot to `data/events/snapshots/<season>/<date>.csv`.
3. Regenerates `data/events/manifest.json`.
4. Commits and pushes.

The page picks up new data on the next refresh; no client-side change required.

## Manifest contract

`data/events/manifest.json`:

```json
{
  "generated_at": "2026-05-11T07:04:47Z",
  "all_time": {
    "tag": "all",
    "label": "All-time",
    "leaderboard": "coefficients_all_enriched.csv"
  },
  "current_season": {
    "tag": "2026",
    "season": 2026,
    "label": "2026 season",
    "leaderboard": "coefficients_2026_enriched.csv",
    "snapshots": [
      { "date": "2026-05-11", "file": "snapshots/2026/2026-05-11.csv" }
    ]
  }
}
```

Either top-level key may be `null` until that view has data on disk.
