# MLB WAR by direct regression

A wins-above-replacement-style baseball metric computed by **direct ridge regression** of half-inning run differential on player presence — not assembled bottom-up from box-score components like Fangraphs / Baseball-Reference WAR.

The methodology is regularized adjusted plus-minus (RAPM, well-known from basketball) adapted to baseball. See [REPORT.md](REPORT.md) for the full writeup, including all design choices, calibration, and caveats.

## Top 15 all-time (park-adjusted)

| # | Player | total_war | off | pit | fld |
|--|--|--|--|--|--|
| 1 | Willie Mays | 264.9 | 193.0 | — | 71.8 |
| 2 | Barry Bonds | 243.4 | 222.5 | — | 20.9 |
| 3 | Albert Pujols | 231.7 | 230.4 | 0.0 | 1.2 |
| 4 | Hank Aaron | 231.5 | 216.5 | — | 15.0 |
| 5 | Cal Ripken | 231.2 | 222.5 | — | 8.7 |
| 6 | Eddie Murray | 220.2 | 221.5 | — | -1.3 |
| 7 | Stan Musial | 219.2 | 178.9 | 0.0 | 40.3 |
| 8 | Dave Winfield | 216.4 | 205.2 | — | 11.2 |
| 9 | Pete Rose | 214.0 | 212.3 | — | 1.7 |
| 10 | Mel Ott | 213.9 | 199.5 | — | 14.3 |
| 11 | Adrian Beltré | 211.6 | 197.5 | — | 14.1 |
| 12 | Rickey Henderson | 207.7 | 212.8 | — | -5.1 |
| 13 | Carl Yastrzemski | 207.0 | 212.5 | — | -5.4 |
| 14 | Frank Robinson | 204.2 | 188.4 | — | 15.8 |
| 15 | Babe Ruth | 200.6 | 186.3 | 5.9 | 8.4 |

Each player has up to three coefficients: offense (when batting), pitcher (when pitching), fielder (when fielding a non-pitcher position). Numbers are runs above MLB-average / 10, summed over the player's career. Note that "0 = average MLB regular," not "replacement," so totals run ~1.5–2× higher than Fangraphs WAR — the *ranking* is what's meaningful.

Top pitcher: Walter Johnson (49.3). Top fielder: Willie Mays (71.2). Full leaderboards in [REPORT.md](REPORT.md).

## How to reproduce

Requires `chadwick` (Retrosheet's `cwevent` C tool, available via `brew install chadwick`) and Python with `pandas`, `scipy`, `scikit-learn`.

```bash
# 1. Download all available Retrosheet seasons (1910-2025, ~900MB)
bash download_all.sh

# 2. Parse events and aggregate to half-inning rows (~3 min)
python3 build_half_innings.py

# 3. Extract game→park lookup from event metadata (~10 sec)
python3 build_game_meta.py

# 4. Fit ridge regression with player + season + park fixed effects (~2 min)
python3 fit_ridge_all.py

# 5. Compute per-season, peak-rate, and per-position-z-score views
python3 make_views.py
```

Output leaderboards are checked in under `data/events/`:
- `coefficients_all.csv` — career totals, sorted by total_war
- `coefficients_all_enriched.csv` — adds per-season, peak-rate, and within-position z-scores

## Files

| File | Purpose |
|---|---|
| `download_all.sh` | Fetch all `<YYYY>eve.zip` from retrosheet.org. |
| `build_half_innings.py` | Run `cwevent` on all year directories; aggregate to half-inning rows. |
| `build_game_meta.py` | Scan event metadata for `info,site` lines; build a game-id → park lookup. |
| `fit_ridge_all.py` | Build sparse design matrix with player + season + park fixed effects; ridge regression with per-role column scaling. |
| `fit_ridge.py` | Earlier single-season (2024-only) version of the regression, kept for reference. |
| `make_views.py` | Post-process coefficients into per-season, full-season-rate, and per-position-z-score derived metrics. |
| `REPORT.md` | Full methodology and results writeup. |
