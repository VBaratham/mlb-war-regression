# MLB WAR by direct regression

A wins-above-replacement-style baseball metric computed by **direct ridge regression** of half-inning run differential on player presence — not assembled bottom-up from box-score components like Fangraphs / Baseball-Reference WAR.

The methodology is regularized adjusted plus-minus (RAPM, well-known from basketball) adapted to baseball. See [REPORT.md](REPORT.md) for the full writeup, including all design choices, calibration, and caveats.

## Top 15 all-time

| # | Player | total_war | off | pit | fld |
|--|--|--|--|--|--|
| 1 | Willie Mays | 262.7 | 191.4 | — | 71.2 |
| 2 | Barry Bonds | 245.9 | 220.4 | — | 25.4 |
| 3 | Cal Ripken | 235.6 | 217.6 | — | 18.1 |
| 4 | Albert Pujols | 229.9 | 224.7 | 0.0 | 5.2 |
| 5 | Hank Aaron | 228.7 | 216.8 | — | 11.9 |
| 6 | Eddie Murray | 220.9 | 214.4 | — | 6.5 |
| 7 | Dave Winfield | 219.7 | 201.0 | — | 18.6 |
| 8 | Adrian Beltré | 213.8 | 194.8 | — | 19.0 |
| 9 | Stan Musial | 213.0 | 185.1 | 0.0 | 27.9 |
| 10 | Pete Rose | 212.8 | 215.9 | — | -3.0 |
| 11 | Mel Ott | 212.2 | 198.9 | — | 13.3 |
| 12 | Brooks Robinson | 209.1 | 165.1 | — | 44.0 |
| 13 | Rickey Henderson | 209.0 | 205.9 | — | 3.1 |
| 14 | Babe Ruth | 204.8 | 181.7 | 5.9 | 17.2 |
| 15 | Carl Yastrzemski | 204.5 | 219.6 | — | -15.1 |

Each player has up to three coefficients: offense (when batting), pitcher (when pitching), fielder (when fielding a non-pitcher position). Numbers are runs above MLB-average / 10, summed over the player's career. Note that "0 = average MLB regular," not "replacement," so totals run ~1.5–2× higher than Fangraphs WAR — the *ranking* is what's meaningful.

Top pitcher: Walter Johnson (49.3). Top fielder: Willie Mays (71.2). Full leaderboards in [REPORT.md](REPORT.md).

## How to reproduce

Requires `chadwick` (Retrosheet's `cwevent` C tool, available via `brew install chadwick`) and Python with `pandas`, `scipy`, `scikit-learn`.

```bash
# 1. Download all available Retrosheet seasons (1910-2025, ~900MB)
bash download_all.sh

# 2. Parse events and aggregate to half-inning rows (~3 min)
python3 build_half_innings.py

# 3. Fit ridge regression on full dataset (~2 min)
python3 fit_ridge_all.py

# 4. Compute per-season, peak-rate, and per-position-z-score views
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
| `fit_ridge_all.py` | Build sparse 34k-column design matrix; ridge regression with per-role column scaling and season fixed effects. |
| `fit_ridge.py` | Earlier single-season (2024-only) version of the regression, kept for reference. |
| `make_views.py` | Post-process coefficients into per-season, full-season-rate, and per-position-z-score derived metrics. |
| `REPORT.md` | Full methodology and results writeup. |
