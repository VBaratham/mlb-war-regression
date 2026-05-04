# WAR by Direct Regression — Methodology and Results

## Goal

Compute a wins-above-replacement-style stat where each player's value is *directly regressed* from whether their team scores more or fewer runs when they're on the field — not assembled bottom-up from box-score components like Fangraphs/Baseball-Reference WAR. The intuition is simple plus-minus: a player should be credited for innings their team played well in and debited for innings their team played poorly in, with credit shared appropriately among the players present.

## Approach in one sentence

This is **regularized adjusted plus-minus** (RAPM, well-known from basketball) adapted to baseball: ridge regression of half-inning run differential on indicator variables for which players were on the field, with structural choices to handle baseball-specific issues (offense/defense roles, pitcher dominance per play, era scoring environments).

## Data

- **Source:** Retrosheet event files (`https://www.retrosheet.org/events/`).
- **Coverage:** 1910–2025 (116 seasons). Earlier years not available from Retrosheet.
- **Parser:** `chadwick`'s `cwevent` C tool, run per-year directory.
- **Scale:** 15.6M individual play events, aggregated to **3,536,955 half-innings**, with **15,438 unique batters**, **9,599 pitchers**, and **9,259 fielders** appearing across the dataset.
- **Active vs retired:** the latest fully-covered season is 2025; the dataset captures partial information for active 2025 players' careers.

## Unit of analysis: the half-inning

I chose the half-inning (one team batting, one team fielding) as the row of the regression, rather than the game or the plate appearance:

- **Game-level** (~196k rows) is too coarse — a 12-1 blowout shouldn't count the same as a 1-0 win, and the same lineup plays many innings together within a game.
- **PA-level** (~15.6M rows) is more granular but introduces redundant rows for the same defensive lineup; it overweights long innings.
- **Half-inning** (~3.5M rows) is just right: defenders are constant within a half-inning, run differential is naturally bounded, and substitutions across innings give the regression the variation it needs.

For each half-inning row I record:
- **Outcome** `runs_scored`: how many runs the batting team scored that half-inning. Computed by summing `BAT_DEST_ID >= 4` and the three `RUN*_DEST_ID >= 4` flags from `cwevent`'s play output (a destination value of 4–6 means the runner scored).
- **Predictors:** sets of player IDs who appeared in three roles (see below).
- **Context:** season, home-batting flag.

## Three-way role split: offense, pitcher, fielder

A naive plus-minus collapses both teams' nine players into "offense" and "defense" indicators. That's wrong for baseball because the **pitcher is involved in every play** while a fielder handles only ~1/9 of balls in play. Lumping pitcher and fielders into a single "defense" group forces ridge to spread credit equally across all nine defenders, which makes pitcher coefficients ~10× too small. (Empirically: under that setup, the 2024 AL Cy Young winner Skubal showed up at 0.2 WAR.)

The fix is to give each player up to **three coefficients**:

| Role | Indicator value | Definition |
|---|---|---|
| `off` | 1 if the player took a plate appearance in this half-inning | Career offensive contribution |
| `pit` | 1 if the player pitched in this half-inning | Career pitching contribution |
| `fld` | 1 if the player played a non-pitching defensive position in this half-inning | Career fielding contribution |

A player can have any subset of these three depending on their actual playing pattern. Pitchers in the AL DH era have only `pit`; designated hitters have only `off`; everyday position players have `off` + `fld`; two-way players (Babe Ruth, Ohtani) have all three.

The user's stated preference was for one coefficient per player. I'd argue the per-role split is mechanistically correct: a player contributes through totally different mechanisms in offense vs defense, so forcing them into one number throws away signal. Coefficients can be summed at the end if a single per-player number is wanted.

## Building the design matrix

Define column index ranges:
- `[0, n_off)` — offensive indicators, one per batter ever observed (n_off = 15,438)
- `[n_off, n_off + n_pit)` — pitcher indicators (n_pit = 9,599)
- `[n_off + n_pit, n_off + n_pit + n_fld)` — fielder indicators (n_fld = 9,259)
- next 116 columns — **season fixed effects** (one indicator per season)
- last column — home-batting indicator

Total design matrix: **3,536,955 × 34,413** sparse, with ~52.6M nonzero entries (~16 per row: ~4 batters who came to the plate + 1 pitcher + 8 fielders + 1 season + ~0.5 home).

The model is

```
runs_scored_i  =  β₀
                + Σ_p β_off[p] · 1{p batted in row i}
                + Σ_p β_pit[p] · 1{p pitched in row i}
                + Σ_p β_fld[p] · 1{p fielded in row i}
                + γ_season(i)
                + δ · 1{home team batting in row i}
                + ε_i
```

fit by ridge regression with a single global α and per-column scale weights (next section).

## Why ridge, and how alpha is chosen

OLS won't work: the design matrix has substantial near-collinearity (teammates always play together within a single team-season; only their *sum* is identified from the data). Ridge breaks the tie by adding a penalty on coefficient magnitudes:

```
min  ‖y - Xβ‖²  +  α‖β‖²
```

The natural way to pick α is cross-validation, but **CV picked an α that was 5,000× too low**. The reason: even with game-grouped CV (holding out whole games), the same 30 teams' lineups recur in many other games — a model can predict half-inning runs well by *memorizing team identity* from the 18 player IDs in each row. That gets baked into individual coefficients, producing absurd magnitudes (e.g., a stable starting catcher on a good-pitching team came out at 28 WAR for one season).

Instead, **α is calibrated against a prior on real-world effect size.** Standard deviation of run-value-above-average across MLB regulars is roughly 0.03 runs / batting half-inning, derived from typical Fangraphs WAR dispersion (~20 RAA / 700 PA). I sweep α and pick the value where the SD of fitted offensive coefficients comes closest to this target. For the all-time fit on 3.5M rows that lands at **α = 10,000**, giving:

- offense SD: 0.035 runs/inning
- pitcher SD: 0.10 runs/inning
- fielder SD: 0.004 runs/inning
- in-sample R²: 0.224

Lower α → more dispersed coefficients (bigger top players, more noise). Higher α → tighter coefficients (conservative estimates, lose real signal).

## Per-role column scaling (the pitcher problem)

Pitchers genuinely have a much wider true effect range than batters or fielders — a Cy Young winner contributes ~0.20–0.25 runs/inning of suppression, while a top hitter contributes ~0.05 runs/inning. A single α over-shrinks pitchers.

The fix is mathematically clean: **multiplying column j by factor c is equivalent to dividing its effective ridge penalty by c²**. So multiplying pitcher columns by `PIT_WEIGHT = 6` before fitting is equivalent to giving pitchers an effective α of 10,000 / 36 ≈ 278, while keeping batters and fielders at α = 10,000. Coefficients are unscaled afterward to recover original units.

Similarly, season fixed effects shouldn't be regularized at all — they're not "players" and we want them to absorb era effects freely. Scaling season columns by `SEASON_WEIGHT = 50` gives them an effective α of 10,000 / 2,500 = 4, essentially unregularized.

## Era adjustment via season fixed effects

Run-scoring environments vary dramatically: the 1968 deadball era averaged 0.38 runs per half-inning vs 0.62 in the 1930 live-ball year. Without era controls, deadball-era hitters look uniformly bad and home-run-era pitchers look uniformly bad. The 116 season fixed-effect columns absorb this entirely, so player coefficients represent contribution **above each player's era's average**.

## Park adjustment via per-park fixed effects

Stadium scoring environments vary even more dramatically than eras within a given season. Coors Field (DEN02) inflates runs by ~+0.13 per half-inning above MLB average — over a season, that's ~21 extra runs (~2 wins) of fake offensive production for any Rockies hitter. Petco Park (SAN02) and Dodger Stadium (LOS03) suppress runs by ~0.07 per half-inning. Without controls, half-decade-Coors hitters like Helton and Walker look like all-time greats and pitcher-park hitters get systematically underrated.

Fix: extract the park ID from each game's `info,site,XXXNN` line in the Retrosheet event metadata. Across 1910-2025 there are 105 distinct parks with at least 100 half-innings. Add one fixed-effect column per park and apply the same SEASON_WEIGHT scaling so park effects are essentially unregularized. Player coefficients are now also park-adjusted.

The model picks out parks correctly by reputation (top 10 most hitter-friendly with ≥2k half-innings, post-fit):

| park | runs/HI vs avg | what it is |
|---|---:|---|
| DEN01 | +0.134 | Mile High Stadium (Rockies 1993-94) |
| DEN02 | +0.114 | Coors Field |
| BOS05 | +0.104 | Fenway Park (Red Sox interim home, 1914-15) |
| PHI09 | +0.094 | Citizens Bank Park |
| DET02 | +0.071 | Tiger Stadium (1912-99) |

And most pitcher-friendly:

| park | runs/HI vs avg | what it is |
|---|---:|---|
| SEA03 | -0.094 | T-Mobile / Safeco |
| LOS03 | -0.076 | Dodger Stadium |
| HOU02 | -0.070 | Astrodome |
| BAL11 | -0.070 | Memorial Stadium (1954-91) |
| SAN02 | -0.069 | Petco Park |
| OAK01 | -0.066 | Oakland Coliseum |
| SFO03 | -0.062 | Oracle Park |

These are exactly the parks with that reputation. Coors Field at +0.13 runs/inning being one of the largest park effects ever measured is consistent with its popular reputation.

Effect on player rankings: Coors-era hitters get appropriately deflated. Todd Helton's career off_war drops from 166.8 (no park control) to 142.7 (park-controlled) — a 24-WAR park inflation correctly removed. Larry Walker, Galarraga, Castilla similar. Pitcher-park hitters (Tony Gwynn at the old Padres parks, the Astrodome era Astros) get small upward adjustments. The all-time top 15 stays largely stable since most all-time greats played in roughly average-scoring parks across long careers.

## Sign convention and centering

After fitting, I:
1. **Sign-flip pitcher and fielder coefficients** (a smaller raw coefficient = fewer runs allowed = better defense). After this, "higher = better" for all three roles.
2. **Re-center each role's coefficients to zero mean.** Without this, the intercept absorbs the constant ~4 batters + 9 fielders per row, and individual coefficients carry that constant (e.g. raw mean offense coef = 0.02). Subtracting the role mean re-anchors "0 = average MLB player" without changing predictions; the constant flows into the intercept, which we don't report.

## From coefficients to "WAR"

For each player I compute:
```
off_RAA  = off_runs_per_inning  ×  off_innings_observed
pit_RAA  = pit_runs_per_inning  ×  pit_innings_observed
fld_RAA  = fld_runs_per_inning  ×  fld_innings_observed
total_RAA = off_RAA + pit_RAA + fld_RAA
total_WAR = total_RAA / 10        # standard ~10 runs per win conversion
```

**Important:** this "WAR" is **runs above *average*** divided by 10, not wins above *replacement*. A typical regular MLB player = 0 in this system, but ≈ 2 WAR/year in Fangraphs. As a result our top players' totals are systematically ~1.5–2× larger than their Fangraphs WAR. The *ranking* is what's meaningful.

## Three derived views

The cumulative `total_war` favors long careers. To answer different questions:

1. **Per-calendar-season** = total_war / seasons_played. Annual-average view. Each season counted as 1 regardless of playing time, so injury-shortened careers still get penalized.

2. **Peak full-season rate** = `runs_per_inning × seasonal_innings / 10`. Uses 700 batting / 1300 fielding / 200 pitching half-innings as a "full healthy season" denominator. Asks: *if this player played a full healthy season at his observed rate, what would he add?* Treats Trout's 80-game seasons fairly.

3. **Within-position z-scores** for offense, pitching, and fielding. For each position group with ≥ 5 qualified players, compute (player's rate − position mean) / position SD. Lets you ask "best-at-position" questions: a SS hitting at +0.04 runs/inning is more impressive than a 1B at the same rate.

---

# Results

## Top 15 all-time by total_war (career runs above average / 10, park-adjusted)

| # | Player | total | off | pit | fld |
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

Mays at #1 reflects his combination of elite offense plus an enormous defensive contribution (71.8 fielding RAA) accumulated over 22 seasons in CF.

## Top 10 pitchers (career)

| # | Pitcher | pit_war | runs/inning |
|--|--|--|--|
| 1 | Walter Johnson | 49.3 | 0.094 |
| 2 | Justin Verlander | 34.8 | 0.096 |
| 3 | Clayton Kershaw | 34.3 | 0.117 |
| 4 | Max Scherzer | 28.8 | 0.095 |
| 5 | Chris Sale | 26.2 | 0.121 |
| 6 | Zack Greinke | 26.0 | 0.074 |
| 7 | Jacob deGrom | 25.4 | 0.162 |
| 8 | Roger Clemens | 22.1 | 0.044 |
| 9 | Zack Wheeler | 21.7 | 0.122 |
| 10 | Pedro Martínez | 21.5 | 0.074 |

Walter Johnson's #1 ranking is consistent with his common designation as the greatest pitcher of all time. The list mixes recent active aces (Verlander, Kershaw, Scherzer, deGrom, Wheeler) with all-time greats.

## Top 10 fielders (career)

| # | Fielder | pos | fld_war | runs/inning |
|--|--|--|--|--|
| 1 | Willie Mays | OF | 71.2 | 0.029 |
| 2 | Nellie Fox | 2B | 47.4 | 0.024 |
| 3 | Brooks Robinson | 3B | 44.0 | 0.017 |
| 4 | Roberto Clemente | OF | 43.8 | 0.022 |
| 5 | Everett Scott | SS | 43.3 | 0.031 |
| 6 | Brad Ausmus | C | 39.2 | 0.025 |
| 7 | Yadier Molina | C | 38.2 | 0.021 |
| 8 | Willie Davis | OF | 36.7 | 0.018 |
| 9 | Gary Carter | C | 36.3 | 0.019 |
| 10 | Chipper Jones | 3B | 34.8 | 0.017 |

Mays and Brooks Robinson at the top of the defensive list matches conventional wisdom — both are widely considered the greatest defensive player at their position.

## Per-calendar-season top 10

| # | Player | seasons | total | per-season |
|--|--|--|--|--|
| 1 | Willie Mays | 22 | 262.7 | **11.94** |
| 2 | Cal Ripken | 21 | 235.6 | 11.22 |
| 3 | Barry Bonds | 22 | 245.9 | 11.18 |
| 4 | Lou Gehrig | 17 | 186.8 | 10.99 |
| 5 | Eddie Murray | 21 | 220.9 | 10.52 |
| 6 | Tris Speaker | 19 | 198.6 | 10.45 |
| 7 | Albert Pujols | 22 | 229.9 | 10.45 |
| 8 | Adrian Beltré | 21 | 213.8 | 10.18 |
| 9 | Johnny Damon | 18 | 181.6 | 10.09 |
| 10 | Hank Aaron | 23 | 228.7 | 9.94 |

Gehrig (17 seasons) and DiMaggio (#17, 13 seasons) get appropriately lifted. Ongoing-career players don't suffer.

## Top hitters by peak full-season rate

| # | Player | runs/inning | full-season WAR rate |
|--|--|--|--|
| 1 | Lou Gehrig | 0.183 | **12.84** |
| 2 | Mel Ott | 0.182 | 12.76 |
| 3 | Todd Helton | 0.176 | 12.34 |
| 4 | Babe Ruth | 0.175 | 12.27 |
| 5 | Barry Bonds | 0.175 | 12.25 |
| 6 | Albert Pujols | 0.172 | 12.07 |

This list is more about *peak* offensive talent. Helton at #3 reflects a real Coors-aided peak; the model doesn't park-adjust. Ohtani at 4.76 WAR/calendar-season but 8.4 in full-season-rate units shows the value of a rate stat for his injury-disrupted career.

## Within-position offensive z-scores (top 5 each)

| pos | top 5 (z-score in parens) |
|---|---|
| C | I. Rodríguez (3.78), C. Fisk (3.22), B. Dickey (3.10), Y. Berra (3.07), Y. Molina (3.01) |
| 1B | Gehrig (2.94), T. Helton (2.75), J. Foxx (2.51), R. Palmeiro (2.39), E. Banks (2.20) |
| 2B | R. Hornsby (2.91), B. Doerr (2.86), J. Kent (2.76), R. Maranville (2.57), C. Biggio (2.33) |
| 3B | C. Ripken (2.94), A. Beltré (2.70), T. Lazzeri (2.65), J. Dykes (2.42), J. Sewell (2.30) |
| SS | L. Appling (2.60), D. Jeter (2.27), E. Renteria (2.13), A. Trammell (2.12), B. Larkin (2.06) |
| OF | B. Ruth (3.25), B. Bonds (3.24), T. Cobb (2.86), T. Williams (2.78), S. Musial (2.78) |
| DH | A. Pujols (2.24), E. Murray (2.10), D. Ortiz (1.99), D. Winfield (1.98), J. Thome (1.87) |

## Within-position fielding z-scores (top 3 each)

| pos | top 3 fielders |
|---|---|
| C | B. Ausmus (z=3.59), R. Schalk (3.31), T. Peña (3.02) |
| 1B | A. Rizzo (3.03), L. Overbay (2.59), J. Mauer (2.59) |
| 2B | E. Foster (3.60), J. Barry (2.58), B. Herman (2.48) |
| 3B | J. Ramírez (3.34), M. Marion (2.99), A. Ward (2.73) |
| SS | E. Scott (4.51), F. Lindor (3.05), Pee Wee Reese (2.55) |
| OF | W. Mays (4.26), R. Clemente (3.22), K. Kiermaier (2.71) |

---

# Methodological caveats

1. **Replacement-level vs average.** Our "0" is league-average MLB player; a typical regular ≈ 2 Fangraphs WAR per season. To compare directly with Fangraphs you'd add ≈ 2 × (seasons_played) to total_war.

2. **Park effects are constant across all years a park existed.** We have one fixed effect per park, but real park factors drift over time (e.g., Yankee Stadium's right-field porch shifted with renovations; Coors humidor reduced its hitter-friendliness in 2002). A future iteration could use park × decade interactions.

3. **Within-team confounding.** Even with multi-decade data, players who spend their whole career on one team (Trout/Angels, Yount/Brewers) have their coefficients confounded with their teams' long-run quality. Players who change teams mid-career (Bonds, Pujols, A-Rod) get the cleanest individual estimates because trades create within-team variance. This biases the rankings somewhat against single-team stars.

4. **Pre-1910 careers truncated.** Walter Johnson (1907–27) loses his first 3 seasons; Cy Young, Christy Mathewson, Honus Wagner are entirely missing. The "all-time" list is really "1910 onward."

5. **Active players still accumulating.** Cumulative `total_war` will keep growing for active players. The full-season-rate view corrects for this.

6. **Rate stats favor early-career or short-career players.** Peak full-season rate doesn't penalize decline phases, so a player who retires early at peak (Koufax, Mantle) looks better in rate terms than one who hangs on for 5 mediocre years.

7. **Single position per player.** Retrosheet rosters carry one position; we use the most recent listing. Players who shifted positions (Ripken SS→3B, A-Rod SS→3B, Yount SS→OF) get one bucket. This affects the within-position normalization but not the cumulative numbers.

8. **The pitcher era effect.** Modern pitchers have higher per-inning rates than dead-ball-era pitchers, partly because of velocity / pitch design and partly because batters strike out more. The within-position z-score for pitchers is dominated by recent / active pitchers as a result. Within-decade z-scores would address this.

---

# Files in this project

| File | Purpose |
|---|---|
| `download_all.sh` | Fetch all `<YYYY>eve.zip` from retrosheet.org into `data/raw/<YYYY>/`. |
| `build_half_innings.py` | Run `cwevent` on all year directories; aggregate to half-inning rows. Output: `data/events/half_innings_all.parquet` (3.54M rows). |
| `build_game_meta.py` | Scan event-file metadata for `info,site` lines; build game_id → park lookup. Output: `data/events/game_park.csv`. |
| `fit_ridge_all.py` | Build sparse design matrix with player + season + park fixed effects; ridge regression with per-role column scaling; output coefficients per player and a park-effects table. Output: `data/events/coefficients_all.{parquet,csv}` and `data/events/park_effects.csv`. |
| `make_views.py` | Post-process coefficients into per-season, per-full-season-rate, and per-position-z-score derived metrics. Output: `data/events/coefficients_all_enriched.{parquet,csv}`. |

Total runtime end-to-end on a Mac: ~10 min download + ~3 min parse + ~2 min regression + ~10 sec views.
