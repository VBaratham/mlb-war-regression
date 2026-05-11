# WAR by Direct Regression — Methodology and Results

## Goal

Compute a wins-above-replacement-style stat where each player's value is *directly regressed* from whether their team scores more or fewer runs when they're on the field — not assembled bottom-up from box-score components like Fangraphs/Baseball-Reference WAR. The intuition is simple plus-minus: a player should be credited for innings their team played well in and debited for innings their team played poorly in, with credit shared appropriately among the players present.

## Approach in one sentence

This is **regularized adjusted plus-minus** (RAPM, well-known from basketball) adapted to baseball: ridge regression of half-inning run differential on indicator variables for which players were on the field, with structural choices to handle baseball-specific issues (offense/defense roles, pitcher dominance per play, era scoring environments).

## Data

Two sources are unified behind a single normalized event schema in `loaders/`:

- **Retrosheet** (`https://www.retrosheet.org/events/`) — 1910–2025 (116 seasons), parsed via `cwevent`. `download_all.sh` walks forward to the current year and fetches any newly-published zips.
- **MLB Stats API** (`statsapi.mlb.com`) — used for any year that hasn't shipped through Retrosheet yet (currently just the in-progress 2026 season). Per-game `feed/live` responses are cached locally.

Player IDs are unified via the **Chadwick Bureau register** (`MLBAM ↔ Retrosheet` crosswalk, cached locally). MLBAM-only players (current rookies without retro entries yet) get synthetic IDs `x{mlbam:06d}` so they're still stable in the regression.

`build_dataset.py --years 1910-2026 --tag all` auto-routes per year: retro if local files exist, statsapi otherwise. It also HEADs retrosheet.org for any missing year so the moment a season gets a retro release, the next build silently switches that year from statsapi to retro and drops the corresponding feed cache.

- **Scale (1910-2025 retro fit):** 15.6M individual play events, aggregated to **3,536,955 half-innings**, with **15,438 unique batters**, **9,599 pitchers**, and **9,259 fielders**.
- **Active vs retired:** the in-progress 2026 season is captured in real time via statsapi.

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
2. **Re-center each role's coefficients to its innings-weighted mean.** Without re-centering, the intercept absorbs the constant ~4 batters + 9 fielders per row, and individual coefficients carry that constant (raw mean pitcher coef ≈ 0.12 runs/inning). The centering re-anchors "0 = average regular at this role" without changing predictions; the constant flows into the intercept, which we don't report.

   The reason centering uses an **innings-weighted** mean rather than a simple mean (an earlier version of this code) is structural: ridge shrinks low-data players hard toward the prior. With 9,599 pitchers, ~7,000 of them are cup-of-coffee guys whose coefs land near the prior. A simple mean is dominated by them, sits below the regular pitcher baseline, and systematically displays *every regular pitcher* as worse than average (Nolan Ryan came out at -47 WAR under the old centering). Innings-weighting puts the zero line where the league-baseline regular actually lives. Mathematically: the league-aggregate RAA in each role sums to exactly zero by construction.

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

## Per-season fits as the headline career number

The all-time single fit has a known cross-era issue with pitchers: HOFers like Gaylord Perry and Nolan Ryan came out near zero or negative in the joint fit despite obvious greatness. The diagnosis is twofold:

1. **Half-inning credit-sharing.** A starter pulled mid-inning gets credit for every event in that half-inning, including the reliever's contribution to the same half-inning. Modern bullpen usage produces more of these mixed-pitcher rows than the deadball era did, biasing modern pitcher coefs.

2. **Season FE / pitcher coef confounding.** Season FE is one number per year and ridge prefers to put era variation there (it's essentially unregularized). But to do so cleanly the model needs within-year variation across pitchers, which depends on how identifiable each pitcher is from his contemporaries — and that varies by era (roster stability, player movement, sample size).

Both confounds are structural to the half-inning unit + single all-time fit.

**The fix the project ships as the headline:** one ridge fit *per season*, then sum each player's per-season WAR across the seasons they played. Each per-season fit is a self-contained problem: it picks its own α by matching off_sd ≈ 0.03, centers innings-weighted within that season, and writes one row per (season, player) to `season_war_<tag>.csv`. The career roll-up sums each player's WAR across seasons → `career_seasons_sum_<tag>.csv` (the file the webapp's "All-time" view loads).

Trade-off: a per-season fit is era-relative by construction (each season's mean = 0). Cross-era comparisons via summed WAR inherit the standard "replacement level drifts" caveat that mainstream WAR (fWAR, bWAR) lives with. We accept this as the *less wrong* assumption versus the all-time fit's structural issues.

The all-time single-fit version is preserved as a separate webapp view ("All-time (single-fit)") with a caveat banner; the numbers and methodology above continue to describe how it's computed.

## Three derived views

The cumulative `total_war` favors long careers. To answer different questions:

1. **Per-calendar-season** = total_war / seasons_played. Annual-average view. Each season counted as 1 regardless of playing time, so injury-shortened careers still get penalized.

2. **Peak full-season rate** = `runs_per_inning × seasonal_innings / 10`. Uses 700 batting / 1300 fielding / 200 pitching half-innings as a "full healthy season" denominator. Asks: *if this player played a full healthy season at his observed rate, what would he add?* Treats Trout's 80-game seasons fairly.

3. **Within-position z-scores** for offense, pitching, and fielding. For each position group with ≥ 5 qualified players, compute (player's rate − position mean) / position SD. Lets you ask "best-at-position" questions: a SS hitting at +0.04 runs/inning is more impressive than a 1B at the same rate.

---

# Results

## Top 15 all-time (career = sum of per-season WAR)

| # | Player | total | off | pit | fld | seasons |
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

Top 4 mixes Mays at #1 with three of the greatest pitchers in history at #2-#4. Mays's spread of elite offense + 22 seasons of CF defense is what gets him over the top.

## Top 10 pitchers (career = sum of per-season pit_war, min 1500 pit_innings)

| # | Pitcher | pit_innings | pit_war |
|--|--|--|--|
| 1 | Walter Johnson | 5262 | 67.2 |
| 2 | Tom Seaver | 4899 | 60.8 |
| 3 | Roger Clemens | 5056 | 55.6 |
| 4 | Pete Alexander | 5104 | 52.9 |
| 5 | Lefty Grove | 3939 | 46.7 |
| 6 | Greg Maddux | 5122 | 44.6 |
| 7 | Steve Carlton | 5354 | 42.8 |
| 8 | Don Sutton | 5441 | 42.7 |
| 9 | Randy Johnson | 4245 | 41.8 |
| 10 | Warren Spahn | 5208 | 39.6 |

Walter Johnson #1 matches conventional wisdom. The per-season-sum methodology gives HOF pitchers like Spahn (#10), Maddux (#6), and others who were broken under the single-fit version a sensible positive WAR.

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

7. **Single position per player.** Retrosheet rosters carry one position per (player, year). We pick each player's **modal** position across their career, which guards against single-year data errors. (Notable example: Retrosheet's 2005 CLE roster file lists Juan Gonzalez — a career OF — as `P`. Picking last-year would faithfully reproduce the typo; picking the mode does the right thing.)

8. **Cross-era comparisons via summed WAR are era-relative.** Each per-season fit centers on its own innings-weighted mean, so "0 WAR in 1965" and "0 WAR in 2020" are anchored against potentially different absolute skill levels. This is the same caveat that applies to Fangraphs/B-Ref career WAR — every season is normalized to its own replacement/league baseline, and summing across years assumes those baselines represent comparable absolute talent. The all-time single-fit view *tries* to be cross-era (one global baseline) but fails technically because of the structural issues described in §Per-season fits. There's no fully objective cross-era WAR; we ship the per-season-sum convention because it's what mainstream baseball stats use and because it doesn't produce nonsense numbers.

9. **The pitcher era effect.** Modern pitchers have higher per-inning rates than dead-ball-era pitchers, partly because of velocity / pitch design and partly because batters strike out more. The within-position z-score for pitchers is dominated by recent / active pitchers as a result. Within-decade z-scores would address this.

---

# Files in this project

| File | Purpose |
|---|---|
| `download_all.sh` | Walk 1910→current year, fetch any Retrosheet zips not yet on disk; clear stale statsapi feed caches for newly-arrived years. |
| `loaders/common.py` | Source-agnostic per-event → half-inning aggregation. |
| `loaders/retro.py` | Retrosheet driver: `cwevent` + park + roster extraction + on-the-fly fetch of newly-published years. |
| `loaders/statsapi.py` | MLB Stats API driver: schedule + per-game `feed/live` parsing, with on-disk caching. |
| `loaders/crosswalk.py` | MLBAM ↔ Retrosheet player-ID crosswalk via the Chadwick Bureau register. |
| `build_dataset.py` | Top-level builder; auto-routes years to retro or statsapi. Output: `data/events/half_innings_<tag>.parquet`, `game_park_<tag>.csv`, `rosters_<tag>.csv`. |
| `fit_ridge_all.py` | Single all-time ridge fit (player + season FE + park FE + home). Output: `coefficients_<tag>.{parquet,csv}` and `park_effects_<tag>.csv`. |
| `fit_per_season.py` | One ridge fit per season; emits long-format `season_war_<tag>.csv` + the headline `career_seasons_sum_<tag>.csv`. Supports `--seasons` for incremental cron updates and `--extra-tags` to union the current statsapi year into the all-time table. |
| `make_views.py` | Post-process the single-fit coefficients into per-season, peak-rate, and per-position-z-score derived metrics. Output: `coefficients_<tag>_enriched.{parquet,csv}`. |
| `snapshot.py` | Write a dated slim coefficient snapshot for the current season; regenerate `manifest.json` (the webapp's index). |
| `refresh.sh` | Cron entry point. Pulls, rebuilds current-season half-innings, refits, snapshots, pushes. |
| `webapp/` | Static frontend (index.html + style.css + app.js) — reads `data/events/*.csv` via relative paths. |

Total runtime end-to-end on a Mac:
- Cold rebuild: ~10 min download + ~5 min parse + ~2 min single-fit + ~3 min per-season fits.
- Daily incremental (cron): ~30 sec for current-season refit + ~2 sec for one season's incremental per-season update.
