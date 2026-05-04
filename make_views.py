"""Post-process coefficients_all into two derived views:

  1. Per-season rates: career WAR divided by seasons-played, so a short or
     ongoing career is not penalized for low cumulative innings. We compute
     seasons-played by counting distinct SEASON values where the player
     appeared in any role.

  2. Per-position normalization: within each primary position bucket, rank
     players and compute a z-score against that position's regulars. A SS
     hitting at +0.04 runs/inning is more impressive than a 1B at the same
     rate, so position-relative scores answer "best-at-position" questions.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
COEFS = ROOT / "data" / "events" / "coefficients_all.parquet"
HALF = ROOT / "data" / "events" / "half_innings_all.parquet"
OUT_DIR = ROOT / "data" / "events"

print("loading...")
coefs = pd.read_parquet(COEFS)
half = pd.read_parquet(HALF, columns=["SEASON", "batters", "pitchers", "fielders"])

# Compute seasons-played per player by scanning the half-innings.
print("counting seasons per player...")
seasons_by_player = {}
for season, b, p, f in zip(half.SEASON.values, half.batters.values,
                           half.pitchers.values, half.fielders.values):
    if b:
        for pid in b.split("|"):
            if pid:
                seasons_by_player.setdefault(pid, set()).add(season)
    if p:
        for pid in p.split("|"):
            if pid:
                seasons_by_player.setdefault(pid, set()).add(season)
    if f:
        for pid in f.split("|"):
            if pid:
                seasons_by_player.setdefault(pid, set()).add(season)
seasons_df = pd.DataFrame([
    {"player_id": pid,
     "seasons_played": len(s),
     "first_year": min(s),
     "last_year_played": max(s)}
    for pid, s in seasons_by_player.items()
])
coefs = coefs.merge(seasons_df, on="player_id", how="left")

# 1a. Per-calendar-season: total / seasons-played. Penalizes players who had
# injury-shortened seasons since each season counts as 1 regardless of playing
# time. Useful for "average annual contribution" view.
coefs["off_war_per_season"] = coefs["off_war"] / coefs["seasons_played"]
coefs["pit_war_per_season"] = coefs["pit_war"] / coefs["seasons_played"]
coefs["fld_war_per_season"] = coefs["fld_war"] / coefs["seasons_played"]
coefs["total_war_per_season"] = coefs["total_war"] / coefs["seasons_played"]

# 1b. Per-full-season rate: scaled to a notional "full season" of innings.
# A regular MLB hitter accumulates ~700 batting half-inning appearances and
# ~1300 fielding half-innings per season; a starting pitcher ~200 pitching
# half-innings. This gives a peak-rate view that doesn't punish injury time
# or short careers -- it asks "if this player played a full healthy season,
# how many WAR would he add?"
PER_SEASON_OFF_HI = 700
PER_SEASON_FLD_HI = 1300
PER_SEASON_PIT_HI = 200  # ~200 IP equivalent for a full-season starter
coefs["off_war_full_season_rate"] = (
    coefs["off_runs_per_inning"] * PER_SEASON_OFF_HI / 10.0)
coefs["pit_war_full_season_rate"] = (
    coefs["pit_runs_per_inning"] * PER_SEASON_PIT_HI / 10.0)
coefs["fld_war_full_season_rate"] = (
    coefs["fld_runs_per_inning"] * PER_SEASON_FLD_HI / 10.0)

# 2. Position-normalized scores. We compute z-score of off_runs_per_inning
# vs. that position's regulars, and same for fld_runs_per_inning. Pitchers
# are normalized separately on pit_runs_per_inning.
def position_zscores(df, value_col, innings_col, group_col, min_innings):
    z = pd.Series(np.nan, index=df.index, dtype=np.float64)
    for grp, sub in df.groupby(group_col):
        regulars = sub[sub[innings_col] >= min_innings]
        if len(regulars) < 5:
            continue
        mu = regulars[value_col].mean()
        sd = regulars[value_col].std()
        if sd > 0:
            z.loc[sub.index] = (sub[value_col] - mu) / sd
    return z

# Position grouping for offense/fielding (pitchers separate).
coefs["pos_group"] = coefs["pos"].fillna("UNK")
# Treat OF together and treat DH separately (DH has no fielding component).
coefs["off_pos_z"] = position_zscores(
    coefs, "off_runs_per_inning", "off_innings", "pos_group", min_innings=1500)
coefs["fld_pos_z"] = position_zscores(
    coefs, "fld_runs_per_inning", "fld_innings", "pos_group", min_innings=2000)
coefs["pit_pos_z"] = position_zscores(
    coefs, "pit_runs_per_inning", "pit_innings", "pos_group", min_innings=500)

# Save enriched table.
ENRICHED = OUT_DIR / "coefficients_all_enriched.parquet"
coefs.to_parquet(ENRICHED, index=False)
coefs.sort_values("total_war_per_season", ascending=False).to_csv(
    OUT_DIR / "coefficients_all_enriched.csv", index=False)
print(f"wrote {ENRICHED}")

# Display: per-calendar-season leaders.
print("\n" + "=" * 80)
print("TOP 25 BY total_war_per_season  (calendar-season avg; min 5 seasons)")
print("=" * 80)
qual = (coefs["seasons_played"] >= 5) & (
    coefs[["off_innings", "pit_innings", "fld_innings"]].sum(axis=1) >= 1500)
view = coefs[qual].sort_values("total_war_per_season", ascending=False).head(25)
print(view[["name", "first_year", "last_year_played", "pos", "seasons_played",
            "total_war", "total_war_per_season"]]
      .to_string(index=False))

# Per-full-season rate leaders (hitters).
print("\n" + "=" * 80)
print("TOP 25 BATTERS BY off_war_full_season_rate  (peak rate; min 1500 off_innings)")
print("=" * 80)
view = coefs[coefs.off_innings >= 1500].sort_values(
    "off_war_full_season_rate", ascending=False).head(25)
print(view[["name", "first_year", "last_year_played", "pos", "off_innings",
            "off_runs_per_inning", "off_war_full_season_rate", "off_war"]]
      .to_string(index=False))

print("\n" + "=" * 80)
print("TOP 25 PITCHERS BY pit_war_full_season_rate  (peak rate; min 500 pit_innings)")
print("=" * 80)
view = coefs[(coefs.pos == "P") & (coefs.pit_innings >= 500)].sort_values(
    "pit_war_full_season_rate", ascending=False).head(25)
print(view[["name", "first_year", "last_year_played", "pit_innings",
            "pit_runs_per_inning", "pit_war_full_season_rate", "pit_war"]]
      .to_string(index=False))

# Per-position rankings.
print("\n" + "=" * 80)
print("TOP 10 BY POSITION  (offensive z-score, ranked within position)")
print("=" * 80)
for pos in ["C", "1B", "2B", "3B", "SS", "OF", "DH"]:
    sub = coefs[(coefs["pos"] == pos) & (coefs["off_innings"] >= 1500)]
    sub = sub.sort_values("off_pos_z", ascending=False).head(10)
    if sub.empty:
        continue
    print(f"\n--- {pos} (top 10 hitters at position) ---")
    print(sub[["name", "first_year", "last_year_played", "off_innings",
               "off_runs_per_inning", "off_pos_z", "off_war",
               "off_war_per_season"]]
          .to_string(index=False))

print("\n--- P (top 10 pitchers, by within-position z) ---")
sub = coefs[(coefs["pos"] == "P") & (coefs["pit_innings"] >= 500)]
sub = sub.sort_values("pit_pos_z", ascending=False).head(10)
print(sub[["name", "first_year", "last_year_played", "pit_innings",
           "pit_runs_per_inning", "pit_pos_z", "pit_war",
           "pit_war_per_season"]]
      .to_string(index=False))

# Per-position fielders.
print("\n" + "=" * 80)
print("TOP 5 FIELDERS BY POSITION  (fielding z-score)")
print("=" * 80)
for pos in ["C", "1B", "2B", "3B", "SS", "OF"]:
    sub = coefs[(coefs["pos"] == pos) & (coefs["fld_innings"] >= 2000)]
    sub = sub.sort_values("fld_pos_z", ascending=False).head(5)
    if sub.empty:
        continue
    print(f"\n--- {pos} (top 5 by glove) ---")
    print(sub[["name", "first_year", "last_year_played", "fld_innings",
               "fld_runs_per_inning", "fld_pos_z", "fld_war"]]
          .to_string(index=False))
