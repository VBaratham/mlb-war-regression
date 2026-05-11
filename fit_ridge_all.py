"""All-time ridge regression: career-average per-player coefficient,
era-adjusted via per-season fixed effects, park-adjusted via per-park FEs.

For each half-inning we have:
  runs_scored ~ off_indicators + pit_indicators + fld_indicators
                + season_intercept + park_intercept + home_indicator

Player coefs (offense / pitcher / fielder) are career-long averages weighted
by where the player played their innings. Season fixed effects absorb era
scoring environment; park fixed effects absorb stadium effects (Coors,
Petco, Fenway, etc.) so a Rockies hitter isn't credited with park-inflated
production.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import Ridge

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="all",
                help="dataset tag built by build_dataset.py; reads half_innings_<tag>.parquet etc.")
args = ap.parse_args()
TAG = args.tag

ROOT = Path(__file__).parent
HALF = ROOT / "data" / "events" / f"half_innings_{TAG}.parquet"
# Back-compat: original retro pipeline wrote game_park.csv (no tag).
PARK = ROOT / "data" / "events" / f"game_park_{TAG}.csv"
if not PARK.exists():
    legacy = ROOT / "data" / "events" / "game_park.csv"
    if legacy.exists():
        PARK = legacy
ROSTERS = ROOT / "data" / "events" / f"rosters_{TAG}.csv"
OUT = ROOT / "data" / "events" / f"coefficients_{TAG}.parquet"

print("loading half-innings...")
df = pd.read_parquet(HALF)
print(f"{len(df):,} half-innings, {df.SEASON.nunique()} seasons "
      f"({df.SEASON.min()}-{df.SEASON.max()})")

# Merge park IDs.
print("loading park lookup...")
park_lookup = pd.read_csv(PARK)
df = df.merge(park_lookup, on="GAME_ID", how="left")
n_missing = df["PARK"].isna().sum()
if n_missing:
    print(f"  warning: {n_missing:,} half-innings missing park (filling with 'UNK')")
    df["PARK"] = df["PARK"].fillna("UNK")
print(f"  {df.PARK.nunique()} unique parks")

# Collect role player sets.
print("indexing players...")
off_players, pit_players, fld_players = set(), set(), set()
for s in df["batters"]:
    off_players.update(s.split("|"))
for s in df["pitchers"]:
    pit_players.update(s.split("|"))
for s in df["fielders"]:
    fld_players.update(s.split("|"))
for s in (off_players, pit_players, fld_players):
    s.discard("")
n_off, n_pit, n_fld = len(off_players), len(pit_players), len(fld_players)
off_idx = {p: i for i, p in enumerate(sorted(off_players))}
pit_idx = {p: i + n_off for i, p in enumerate(sorted(pit_players))}
fld_idx = {p: i + n_off + n_pit for i, p in enumerate(sorted(fld_players))}
print(f"  {n_off:,} batters, {n_pit:,} pitchers, {n_fld:,} fielders")

seasons = sorted(df.SEASON.unique())
season_idx = {s: i + n_off + n_pit + n_fld for i, s in enumerate(seasons)}
n_seasons = len(seasons)

parks = sorted(df.PARK.unique())
park_offset = n_off + n_pit + n_fld + n_seasons
park_idx = {p: i + park_offset for i, p in enumerate(parks)}
n_parks = len(parks)

home_col = park_offset + n_parks
n_cols = home_col + 1
print(f"  {n_seasons} season FE columns, {n_parks} park FE columns")
print(f"design matrix: {len(df):,} rows × {n_cols:,} cols")

# Build sparse matrix incrementally; for memory, pre-size arrays.
print("building sparse matrix...")
# Estimate nnz: each row has avg 4 batters + 1 pitcher + 8 fielders + 1 season + 1 park + ~0.5 home = ~16 entries
est_nnz = len(df) * 18
rows_arr = np.empty(est_nnz, dtype=np.int32)
cols_arr = np.empty(est_nnz, dtype=np.int32)
k = 0
for i, (b, p_, f_, h, ssn, prk) in enumerate(zip(
        df["batters"].values, df["pitchers"].values,
        df["fielders"].values, df["BAT_HOME_ID"].values,
        df["SEASON"].values, df["PARK"].values)):
    if b:
        for p in b.split("|"):
            if p:
                rows_arr[k] = i; cols_arr[k] = off_idx[p]; k += 1
    if p_:
        for p in p_.split("|"):
            if p:
                rows_arr[k] = i; cols_arr[k] = pit_idx[p]; k += 1
    if f_:
        for p in f_.split("|"):
            if p:
                rows_arr[k] = i; cols_arr[k] = fld_idx[p]; k += 1
    rows_arr[k] = i; cols_arr[k] = season_idx[ssn]; k += 1
    rows_arr[k] = i; cols_arr[k] = park_idx[prk]; k += 1
    if h == 1:
        rows_arr[k] = i; cols_arr[k] = home_col; k += 1
    if i % 200_000 == 0 and i > 0:
        print(f"  {i:,}/{len(df):,}")

rows_arr = rows_arr[:k]
cols_arr = cols_arr[:k]
data_arr = np.ones(k, dtype=np.float32)
X = csr_matrix((data_arr, (rows_arr, cols_arr)), shape=(len(df), n_cols))
y = df["runs_scored"].astype(np.float32).values
print(f"X: {X.shape}, nnz={X.nnz:,}")

# Per-role column scaling (pitchers get less effective regularization).
PIT_WEIGHT = 6.0
col_scale = np.ones(n_cols, dtype=np.float32)
col_scale[n_off:n_off + n_pit] = PIT_WEIGHT
# Season + park FE columns get larger weight -- they should be free to absorb
# era and stadium effects without ridge shrinking them toward zero.
SEASON_WEIGHT = 50.0
col_scale[n_off + n_pit + n_fld:park_offset] = SEASON_WEIGHT  # season cols
col_scale[park_offset:home_col] = SEASON_WEIGHT  # park cols
X = X.multiply(col_scale[np.newaxis, :]).tocsr()

# Fit ridge. With 6M+ rows, calibration sweep is expensive; pick a calibrated
# alpha based on previous experience and verify SD post-fit. With more data
# and trades providing within-team variation, less regularization is needed
# than the 2024-only run.
print("fitting ridge...")
ALPHA = 10_000.0  # scaled to dataset size; verify off_sd matches target post-fit
model = Ridge(alpha=ALPHA, fit_intercept=True, solver="sparse_cg",
              max_iter=2000, tol=1e-5)
model.fit(X, y)
print(f"alpha={ALPHA:.3g}, intercept={model.intercept_:.4f}, "
      f"in-sample R²={model.score(X, y):.4f}")

# Undo column scaling.
model.coef_ = model.coef_ * col_scale
coefs = model.coef_.copy()

off_slice = coefs[:n_off]
pit_slice = coefs[n_off:n_off + n_pit]
fld_slice = coefs[n_off + n_pit:n_off + n_pit + n_fld]
park_slice = coefs[park_offset:home_col]
print(f"raw off coef: mean={off_slice.mean():.4f} sd={off_slice.std():.4f}")
print(f"raw pit coef: mean={pit_slice.mean():.4f} sd={pit_slice.std():.4f}")
print(f"raw fld coef: mean={fld_slice.mean():.4f} sd={fld_slice.std():.4f}")
print(f"park coef: mean={park_slice.mean():.4f} sd={park_slice.std():.4f}")

# Save park effects table for inspection.
park_appearances = df.groupby("PARK").size().rename("half_innings")
park_df = pd.DataFrame({
    "park": parks,
    "park_runs_per_inning": park_slice,
    "half_innings": [park_appearances.get(p, 0) for p in parks],
})
# Filter to parks with substantial usage and show extremes.
substantial = park_df[park_df.half_innings >= 2000].sort_values("park_runs_per_inning")
print("\nbottom 10 hitter-friendliness parks (most pitcher-friendly, min 2000 HI):")
print(substantial.head(10).to_string(index=False))
print("\ntop 10 hitter-friendliness parks (most hitter-friendly, min 2000 HI):")
print(substantial.tail(10).to_string(index=False))
park_df.to_csv(ROOT / "data" / "events" / "park_effects.csv", index=False)

# Center role coefs (intercept absorbs the constant).
coefs[:n_off] = off_slice - off_slice.mean()
coefs[n_off:n_off + n_pit] = pit_slice - pit_slice.mean()
coefs[n_off + n_pit:n_off + n_pit + n_fld] = fld_slice - fld_slice.mean()

# Appearance counts.
print("counting appearances...")
off_counts = np.zeros(n_off, dtype=np.int32)
pit_counts = np.zeros(n_pit, dtype=np.int32)
fld_counts = np.zeros(n_fld, dtype=np.int32)
for b, p_, f_ in zip(df["batters"].values, df["pitchers"].values, df["fielders"].values):
    if b:
        for p in b.split("|"):
            if p:
                off_counts[off_idx[p]] += 1
    if p_:
        for p in p_.split("|"):
            if p:
                pit_counts[pit_idx[p] - n_off] += 1
    if f_:
        for p in f_.split("|"):
            if p:
                fld_counts[fld_idx[p] - n_off - n_pit] += 1

# Build player table.
print("assembling output...")
all_players = sorted(off_players | pit_players | fld_players)
recs = []
for p in all_players:
    off_c = coefs[off_idx[p]] if p in off_idx else np.nan
    pit_c = -coefs[pit_idx[p]] if p in pit_idx else np.nan
    fld_c = -coefs[fld_idx[p]] if p in fld_idx else np.nan
    recs.append({
        "player_id": p,
        "off_runs_per_inning": off_c,
        "pit_runs_per_inning": pit_c,
        "fld_runs_per_inning": fld_c,
        "off_innings": off_counts[off_idx[p]] if p in off_idx else 0,
        "pit_innings": pit_counts[pit_idx[p] - n_off] if p in pit_idx else 0,
        "fld_innings": fld_counts[fld_idx[p] - n_off - n_pit] if p in fld_idx else 0,
    })
out = pd.DataFrame(recs)
out["off_raa"] = out["off_runs_per_inning"] * out["off_innings"]
out["pit_raa"] = out["pit_runs_per_inning"] * out["pit_innings"]
out["fld_raa"] = out["fld_runs_per_inning"] * out["fld_innings"]
out["total_raa"] = out[["off_raa", "pit_raa", "fld_raa"]].sum(axis=1, skipna=True)
out["off_war"] = out["off_raa"] / 10.0
out["pit_war"] = out["pit_raa"] / 10.0
out["fld_war"] = out["fld_raa"] / 10.0
out["total_war"] = out["total_raa"] / 10.0

# Load unified roster table emitted by build_dataset.py. Keep most-recent
# listing per player for name and primary position.
print("loading rosters...")
if ROSTERS.exists():
    ros = pd.read_csv(ROSTERS)
else:
    # Back-compat path: fall back to globbing *.ROS under data/raw.
    print(f"  {ROSTERS} not found, falling back to raw *.ROS scan")
    roster_dir = ROOT / "data" / "raw"
    roster_frames = []
    for ydir in roster_dir.iterdir():
        if not (ydir.is_dir() and ydir.name.isdigit()):
            continue
        year = int(ydir.name)
        for f in ydir.glob("*.ROS"):
            try:
                r = pd.read_csv(f, header=None, usecols=[0, 1, 2, 5, 6],
                                names=["player_id", "last", "first", "team", "pos"])
                r["year"] = year
                roster_frames.append(r)
            except Exception:
                pass
    ros = pd.concat(roster_frames, ignore_index=True)
    ros["name"] = ros["first"].fillna("") + " " + ros["last"].fillna("")
    ros = ros[["player_id", "name", "team", "pos", "year"]]
# Pick primary position by mode across a player's career rather than the
# last roster listing -- guards against single-year data errors (e.g. the
# 2005 CLE roster file lists Juan Gonzalez, a career OF, as P). Ties broken
# by most recent appearance at that position.
ros_nonempty = ros.dropna(subset=["pos"])
ros_nonempty = ros_nonempty[ros_nonempty["pos"].astype(str).str.len() > 0]
pos_counts = (ros_nonempty
              .groupby(["player_id", "pos"])
              .agg(n=("year", "size"), recency=("year", "max"))
              .reset_index()
              .sort_values(["player_id", "n", "recency"], ascending=[True, False, False])
              .drop_duplicates("player_id", keep="first")
              [["player_id", "pos"]])

# Team: emit the modal team (replaces last-team) plus a pipe-joined
# chronological list of every team this player appeared on. The webapp
# bolds the modal entry in the multi-team display.
ros_teams = ros.dropna(subset=["team"])
ros_teams = ros_teams[ros_teams["team"].astype(str).str.len() > 0]
team_counts = (ros_teams
               .groupby(["player_id", "team"])
               .agg(n=("year", "size"), recency=("year", "max"))
               .reset_index()
               .sort_values(["player_id", "n", "recency"], ascending=[True, False, False])
               .drop_duplicates("player_id", keep="first")
               [["player_id", "team"]]
               .rename(columns={"team": "modal_team"}))
team_first_year = (ros_teams
                   .groupby(["player_id", "team"])["year"].min()
                   .reset_index()
                   .sort_values(["player_id", "year"]))
teams_chronological = (team_first_year
                       .groupby("player_id")["team"]
                       .apply(lambda s: "|".join(s))
                       .reset_index()
                       .rename(columns={"team": "teams"}))

# Most recent roster row for name + last_year, then swap in modal pos/team.
ros = ros.sort_values("year").drop_duplicates("player_id", keep="last")
ros = ros.drop(columns=["pos", "team"])
ros = (ros.merge(pos_counts, on="player_id", how="left")
          .merge(team_counts, on="player_id", how="left")
          .merge(teams_chronological, on="player_id", how="left"))
ros = ros.rename(columns={"year": "last_year", "modal_team": "team"})
out = out.merge(ros, on="player_id", how="left")

cols = ["player_id", "name", "last_year", "team", "teams", "pos",
        "off_runs_per_inning", "pit_runs_per_inning", "fld_runs_per_inning",
        "off_innings", "pit_innings", "fld_innings",
        "off_raa", "pit_raa", "fld_raa", "total_raa",
        "off_war", "pit_war", "fld_war", "total_war"]
out = out[cols]

print(f"\ntop 25 all-time by total_war:")
print(out.sort_values("total_war", ascending=False).head(25).to_string(index=False))
print(f"\ntop 25 batters (off_war), min 1500 off_innings:")
print(out[out.off_innings >= 1500].sort_values("off_war", ascending=False).head(25)
      [["name", "last_year", "team", "pos", "off_runs_per_inning", "off_innings", "off_war"]].to_string(index=False))
print(f"\ntop 25 pitchers (pit_war), pos=P, min 500 pit_innings:")
print(out[(out.pos == "P") & (out.pit_innings >= 500)].sort_values("pit_war", ascending=False).head(25)
      [["name", "last_year", "team", "pit_runs_per_inning", "pit_innings", "pit_war"]].to_string(index=False))
print(f"\ntop 25 fielders (fld_war), excludes pitchers, min 2000 fld_innings:")
print(out[(out.pos != "P") & (out.fld_innings >= 2000)].sort_values("fld_war", ascending=False).head(25)
      [["name", "last_year", "team", "pos", "fld_runs_per_inning", "fld_innings", "fld_war"]].to_string(index=False))

out.to_parquet(OUT, index=False)
out.sort_values("total_war", ascending=False).to_csv(OUT.with_suffix(".csv"), index=False)
print(f"\nwrote {OUT}")
print(f"wrote {OUT.with_suffix('.csv')}")
