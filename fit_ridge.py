"""Ridge regression: runs scored per half-inning ~ off + pit + fld indicators.

Each player gets up to three coefficients:
  off: contribution per half-inning when this player is among the batters
  pit: contribution per half-inning when this player pitched
  fld: contribution per half-inning when this player fielded a non-pitcher position

Pitchers and fielders are split because the pitcher is involved in every play
while each fielder handles only ~1/8 of balls in play. Lumping them together
forces ridge to spread credit equally across all 9 defenders, which makes
pitcher coefficients ~10x too small.

Coefficients are in units of runs per half-inning. We sign-flip pit and fld at
the end so "higher = better" for all three.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_score

ROOT = Path(__file__).parent
HALF = ROOT / "data" / "events" / "half_innings_2024.parquet"
OUT = ROOT / "data" / "events" / "coefficients_2024.parquet"

df = pd.read_parquet(HALF)
print(f"loaded {len(df):,} half-innings")

# Collect unique players in each role.
off_players, pit_players, fld_players = set(), set(), set()
for s in df["batters"]:
    off_players.update(s.split("|"))
for s in df["pitchers"]:
    pit_players.update(s.split("|"))
for s in df["fielders"]:
    fld_players.update(s.split("|"))
for s in (off_players, pit_players, fld_players):
    s.discard("")
n_off = len(off_players)
n_pit = len(pit_players)
n_fld = len(fld_players)
off_idx = {p: i for i, p in enumerate(sorted(off_players))}
pit_idx = {p: i + n_off for i, p in enumerate(sorted(pit_players))}
fld_idx = {p: i + n_off + n_pit for i, p in enumerate(sorted(fld_players))}
print(f"{n_off:,} batters, {n_pit:,} pitchers, {n_fld:,} fielders")

home_col = n_off + n_pit + n_fld
n_cols = home_col + 1

# Build sparse design matrix.
rows, cols = [], []
for i, (b, p_, f_, h) in enumerate(zip(df["batters"], df["pitchers"],
                                       df["fielders"], df["BAT_HOME_ID"])):
    for p in b.split("|"):
        if p:
            rows.append(i); cols.append(off_idx[p])
    for p in p_.split("|"):
        if p:
            rows.append(i); cols.append(pit_idx[p])
    for p in f_.split("|"):
        if p:
            rows.append(i); cols.append(fld_idx[p])
    if h == 1:
        rows.append(i); cols.append(home_col)
data = np.ones(len(rows), dtype=np.float32)
X = csr_matrix((data, (rows, cols)), shape=(len(df), n_cols))
y = df["runs_scored"].astype(np.float32).values
print(f"X: {X.shape}, nnz={X.nnz:,}")

# Per-role column scaling. Scaling column j by factor c is equivalent to
# reducing its ridge penalty by c² -- so larger weight = less regularization.
# Pitchers have a much wider true effect range than batters or fielders
# (a Cy Young winner is ~0.25 runs/inning, vs ~0.05 for a top hitter), so we
# give pitcher columns a higher weight. The factor below is calibrated below
# to land pitcher SD near 0.1 runs/inning, which matches FanGraphs RA/9 spread.
PIT_WEIGHT = 6.0
col_scale = np.ones(n_cols, dtype=np.float32)
col_scale[n_off:n_off + n_pit] = PIT_WEIGHT
# Apply scaling: multiply each column by its weight.
X = X.multiply(col_scale[np.newaxis, :]).tocsr()

# Pick alpha to calibrate the spread of player coefficients to a realistic prior.
# In MLB, the SD of run-value-above-average across regulars is roughly
#   ~20 runs / 700 batting innings = 0.03 runs/inning
# Standard CV picks an alpha that's too low because a model can predict half-
# inning runs well by memorizing team identity (same lineups recur all season),
# but those team-level effects get absorbed into individual coefficients --
# producing implausibly large per-player estimates. We sweep alpha and pick
# the value where the SD of offensive coefficients is closest to a target.
groups = df["GAME_ID"].values
splits = list(GroupKFold(n_splits=5).split(X, y, groups))
TARGET_OFF_SD = 0.03  # runs/inning; calibrated to MLB regulars

alphas = np.logspace(1, 5, 9)
sweep_records = []
print(f"{'alpha':>10}  {'CV R²':>7}  {'off_sd':>7}  {'pit_sd':>7}  {'fld_sd':>7}")
for a in alphas:
    m = Ridge(alpha=a, fit_intercept=True).fit(X, y)
    cv_r2 = cross_val_score(Ridge(alpha=a, fit_intercept=True), X, y,
                            cv=splits, scoring="r2").mean()
    off_sd = m.coef_[:n_off].std()
    pit_sd = m.coef_[n_off:n_off + n_pit].std()
    fld_sd = m.coef_[n_off + n_pit:n_off + n_pit + n_fld].std()
    sweep_records.append((a, cv_r2, off_sd, pit_sd, fld_sd))
    print(f"{a:>10.3g}  {cv_r2:>7.4f}  {off_sd:>7.4f}  {pit_sd:>7.4f}  {fld_sd:>7.4f}")

chosen_alpha = min(sweep_records, key=lambda r: abs(r[2] - TARGET_OFF_SD))[0]
print(f"\nchosen alpha={chosen_alpha:.3g} (off_sd closest to target {TARGET_OFF_SD})")
model = Ridge(alpha=chosen_alpha, fit_intercept=True).fit(X, y)
print(f"intercept={model.intercept_:.4f}, in-sample R²={model.score(X, y):.4f}")

# Undo column scaling so reported coefficients are in original "runs per
# inning when this player appears" units.
model.coef_ = model.coef_ * col_scale

coefs = model.coef_.copy()

# Center coefficients within each role so "0 = average player". The raw ridge
# fit pushes the intercept very negative to absorb the baseline number of
# batters/pitchers/fielders per row, leaving each player's coef representing
# "average major leaguer" rather than "deviation from average". Subtracting
# the role mean re-anchors "0" without changing predictions.
off_slice = coefs[:n_off]
pit_slice = coefs[n_off:n_off + n_pit]
fld_slice = coefs[n_off + n_pit:n_off + n_pit + n_fld]
print(f"raw off coef: mean={off_slice.mean():.4f} sd={off_slice.std():.4f}")
print(f"raw pit coef: mean={pit_slice.mean():.4f} sd={pit_slice.std():.4f}")
print(f"raw fld coef: mean={fld_slice.mean():.4f} sd={fld_slice.std():.4f}")
coefs[:n_off] = off_slice - off_slice.mean()
coefs[n_off:n_off + n_pit] = pit_slice - pit_slice.mean()
coefs[n_off + n_pit:n_off + n_pit + n_fld] = fld_slice - fld_slice.mean()

# Compute appearance counts for filtering / context.
off_counts = np.zeros(n_off, dtype=int)
pit_counts = np.zeros(n_pit, dtype=int)
fld_counts = np.zeros(n_fld, dtype=int)
for b, p_, f_ in zip(df["batters"], df["pitchers"], df["fielders"]):
    for p in b.split("|"):
        if p:
            off_counts[off_idx[p]] += 1
    for p in p_.split("|"):
        if p:
            pit_counts[pit_idx[p] - n_off] += 1
    for p in f_.split("|"):
        if p:
            fld_counts[fld_idx[p] - n_off - n_pit] += 1

# Assemble per-player table. Sign-flip pit/fld so higher = better at preventing runs.
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

# Season totals: runs above average over the player's appearances.
out["off_raa"] = out["off_runs_per_inning"] * out["off_innings"]
out["pit_raa"] = out["pit_runs_per_inning"] * out["pit_innings"]
out["fld_raa"] = out["fld_runs_per_inning"] * out["fld_innings"]
out["total_raa"] = out[["off_raa", "pit_raa", "fld_raa"]].sum(axis=1, skipna=True)
out["off_war"] = out["off_raa"] / 10.0
out["pit_war"] = out["pit_raa"] / 10.0
out["fld_war"] = out["fld_raa"] / 10.0
out["total_war"] = out["total_raa"] / 10.0

# Join roster files for name + primary position.
roster_dir = ROOT / "data" / "raw"
roster_frames = []
for f in roster_dir.glob("*2024.ROS"):
    r = pd.read_csv(f, header=None,
                    names=["player_id", "last", "first", "bats", "throws", "team", "pos"])
    roster_frames.append(r)
ros = pd.concat(roster_frames, ignore_index=True)
ros["name"] = ros["first"] + " " + ros["last"]
# A player can be on multiple teams (trades); keep the first listing for primary position.
ros = ros.drop_duplicates("player_id", keep="first")[["player_id", "name", "team", "pos"]]
out = out.merge(ros, on="player_id", how="left")

cols = ["player_id", "name", "team", "pos",
        "off_runs_per_inning", "pit_runs_per_inning", "fld_runs_per_inning",
        "off_innings", "pit_innings", "fld_innings",
        "off_raa", "pit_raa", "fld_raa", "total_raa",
        "off_war", "pit_war", "fld_war", "total_war"]
out = out[cols]

print(f"\ntop 15 by total_war (alpha={chosen_alpha:.3g}):")
print(out.sort_values("total_war", ascending=False).head(15).to_string(index=False))
print(f"\nbottom 10 by total_war:")
print(out.sort_values("total_war").head(10).to_string(index=False))
print(f"\ntop 10 batters (off_war), min 400 off_innings:")
print(out[out.off_innings >= 400].sort_values("off_war", ascending=False).head(10)
      [["name", "team", "pos", "off_runs_per_inning", "off_innings", "off_war"]].to_string(index=False))
print(f"\ntop 10 pitchers (pit_war), pos=P, min 100 pit_innings:")
print(out[(out.pos == "P") & (out.pit_innings >= 100)].sort_values("pit_war", ascending=False).head(10)
      [["name", "team", "pit_runs_per_inning", "pit_innings", "pit_war"]].to_string(index=False))
print(f"\ntop 10 fielders (fld_war), excludes pitchers, min 400 fld_innings:")
print(out[(out.pos != "P") & (out.fld_innings >= 400)].sort_values("fld_war", ascending=False).head(10)
      [["name", "team", "pos", "fld_runs_per_inning", "fld_innings", "fld_war"]].to_string(index=False))

out.to_parquet(OUT, index=False)
out.sort_values("total_war", ascending=False).to_csv(OUT.with_suffix(".csv"), index=False)
print(f"\nwrote {OUT}")
print(f"wrote {OUT.with_suffix('.csv')}")
