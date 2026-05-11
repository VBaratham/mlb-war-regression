"""Fit ridge per individual season and assemble a long-format season_war table.

Per-season fitting (rather than slicing the all-time fit) gives each player
a separate coefficient for each season they played -- so we can show
"Mays 1965 was a +9 WAR season" instead of treating every year as the
career-average rate. Each season picks its own alpha by matching off_sd
to a target (0.03 runs/inning), same calibration trick used in fit_ridge.py.

Output: data/events/season_war_<tag>.parquet (and .csv) with one row per
(season, player_id) where the player appeared in any role that season:

  season, player_id, name, pos, team,
  off_innings, pit_innings, fld_innings,
  off_runs_per_inning, pit_runs_per_inning, fld_runs_per_inning,
  off_war, pit_war, fld_war, total_war,
  alpha_chosen   -- diagnostic

Usage:
    # First-time full build:
    python3 fit_per_season.py --tag all
    # Cron incremental: refit only the current season, replacing those rows:
    python3 fit_per_season.py --tag all --seasons 2026
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

ROOT = Path(__file__).parent
EVENTS = ROOT / "data" / "events"

TARGET_OFF_SD = 0.03           # calibration target for batter spread
PIT_WEIGHT = 6.0               # per-role column scale (matches fit_ridge_all.py)
PARK_WEIGHT = 50.0
RUNS_PER_WIN = 10.0
ALPHA_GRID = np.logspace(1.5, 5, 8)

MIN_HALF_INNINGS_PER_SEASON = 1000   # below this we skip the season entirely


def _index_role_players(rows):
    """Return (off_idx, pit_idx, fld_idx, n_off, n_pit, n_fld). Indices number
    players starting at 0 for offense, then pit, then fld."""
    off_set, pit_set, fld_set = set(), set(), set()
    for s in rows["batters"]:
        off_set.update(s.split("|"))
    for s in rows["pitchers"]:
        pit_set.update(s.split("|"))
    for s in rows["fielders"]:
        fld_set.update(s.split("|"))
    for s in (off_set, pit_set, fld_set):
        s.discard("")
    n_off, n_pit, n_fld = len(off_set), len(pit_set), len(fld_set)
    off_idx = {p: i for i, p in enumerate(sorted(off_set))}
    pit_idx = {p: i + n_off for i, p in enumerate(sorted(pit_set))}
    fld_idx = {p: i + n_off + n_pit for i, p in enumerate(sorted(fld_set))}
    return off_idx, pit_idx, fld_idx, n_off, n_pit, n_fld


def _build_design(rows, off_idx, pit_idx, fld_idx, parks):
    n_off, n_pit, n_fld = len(off_idx), len(pit_idx), len(fld_idx)
    park_offset = n_off + n_pit + n_fld
    n_parks = len(parks)
    park_idx = {p: i + park_offset for i, p in enumerate(sorted(parks))}
    home_col = park_offset + n_parks
    n_cols = home_col + 1

    # Upper bound on nnz: ~5 batters + 1 pitcher + 8 fielders + 1 park + 0.5 home
    est_nnz = len(rows) * 18
    rs = np.empty(est_nnz, dtype=np.int32)
    cs = np.empty(est_nnz, dtype=np.int32)
    k = 0
    batters = rows["batters"].values
    pitchers = rows["pitchers"].values
    fielders = rows["fielders"].values
    homes = rows["BAT_HOME_ID"].values
    parks_col = rows["PARK"].values
    for i in range(len(rows)):
        b = batters[i]
        if b:
            for p in b.split("|"):
                if p:
                    rs[k] = i; cs[k] = off_idx[p]; k += 1
        p_ = pitchers[i]
        if p_:
            for p in p_.split("|"):
                if p:
                    rs[k] = i; cs[k] = pit_idx[p]; k += 1
        f_ = fielders[i]
        if f_:
            for p in f_.split("|"):
                if p:
                    rs[k] = i; cs[k] = fld_idx[p]; k += 1
        rs[k] = i; cs[k] = park_idx[parks_col[i]]; k += 1
        if homes[i] == 1:
            rs[k] = i; cs[k] = home_col; k += 1
    rs = rs[:k]; cs = cs[:k]
    data = np.ones(k, dtype=np.float32)
    X = csr_matrix((data, (rs, cs)), shape=(len(rows), n_cols))

    col_scale = np.ones(n_cols, dtype=np.float32)
    col_scale[n_off:n_off + n_pit] = PIT_WEIGHT
    col_scale[park_offset:home_col] = PARK_WEIGHT
    X = X.multiply(col_scale[np.newaxis, :]).tocsr()
    return X, col_scale, n_cols, park_offset, home_col


def _counts(rows, off_idx, pit_idx, fld_idx):
    n_off, n_pit, n_fld = len(off_idx), len(pit_idx), len(fld_idx)
    off_counts = np.zeros(n_off, dtype=np.int32)
    pit_counts = np.zeros(n_pit, dtype=np.int32)
    fld_counts = np.zeros(n_fld, dtype=np.int32)
    for b, p_, f_ in zip(rows["batters"].values, rows["pitchers"].values, rows["fielders"].values):
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
    return off_counts, pit_counts, fld_counts


def _weighted_mean(coefs, weights):
    w = weights.astype(np.float64)
    s = w.sum()
    return float((coefs.astype(np.float64) * w).sum() / s) if s > 0 else 0.0


def fit_one_season(rows, season):
    """Fit one season's half-innings; return list of per-player dict rows."""
    off_idx, pit_idx, fld_idx, n_off, n_pit, n_fld = _index_role_players(rows)
    parks = sorted(rows["PARK"].unique())
    X, col_scale, n_cols, park_offset, home_col = _build_design(
        rows, off_idx, pit_idx, fld_idx, parks
    )
    y = rows["runs_scored"].astype(np.float32).values

    # Alpha sweep: pick alpha whose off_sd is closest to TARGET.
    best = None
    for a in ALPHA_GRID:
        m = Ridge(alpha=a, fit_intercept=True, solver="sparse_cg", max_iter=1000, tol=1e-5)
        m.fit(X, y)
        coef = m.coef_ * col_scale
        off_sd = coef[:n_off].std() if n_off > 0 else 0.0
        score = abs(off_sd - TARGET_OFF_SD)
        if best is None or score < best[0]:
            best = (score, a, coef, off_sd)
    _, alpha, coefs, off_sd = best

    off_slice = coefs[:n_off].copy()
    pit_slice = coefs[n_off:n_off + n_pit].copy()
    fld_slice = coefs[n_off + n_pit:n_off + n_pit + n_fld].copy()

    off_counts, pit_counts, fld_counts = _counts(rows, off_idx, pit_idx, fld_idx)

    off_center = _weighted_mean(off_slice, off_counts)
    pit_center = _weighted_mean(pit_slice, pit_counts) if n_pit else 0.0
    fld_center = _weighted_mean(fld_slice, fld_counts) if n_fld else 0.0
    off_centered = off_slice - off_center
    pit_centered = pit_slice - pit_center
    fld_centered = fld_slice - fld_center

    all_players = set(off_idx) | set(pit_idx) | set(fld_idx)
    out = []
    for p in all_players:
        off_c = off_centered[off_idx[p]] if p in off_idx else np.nan
        pit_c = -pit_centered[pit_idx[p] - n_off] if p in pit_idx else np.nan
        fld_c = -fld_centered[fld_idx[p] - n_off - n_pit] if p in fld_idx else np.nan
        off_inn = int(off_counts[off_idx[p]]) if p in off_idx else 0
        pit_inn = int(pit_counts[pit_idx[p] - n_off]) if p in pit_idx else 0
        fld_inn = int(fld_counts[fld_idx[p] - n_off - n_pit]) if p in fld_idx else 0
        off_raa = (off_c * off_inn) if not np.isnan(off_c) else 0.0
        pit_raa = (pit_c * pit_inn) if not np.isnan(pit_c) else 0.0
        fld_raa = (fld_c * fld_inn) if not np.isnan(fld_c) else 0.0
        total_raa = off_raa + pit_raa + fld_raa
        out.append({
            "season": int(season),
            "player_id": p,
            "off_innings": off_inn,
            "pit_innings": pit_inn,
            "fld_innings": fld_inn,
            "off_runs_per_inning": float(off_c) if not np.isnan(off_c) else np.nan,
            "pit_runs_per_inning": float(pit_c) if not np.isnan(pit_c) else np.nan,
            "fld_runs_per_inning": float(fld_c) if not np.isnan(fld_c) else np.nan,
            "off_war": off_raa / RUNS_PER_WIN,
            "pit_war": pit_raa / RUNS_PER_WIN,
            "fld_war": fld_raa / RUNS_PER_WIN,
            "total_war": total_raa / RUNS_PER_WIN,
            "alpha_chosen": float(alpha),
        })
    return out, off_sd


def _parse_seasons(spec):
    if not spec:
        return None
    out = set()
    for chunk in spec.split(","):
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(chunk))
    return out


def _load_tag(tag: str):
    """Return (half_innings_df, parks_df) for a single tag."""
    half = pd.read_parquet(EVENTS / f"half_innings_{tag}.parquet")
    half["GAME_ID"] = half["GAME_ID"].astype(str)
    park_path = EVENTS / f"game_park_{tag}.csv"
    if not park_path.exists() and tag == "all":
        park_path = EVENTS / "game_park.csv"
    parks = pd.read_csv(park_path, dtype={"GAME_ID": str})
    return half, parks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all",
                    help="base dataset tag; output goes to season_war_<tag>.csv")
    ap.add_argument("--extra-tags",
                    help="comma-separated additional tags whose half-innings "
                         "are unioned into the base before fitting. Used to "
                         "pull current-season statsapi data (tag=2026) into "
                         "the all-time table without rebuilding half_innings_all.")
    ap.add_argument("--seasons",
                    help="comma-separated list and/or YYYY-YYYY ranges. "
                         "Default: every season present in the tag's data.")
    args = ap.parse_args()
    TAG = args.tag

    print(f"loading half_innings_{TAG}.parquet...")
    half, parks = _load_tag(TAG)
    print(f"  {len(half):,} half-innings, {half.SEASON.min()}-{half.SEASON.max()}")

    if args.extra_tags:
        for extra in args.extra_tags.split(","):
            extra = extra.strip()
            if not extra:
                continue
            print(f"  unioning extra half_innings_{extra}.parquet...")
            extra_half, extra_parks = _load_tag(extra)
            half = pd.concat([half, extra_half], ignore_index=True)
            parks = pd.concat([parks, extra_parks], ignore_index=True).drop_duplicates("GAME_ID")
            print(f"  +{len(extra_half):,} half-innings (now "
                  f"{half.SEASON.min()}-{half.SEASON.max()})")

    half = half.merge(parks, on="GAME_ID", how="left")
    half["PARK"] = half["PARK"].fillna("UNK")

    target = _parse_seasons(args.seasons)
    seasons_present = sorted(half.SEASON.unique().tolist())
    if target is not None:
        seasons_to_fit = [s for s in seasons_present if s in target]
        if not seasons_to_fit:
            raise SystemExit(f"no overlap between --seasons {sorted(target)} and data")
    else:
        seasons_to_fit = seasons_present
    print(f"fitting {len(seasons_to_fit)} season(s): "
          f"{seasons_to_fit[0]}..{seasons_to_fit[-1]}")

    out_rows = []
    t_start = time.time()
    for i, season in enumerate(seasons_to_fit):
        sub = half[half.SEASON == season]
        if len(sub) < MIN_HALF_INNINGS_PER_SEASON:
            print(f"  {season}: {len(sub)} HI (skipping; under {MIN_HALF_INNINGS_PER_SEASON})")
            continue
        t0 = time.time()
        rows, off_sd = fit_one_season(sub, season)
        dt = time.time() - t0
        out_rows.extend(rows)
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(seasons_to_fit) - i - 1)
        print(f"  {season}: {len(sub):>7,} HI -> {len(rows):>5,} players  "
              f"off_sd={off_sd:.4f}  alpha={rows[0]['alpha_chosen']:.0f}  "
              f"{dt:.1f}s  (elapsed {elapsed:.0f}s, eta {eta:.0f}s)")

    new = pd.DataFrame(out_rows)
    out_parquet = EVENTS / f"season_war_{TAG}.parquet"
    out_csv = EVENTS / f"season_war_{TAG}.csv"

    if target is not None and (out_parquet.exists() or out_csv.exists()):
        # Incremental update: drop existing rows for the seasons we just fit,
        # then append the new rows. Fall back to CSV if the parquet was
        # gitignored away (fresh clone) but the CSV is checked in.
        existing = (pd.read_parquet(out_parquet) if out_parquet.exists()
                    else pd.read_csv(out_csv, low_memory=False))
        # Drop metadata columns from existing rows so the re-merge below
        # doesn't produce _x/_y duplicates.
        for c in ("name", "pos", "team"):
            if c in existing.columns:
                existing = existing.drop(columns=c)
        keep = existing[~existing["season"].isin(seasons_to_fit)]
        merged = pd.concat([keep, new], ignore_index=True).sort_values(["season", "player_id"])
    else:
        merged = new.sort_values(["season", "player_id"])

    # Attach name + pos + team. Union ALL coefficients_*_enriched.parquet files
    # so we pick up retro player metadata from the all-time fit *and* current-
    # season statsapi metadata (which lives in coefficients_<current>_enriched
    # and isn't in the all-time table because half_innings_all is retro-only).
    metas = []
    for f in sorted(EVENTS.glob("coefficients_*_enriched.parquet")):
        try:
            df = pd.read_parquet(f, columns=["player_id", "name", "pos", "team"])
            metas.append(df)
        except Exception as e:
            print(f"  skipping {f.name}: {e}")
    if metas:
        roster = pd.concat(metas, ignore_index=True).drop_duplicates("player_id", keep="first")
        merged = merged.merge(roster, on="player_id", how="left")

    merged.to_parquet(out_parquet, index=False)
    merged.to_csv(out_csv, index=False)
    print(f"\nwrote {out_parquet} ({len(merged):,} rows)")
    print(f"wrote {out_csv}")

    # Career roll-up: per-season-summed WAR is the headline cross-era number
    # the webapp displays (more robust than the single all-time fit, which
    # has cross-era identifiability issues for pitchers). One row per player.
    print("\nrolling up career season-sums...")
    sums = (merged.groupby("player_id", as_index=False)
                  .agg(off_war=("off_war", "sum"),
                       pit_war=("pit_war", "sum"),
                       fld_war=("fld_war", "sum"),
                       total_war=("total_war", "sum"),
                       off_innings=("off_innings", "sum"),
                       pit_innings=("pit_innings", "sum"),
                       fld_innings=("fld_innings", "sum"),
                       seasons_played=("season", "nunique"),
                       first_year=("season", "min"),
                       last_year_played=("season", "max"),
                       peak_season_war=("total_war", "max")))
    # "Best annual contribution" rate: career total / seasons played.
    sums["war_per_season"] = sums["total_war"] / sums["seasons_played"].clip(lower=1)
    # Bring along name / pos / team / teams from the existing enriched all-time
    # coefficients file (single source of truth for player metadata).
    coefs_path = EVENTS / f"coefficients_{TAG}_enriched.parquet"
    if coefs_path.exists():
        meta = pd.read_parquet(coefs_path)[
            ["player_id", "name", "pos", "team", "teams"]
        ]
        sums = sums.merge(meta, on="player_id", how="left")

    career_path = EVENTS / f"career_seasons_sum_{TAG}.csv"
    cols = ["player_id", "name", "pos", "team", "teams",
            "off_innings", "pit_innings", "fld_innings",
            "off_war", "pit_war", "fld_war", "total_war",
            "war_per_season", "peak_season_war",
            "seasons_played", "first_year", "last_year_played"]
    sums = sums[[c for c in cols if c in sums.columns]]
    sums.sort_values("total_war", ascending=False).to_csv(career_path, index=False)
    print(f"wrote {career_path} ({len(sums):,} players)")


if __name__ == "__main__":
    main()
