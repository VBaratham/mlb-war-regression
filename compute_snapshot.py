"""Backfill per-day WAR snapshots for the current season.

The webapp's "WAR over time" chart on the current-season view plots each
player's cumulative WAR across snapshot dates. To populate it densely we
refit the ridge model on every date between season opening day and today,
filtering half-innings to games whose `DATE` field is on-or-before that day.
Each fit writes a slim CSV (player_id, name, pos, off/pit/fld/total WAR)
to data/events/snapshots/<tag>/<YYYY-MM-DD>.csv.

The cron entry point runs this with the current-year tag; it skips dates
that already have a CSV, so daily reruns only refit any missed days.

Usage:
    # Fill any missing days for the in-progress season:
    python3 compute_snapshot.py --tag 2026

    # Force a recompute for a single date (overrides the skip-existing check):
    python3 compute_snapshot.py --tag 2026 --date 2026-05-11 --force

    # Backfill an arbitrary range:
    python3 compute_snapshot.py --tag 2026 --start 2026-04-15 --end 2026-05-01
"""
import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fit_per_season import fit_one_season, MIN_HALF_INNINGS_PER_SEASON

ROOT = Path(__file__).parent
EVENTS = ROOT / "data" / "events"

# A snapshot fit needs much less data than a full season; relax the floor.
MIN_HI_FOR_SNAPSHOT = 200


def _load_tag_with_dates(tag: str) -> pd.DataFrame:
    """Return half_innings_<tag> merged with PARK + DATE from game_park_<tag>."""
    half = pd.read_parquet(EVENTS / f"half_innings_{tag}.parquet")
    half["GAME_ID"] = half["GAME_ID"].astype(str)
    parks = pd.read_csv(EVENTS / f"game_park_{tag}.csv", dtype={"GAME_ID": str})
    if "DATE" not in parks.columns:
        raise SystemExit(
            f"game_park_{tag}.csv has no DATE column. "
            f"Rebuild via `build_dataset.py --years <yr> --tag {tag}`."
        )
    half = half.merge(parks, on="GAME_ID", how="left")
    half["PARK"] = half["PARK"].fillna("UNK")
    half["DATE"] = half["DATE"].fillna("")
    return half


def _player_meta(tag: str) -> pd.DataFrame:
    """Roster metadata for the slim snapshot output. Union across every
    coefficients_*_enriched parquet so both retro and statsapi player IDs
    resolve to names."""
    metas = []
    for f in sorted(EVENTS.glob("coefficients_*_enriched.parquet")):
        try:
            metas.append(pd.read_parquet(
                f, columns=["player_id", "name", "pos", "team", "teams"]))
        except Exception:
            try:
                metas.append(pd.read_parquet(
                    f, columns=["player_id", "name", "pos", "team"]))
            except Exception:
                pass
    if not metas:
        return pd.DataFrame(columns=["player_id", "name", "pos"])
    return (pd.concat(metas, ignore_index=True)
              .drop_duplicates("player_id", keep="first"))


def _existing_snapshot_dates(tag: str) -> set:
    d = EVENTS / "snapshots" / tag
    if not d.is_dir():
        return set()
    return {f.stem for f in d.glob("*.csv")}


def _game_dates(half: pd.DataFrame) -> list:
    """All distinct game dates that appear in this tag's data, sorted."""
    return sorted(d for d in half["DATE"].dropna().unique() if d)


def _write_snapshot(rows: list, meta: pd.DataFrame, tag: str, date: str):
    df = pd.DataFrame(rows)
    df = df.merge(meta, on="player_id", how="left")
    keep = ["player_id", "name", "pos",
            "off_war", "pit_war", "fld_war", "total_war"]
    out_dir = EVENTS / "snapshots" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{date}.csv"
    df[keep].to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="dataset tag, e.g. 2026")
    ap.add_argument("--date", help="single ISO date to snapshot")
    ap.add_argument("--start", help="ISO date to start backfill (inclusive)")
    ap.add_argument("--end", help="ISO date to end backfill (inclusive, default = today UTC)")
    ap.add_argument("--force", action="store_true",
                    help="re-fit even if a snapshot file already exists")
    args = ap.parse_args()

    half = _load_tag_with_dates(args.tag)
    meta = _player_meta(args.tag)

    all_game_dates = _game_dates(half)
    if not all_game_dates:
        raise SystemExit(f"no games with DATE found in tag {args.tag}")

    today = dt.date.today().isoformat()
    season_start = all_game_dates[0]

    if args.date:
        targets = [args.date]
    else:
        start = args.start or season_start
        end = args.end or today
        # Snapshot at every date with at least one game played, between start and end.
        targets = [d for d in all_game_dates if start <= d <= end]

    if not targets:
        print("no candidate dates")
        return

    existing = _existing_snapshot_dates(args.tag) if not args.force else set()
    to_fit = [d for d in targets if d not in existing]
    if not to_fit:
        print(f"all {len(targets)} dates already snapshotted (use --force to redo)")
        return

    print(f"{len(to_fit)} date(s) to fit for tag={args.tag} "
          f"({to_fit[0]} → {to_fit[-1]})")

    t_start = time.time()
    for i, date in enumerate(to_fit):
        sub = half[half["DATE"].astype(str) <= date]
        if len(sub) < MIN_HI_FOR_SNAPSHOT:
            print(f"  {date}: only {len(sub)} HI; skipping (need {MIN_HI_FOR_SNAPSHOT})")
            continue
        t0 = time.time()
        # fit_one_season uses the SEASON column for label only; values are
        # written into output rows but we discard them when writing the
        # date-stamped snapshot. fit_one_season expects PARK + BAT_HOME_ID
        # + runs_scored + batters/pitchers/fielders — all already merged.
        rows, off_sd = fit_one_season(sub, season=int(args.tag) if args.tag.isdigit() else 0)
        path = _write_snapshot(rows, meta, args.tag, date)
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(to_fit) - i - 1)
        print(f"  {date}: {len(sub):>6,} HI, {len(rows):>4} players, "
              f"off_sd={off_sd:.4f}  ({time.time() - t0:.1f}s, elapsed {elapsed:.0f}s, eta {eta:.0f}s) "
              f"-> {path.name}")


if __name__ == "__main__":
    main()
