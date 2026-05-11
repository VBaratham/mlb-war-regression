"""Build a half-innings dataset for a range of years.

Auto-routes per year: if data/raw/<YYYY>/ has Retrosheet event files we use
those; otherwise we fall back to MLB Stats API (for current-season data that
hasn't shipped through Retrosheet yet).

Examples:
  # Reproduce the existing 1910-2025 retro dataset under the default tag "all":
  python3 build_dataset.py --years 1910-2025 --tag all

  # Pull the current 2026 season (no retro data exists -> statsapi):
  python3 build_dataset.py --years 2026 --tag 2026

  # Combined: retro for past years, statsapi for current:
  python3 build_dataset.py --years 1910-2026 --tag combined

Outputs (under data/events/):
  half_innings_<tag>.parquet
  game_park_<tag>.csv
  rosters_<tag>.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from loaders.common import aggregate_to_half_innings

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
EVENTS_DIR = ROOT / "data" / "events"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_year_spec(spec: str) -> range:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return range(int(a), int(b) + 1)
    y = int(spec)
    return range(y, y + 1)


def _route_years(years):
    """Split a year range into (retro_years, statsapi_years).

    Retro is preferred when available. For years without local event files we
    ask retro.ensure_year() to HEAD retrosheet.org and pull the zip if it's
    been published since the last build -- so once Retrosheet ships a season,
    the next run silently switches it from statsapi to retro (and discards
    that year's cached feed JSON)."""
    from loaders import retro
    retro_years, statsapi_years = [], []
    for y in years:
        if retro.ensure_year(RAW, y):
            retro_years.append(y)
        else:
            statsapi_years.append(y)
    return retro_years, statsapi_years


def _retro_part(years):
    from loaders import retro
    print(f"[retro] {len(years)} year(s): {years[0]}..{years[-1]}")
    events = retro.load_events(RAW, years)
    print(f"[retro] {len(events):,} events")
    parks = retro.load_parks(RAW, years)
    print(f"[retro] {len(parks):,} game-park rows")
    rosters = retro.load_rosters(RAW, years)
    print(f"[retro] {len(rosters):,} roster rows")
    return events, parks, rosters


def _statsapi_part(years):
    from loaders import statsapi
    print(f"[statsapi] {len(years)} year(s): {years[0]}..{years[-1]}")
    events, parks = statsapi.load_events_and_parks(years)
    print(f"[statsapi] {len(events):,} events, {len(parks):,} games")
    rosters = statsapi.load_rosters(years)
    print(f"[statsapi] {len(rosters):,} roster rows")
    return events, parks, rosters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", required=True, help="YYYY or YYYY-YYYY")
    ap.add_argument("--tag", required=True, help="suffix for output files")
    args = ap.parse_args()

    years = list(_parse_year_spec(args.years))
    retro_years, statsapi_years = _route_years(years)
    print(f"requested {years[0]}..{years[-1]} -> "
          f"retro: {len(retro_years)} season(s), statsapi: {len(statsapi_years)} season(s)")
    if statsapi_years:
        print(f"  statsapi years: {statsapi_years}")

    events_parts, parks_parts, rosters_parts = [], [], []
    if retro_years:
        e, p, r = _retro_part(retro_years)
        events_parts.append(e); parks_parts.append(p); rosters_parts.append(r)
    if statsapi_years:
        e, p, r = _statsapi_part(statsapi_years)
        events_parts.append(e); parks_parts.append(p); rosters_parts.append(r)

    if not events_parts:
        raise SystemExit(f"no data available for years {years[0]}..{years[-1]}")

    events = pd.concat(events_parts, ignore_index=True)
    parks = pd.concat(parks_parts, ignore_index=True).drop_duplicates("GAME_ID")
    rosters = pd.concat(rosters_parts, ignore_index=True)

    print(f"\naggregating {len(events):,} events to half-innings...")
    half = aggregate_to_half_innings(events)
    print(f"  {len(half):,} half-innings, runs/HI mean={half.runs_scored.mean():.3f}")

    half_path = EVENTS_DIR / f"half_innings_{args.tag}.parquet"
    park_path = EVENTS_DIR / f"game_park_{args.tag}.csv"
    ros_path = EVENTS_DIR / f"rosters_{args.tag}.csv"
    half.to_parquet(half_path, index=False)
    parks.to_csv(park_path, index=False)
    rosters.to_csv(ros_path, index=False)
    print(f"\nwrote {half_path}")
    print(f"wrote {park_path}")
    print(f"wrote {ros_path}")


if __name__ == "__main__":
    main()
