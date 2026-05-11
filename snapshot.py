"""Write a dated snapshot of current-season coefficients + regenerate manifest.

The webapp's "WAR over time" chart needs per-day cumulative WAR for each
player. We get that by dropping a slim CSV under
    data/events/snapshots/<tag>/<YYYY-MM-DD>.csv
after each cron-driven refit. The manifest.json at the events root indexes
the available all-time leaderboard, the current-season leaderboard, and the
list of snapshot dates so the static frontend can discover everything
without parsing the directory tree.

Usage:
    python3 snapshot.py --tag 2026 --season 2026
    python3 snapshot.py --update-manifest-only
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
EVENTS = ROOT / "data" / "events"

# Columns we expose to the webapp. Drop the per-season / per-position-z fields
# from the leaderboard CSVs to keep the network payload small. Snapshot CSVs
# get an even slimmer set since they're written daily.
LEADERBOARD_COLS = [
    "player_id", "name", "last_year", "team", "pos",
    "off_innings", "pit_innings", "fld_innings",
    "off_war", "pit_war", "fld_war", "total_war",
    "off_war_per_season", "pit_war_per_season", "fld_war_per_season",
    "total_war_per_season",
    "off_war_full_season_rate", "pit_war_full_season_rate",
    "fld_war_full_season_rate",
]
SNAPSHOT_COLS = [
    "player_id", "name", "pos",
    "off_war", "pit_war", "fld_war", "total_war",
]


def write_snapshot(tag: str, date: dt.date) -> Path:
    src = EVENTS / f"coefficients_{tag}_enriched.parquet"
    if not src.exists():
        src = EVENTS / f"coefficients_{tag}.parquet"
    df = pd.read_parquet(src)
    snap_dir = EVENTS / "snapshots" / tag
    snap_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in SNAPSHOT_COLS if c in df.columns]
    out = snap_dir / f"{date.isoformat()}.csv"
    df[cols].to_csv(out, index=False)
    print(f"wrote snapshot {out} ({len(df):,} rows)")
    return out


def list_snapshot_dates(tag: str) -> list:
    snap_dir = EVENTS / "snapshots" / tag
    if not snap_dir.is_dir():
        return []
    dates = []
    for f in snap_dir.glob("*.csv"):
        try:
            dt.date.fromisoformat(f.stem)
            dates.append(f.stem)
        except ValueError:
            continue
    return sorted(dates)


def _detect_tags() -> dict:
    """Find which leaderboards exist on disk. Returns
        {"all_time": tag_or_None, "current": (tag, season) or None}
    Any tag named purely digits is treated as a single-season tag; "all" is
    the all-time tag; anything else (e.g. "combined") falls into all_time as
    a fallback."""
    tags = set()
    for f in EVENTS.glob("coefficients_*_enriched.csv"):
        stem = f.stem[len("coefficients_"):-len("_enriched")]
        tags.add(stem)
    all_time = None
    current = None
    for t in tags:
        if t.isdigit() and len(t) == 4:
            year = int(t)
            if current is None or year > current[1]:
                current = (t, year)
        else:
            # prefer "all" but accept any non-numeric tag as all-time
            if all_time is None or t == "all":
                all_time = t
    return {"all_time": all_time, "current": current}


def _list_seasons(tag: str) -> list:
    """Return the sorted unique seasons present in season_war_<tag>.parquet,
    or [] if it doesn't exist."""
    p = EVENTS / f"season_war_{tag}.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p, columns=["season"])
    return sorted(int(s) for s in df["season"].unique())


def write_manifest():
    found = _detect_tags()
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z",
        "all_time": None,
        "all_time_single_fit": None,
        "current_season": None,
        "season_index": None,
    }
    if found["all_time"]:
        t = found["all_time"]
        # Default career view: per-season-summed WAR. Robust to the cross-era
        # identifiability issues of the single all-time fit.
        career_sum_path = EVENTS / f"career_seasons_sum_{t}.csv"
        if career_sum_path.exists():
            manifest["all_time"] = {
                "tag": t,
                "label": _label_for(t),
                "leaderboard": f"career_seasons_sum_{t}.csv",
                "kind": "seasons_sum",
            }
            manifest["all_time_single_fit"] = {
                "tag": t,
                "label": f"{_label_for(t)} (single-fit)",
                "leaderboard": f"coefficients_{t}_enriched.csv",
                "kind": "single_fit",
            }
        else:
            manifest["all_time"] = {
                "tag": t,
                "label": _label_for(t),
                "leaderboard": f"coefficients_{t}_enriched.csv",
                "kind": "single_fit",
            }
        seasons = _list_seasons(t)
        if seasons:
            manifest["season_index"] = {
                "tag": t,
                "file": f"season_war_{t}.csv",
                "seasons": seasons,
            }
    if found["current"]:
        t, season = found["current"]
        snapshots = list_snapshot_dates(t)
        manifest["current_season"] = {
            "tag": t,
            "season": season,
            "label": f"{season} season",
            "leaderboard": f"coefficients_{t}_enriched.csv",
            "snapshots": [
                {"date": d, "file": f"snapshots/{t}/{d}.csv"}
                for d in snapshots
            ],
        }
    out = EVENTS / "manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}")
    print(f"  all_time: {manifest['all_time']['tag'] if manifest['all_time'] else None}")
    if manifest["current_season"]:
        cs = manifest["current_season"]
        print(f"  current: {cs['tag']} ({cs['season']}), {len(cs['snapshots'])} snapshots")
    return out


def _label_for(tag: str) -> str:
    if tag == "all":
        return "All-time"
    if tag == "combined":
        return "All-time (combined)"
    if tag.isdigit():
        return f"{tag} season"
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="dataset tag whose coefficients to snapshot")
    ap.add_argument("--date", help="ISO date for the snapshot (default: today UTC)")
    ap.add_argument("--update-manifest-only", action="store_true")
    args = ap.parse_args()

    if not args.update_manifest_only:
        if not args.tag:
            ap.error("--tag required unless --update-manifest-only")
        date = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(dt.timezone.utc).date()
        write_snapshot(args.tag, date)

    write_manifest()


if __name__ == "__main__":
    main()
