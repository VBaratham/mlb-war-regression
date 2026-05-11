"""Retrosheet loader: cwevent over data/raw/<YYYY>/, plus park + roster pulls.

Produces three frames in normalized form:
  events_df    -- per-event rows ready for aggregate_to_half_innings()
  parks_df     -- GAME_ID, PARK
  rosters_df   -- player_id, name, team, pos, year

Also exposes ensure_year(raw_dir, year), which build_dataset.py calls before
falling back to the Stats API. It HEADs retrosheet.org for that season's zip
and fetches it on the fly when available -- so once Retrosheet publishes a
season, the next build run silently switches that year from statsapi to retro.
"""
import io
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CWEVENT_FIELDS = "0,1,2,3,12,16,18,19,20,21,22,23,24,25,34,40,58,59,60,61"
_DEST_COLS = ["BAT_DEST_ID", "RUN1_DEST_ID", "RUN2_DEST_ID", "RUN3_DEST_ID"]

RETROSHEET_BASE = "https://www.retrosheet.org/events"
_HEAD_TIMEOUT_S = 10
_FETCH_TIMEOUT_S = 60


def _has_event_files(ydir: Path) -> bool:
    if not ydir.is_dir():
        return False
    for f in ydir.iterdir():
        if f.suffix.upper() in (".EVA", ".EVN"):
            return True
    return False


def ensure_year(raw_dir: Path, year: int) -> bool:
    """Make Retrosheet event files for `year` available locally if possible.

    If files already exist under data/raw/<year>/, no-op and return True.
    Otherwise issue a HEAD against retrosheet.org/events/<year>eve.zip; if it
    returns 200 we GET the zip, extract it, and invalidate any cached
    statsapi feeds for that season. Returns True iff retro data is available
    locally after the call.
    """
    ydir = raw_dir / str(year)
    if _has_event_files(ydir):
        return True
    url = f"{RETROSHEET_BASE}/{year}eve.zip"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=_HEAD_TIMEOUT_S) as r:
            if getattr(r, "status", 200) != 200:
                return False
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False
    print(f"  [retro] {year} now on retrosheet.org -> downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as r:
            payload = r.read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  [retro] {year} fetch failed: {e}")
        return False
    ydir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(ydir)
    if not _has_event_files(ydir):
        print(f"  [retro] {year} zip extracted but no .EV? files found")
        return False
    # Drop any statsapi feeds we previously cached for this season.
    try:
        from . import statsapi as _statsapi
        if _statsapi.invalidate_year_cache(year):
            print(f"  [retro] {year} cleared statsapi feed cache")
    except Exception as e:
        print(f"  [retro] {year} cache invalidate skipped: {e}")
    return True


def _run_cwevent_year(year_dir: Path, year: int) -> str:
    """Run cwevent in a year's data directory and return its CSV text (with header)."""
    files = sorted(f.name for f in year_dir.iterdir()
                   if f.suffix.upper() in (".EVA", ".EVN"))
    if not files:
        return ""
    cmd = ["cwevent", "-y", str(year), "-q", "-f", CWEVENT_FIELDS, "-n"] + files
    proc = subprocess.run(cmd, cwd=year_dir, capture_output=True, text=True, check=True)
    return proc.stdout


def load_events(raw_dir: Path, years: range) -> pd.DataFrame:
    """Run cwevent across each requested year and assemble a per-event frame
    in the normalized schema. Computes runs_on_play from *_DEST_ID fields."""
    frames = []
    for y in years:
        ydir = raw_dir / str(y)
        if not ydir.is_dir():
            print(f"  {y}: no data, skipping")
            continue
        text = _run_cwevent_year(ydir, y)
        if not text:
            continue
        sub = pd.read_csv(io.StringIO(text), low_memory=False)
        sub["SEASON"] = y
        frames.append(sub)
        print(f"  {y}: {len(sub):,} events")
    if not frames:
        raise RuntimeError(f"no Retrosheet events found under {raw_dir} for {years}")
    df = pd.concat(frames, ignore_index=True)

    runs = np.zeros(len(df), dtype=np.int8)
    for c in _DEST_COLS:
        runs += (df[c].values >= 4).astype(np.int8)
    df["runs_on_play"] = runs
    return df


def load_parks(raw_dir: Path, years: range) -> pd.DataFrame:
    """Scan event files for `info,site,XXX` lines → GAME_ID,PARK rows."""
    records = []
    for y in years:
        ydir = raw_dir / str(y)
        if not ydir.is_dir():
            continue
        for fp in sorted(ydir.glob("*.EV?")):
            current_id = None
            with open(fp, encoding="latin-1") as f:
                for line in f:
                    if line.startswith("id,"):
                        current_id = line[3:].strip()
                    elif line.startswith("info,site,") and current_id:
                        records.append((current_id, line[len("info,site,"):].strip()))
                        current_id = None
    return pd.DataFrame(records, columns=["GAME_ID", "PARK"])


def load_rosters(raw_dir: Path, years: range) -> pd.DataFrame:
    """Read all *.ROS files in the requested years."""
    frames = []
    for y in years:
        ydir = raw_dir / str(y)
        if not ydir.is_dir():
            continue
        for f in ydir.glob("*.ROS"):
            try:
                r = pd.read_csv(f, header=None, usecols=[0, 1, 2, 5, 6],
                                names=["player_id", "last", "first", "team", "pos"])
                r["year"] = y
                frames.append(r)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=["player_id", "name", "team", "pos", "year"])
    ros = pd.concat(frames, ignore_index=True)
    ros["name"] = ros["first"].fillna("") + " " + ros["last"].fillna("")
    return ros[["player_id", "name", "team", "pos", "year"]]
