"""MLBAM ↔ Retrosheet player-id crosswalk via Chadwick Bureau register.

The Chadwick register is the canonical mapping between every baseball ID
system. We download the people-*.csv files once, cache locally, and expose a
single dict {mlbam_id (int) -> retro_id (str)}.

For MLBAM players with no retro_id (e.g. recent rookies whose retro entry
hasn't been issued), we synthesize one as f"x{mlbam_id:06d}" so the
downstream sparse-matrix pipeline still sees a stable string identifier.
"""
import io
from pathlib import Path

import pandas as pd

CHADWICK_BASE = "https://raw.githubusercontent.com/chadwickbureau/register/master/data"
PEOPLE_SHARDS = "0123456789abcdef"  # people-0.csv ... people-f.csv


def _cache_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "data" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_register(force: bool = False) -> pd.DataFrame:
    """Fetch (and locally cache) the full Chadwick people register."""
    cache = _cache_dir() / "chadwick_people.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    import urllib.request
    frames = []
    for shard in PEOPLE_SHARDS:
        url = f"{CHADWICK_BASE}/people-{shard}.csv"
        print(f"  fetching {url}")
        with urllib.request.urlopen(url) as r:
            data = r.read().decode("utf-8")
        sub = pd.read_csv(io.StringIO(data), low_memory=False,
                          usecols=["key_mlbam", "key_retro",
                                   "name_first", "name_last"])
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache, index=False)
    print(f"  cached {len(df):,} player records to {cache}")
    return df


def load_mlbam_to_retro(force: bool = False) -> dict:
    """Return {mlbam_id (int) -> retro_id (str)}. Drops rows lacking either."""
    df = _fetch_register(force=force)
    df = df.dropna(subset=["key_mlbam", "key_retro"])
    df = df[df["key_retro"].astype(str).str.len() > 0]
    df["key_mlbam"] = df["key_mlbam"].astype(int)
    return dict(zip(df["key_mlbam"].values, df["key_retro"].astype(str).values))


def load_mlbam_people(force: bool = False) -> pd.DataFrame:
    """Full register frame with mlbam, retro, and name -- used to build rosters
    for sources that emit MLBAM ids (Stats API)."""
    df = _fetch_register(force=force)
    df = df.dropna(subset=["key_mlbam"]).copy()
    df["key_mlbam"] = df["key_mlbam"].astype(int)
    df["name"] = (df["name_first"].fillna("") + " " + df["name_last"].fillna("")).str.strip()
    return df[["key_mlbam", "key_retro", "name"]]


def synth_retro_id(mlbam_id: int) -> str:
    """Fallback id for MLBAM players with no Chadwick retro entry.
    Prefix `x` distinguishes from real retro ids (which start with a letter
    derived from the surname)."""
    return f"x{int(mlbam_id):06d}"
