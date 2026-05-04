"""Run cwevent across all year directories and aggregate to half-inning rows.

Each year subdirectory data/raw/<YYYY>/ contains the event files, roster files,
and TEAM<YYYY> file required by cwevent. We run cwevent per year, concatenate,
then aggregate to half-innings tagged with season.
"""
import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
EVENTS = ROOT / "data" / "events"
EVENTS.mkdir(parents=True, exist_ok=True)
EVENTS_CSV = EVENTS / "events_all.csv"
HALF_OUT = EVENTS / "half_innings_all.parquet"

CWEVENT_FIELDS = "0,1,2,3,12,16,18,19,20,21,22,23,24,25,34,40,58,59,60,61"

def run_cwevent_for_year(year_dir: Path, year: int, write_header: bool, out_fh):
    files = sorted(f.name for f in year_dir.iterdir()
                   if f.suffix.upper() in (".EVA", ".EVN"))
    if not files:
        return 0
    cmd = ["cwevent", "-y", str(year), "-q",
           "-f", CWEVENT_FIELDS] + (["-n"] if write_header else []) + files
    proc = subprocess.run(cmd, cwd=year_dir, capture_output=True, text=True)
    out = proc.stdout
    if not write_header:
        # Drop header line
        nl = out.find("\n")
        if nl >= 0:
            out = out[nl + 1:]
    # Append a season column. Easiest way: post-process.
    lines = out.splitlines()
    if not lines:
        return 0
    if write_header:
        lines[0] = lines[0] + ',"SEASON"'
        body_start = 1
    else:
        body_start = 0
    for i in range(body_start, len(lines)):
        lines[i] = lines[i] + f",{year}"
    out_fh.write("\n".join(lines) + "\n")
    return len(lines) - body_start

# Concatenate all years' events
if not EVENTS_CSV.exists():
    year_dirs = sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.isdigit())
    total = 0
    with open(EVENTS_CSV, "w") as f:
        for i, yd in enumerate(year_dirs):
            year = int(yd.name)
            n = run_cwevent_for_year(yd, year, write_header=(i == 0), out_fh=f)
            total += n
            print(f"{year}: {n:,} events (cumulative {total:,})")
    print(f"wrote {EVENTS_CSV}")
else:
    print(f"using existing {EVENTS_CSV}")

# Aggregate to half-innings using direct numpy iteration over sorted events --
# pandas groupby.apply with python-level string aggregation is too slow on 15M
# events (1.5M groups × per-group set construction = many minutes).
print("loading events...")
df = pd.read_csv(EVENTS_CSV, low_memory=False)
print(f"loaded {len(df):,} events across {df.SEASON.nunique()} seasons "
      f"({df.SEASON.min()}-{df.SEASON.max()})")

dest_cols = ["BAT_DEST_ID", "RUN1_DEST_ID", "RUN2_DEST_ID", "RUN3_DEST_ID"]
runs_on_play = np.zeros(len(df), dtype=np.int8)
for c in dest_cols:
    runs_on_play += (df[c].values >= 4).astype(np.int8)

# Stable-sort by group key so half-innings are contiguous blocks.
print("sorting events...")
df = df.sort_values(["SEASON", "GAME_ID", "INN_CT", "BAT_HOME_ID"], kind="stable")
runs_on_play = runs_on_play[df.index.values]
df = df.reset_index(drop=True)

# Identify group boundaries.
print("finding group boundaries...")
gkey = (df["GAME_ID"].astype(str) + "|" +
        df["INN_CT"].astype(str) + "|" +
        df["BAT_HOME_ID"].astype(str)).values
boundaries = np.flatnonzero(np.r_[True, gkey[1:] != gkey[:-1], True])
n_groups = len(boundaries) - 1
print(f"  {n_groups:,} half-innings")

fielder_cols = ["POS2_FLD_ID", "POS3_FLD_ID", "POS4_FLD_ID",
                "POS5_FLD_ID", "POS6_FLD_ID", "POS7_FLD_ID",
                "POS8_FLD_ID", "POS9_FLD_ID"]

# Pull arrays once -- accessing df.iloc[i, c] in a loop is slow.
arr_game = df["GAME_ID"].values
arr_inn = df["INN_CT"].values
arr_bat_home = df["BAT_HOME_ID"].values
arr_season = df["SEASON"].values
arr_bat = df["RESP_BAT_ID"].values
arr_pit = df["RESP_PIT_ID"].values
arr_fld = [df[c].values for c in fielder_cols]

print("aggregating half-innings...")
out_records = []
for gi in range(n_groups):
    start, end = boundaries[gi], boundaries[gi + 1]
    runs = int(runs_on_play[start:end].sum())
    bat_set = set()
    pit_set = set()
    fld_set = set()
    for j in range(start, end):
        b = arr_bat[j]
        if b and isinstance(b, str):
            bat_set.add(b)
        p = arr_pit[j]
        if p and isinstance(p, str):
            pit_set.add(p)
        for fa in arr_fld:
            v = fa[j]
            if v and isinstance(v, str):
                fld_set.add(v)
    out_records.append((
        arr_game[start], int(arr_inn[start]), int(arr_bat_home[start]),
        int(arr_season[start]), runs,
        "|".join(sorted(bat_set)),
        "|".join(sorted(pit_set)),
        "|".join(sorted(fld_set)),
    ))
    if gi % 100_000 == 0 and gi > 0:
        print(f"  {gi:,}/{n_groups:,}")

half = pd.DataFrame(out_records, columns=[
    "GAME_ID", "INN_CT", "BAT_HOME_ID", "SEASON",
    "runs_scored", "batters", "pitchers", "fielders"])
print(f"{len(half):,} half-innings, runs/HI mean={half.runs_scored.mean():.3f}")
print("runs/HI by season summary:")
print(half.groupby("SEASON")["runs_scored"].mean().describe().to_string())

half.to_parquet(HALF_OUT, index=False)
print(f"wrote {HALF_OUT}")
