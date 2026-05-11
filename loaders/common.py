"""Source-agnostic event-to-half-inning aggregator.

Each loader (Retrosheet, MLB Stats API) produces a per-event DataFrame in a
normalized schema; this module rolls it up to the half-inning rows that the
regression consumes.

Normalized per-event schema:
    GAME_ID         str
    INN_CT          int        inning number, 1..N
    BAT_HOME_ID     int        0 = visiting team batting (top), 1 = home (bot)
    SEASON          int        4-digit year
    RESP_BAT_ID     str        player id of the batter (may be empty)
    RESP_PIT_ID     str        player id of the pitcher (may be empty)
    runs_on_play    int        runs scored on this event
    POS2_FLD_ID..POS9_FLD_ID   optional; missing columns treated as empty

Output half-innings schema (matches the existing parquet contract):
    GAME_ID, INN_CT, BAT_HOME_ID, SEASON, runs_scored,
    batters, pitchers, fielders   (last three are '|'-joined sorted id sets)
"""
import numpy as np
import pandas as pd

REQUIRED_EVENT_COLS = [
    "GAME_ID", "INN_CT", "BAT_HOME_ID", "SEASON",
    "RESP_BAT_ID", "RESP_PIT_ID", "runs_on_play",
]
FIELDER_COLS = [f"POS{i}_FLD_ID" for i in range(2, 10)]

HALF_INNINGS_COLS = [
    "GAME_ID", "INN_CT", "BAT_HOME_ID", "SEASON",
    "runs_scored", "batters", "pitchers", "fielders",
]


def aggregate_to_half_innings(events: pd.DataFrame) -> pd.DataFrame:
    """Roll per-event rows up to one row per (GAME_ID, INN_CT, BAT_HOME_ID).

    Sorts events into contiguous half-inning blocks then walks the block
    boundaries -- pandas groupby.apply on 10M+ events is too slow because of
    per-group set construction.
    """
    missing = [c for c in REQUIRED_EVENT_COLS if c not in events.columns]
    if missing:
        raise ValueError(f"events frame missing required columns: {missing}")

    df = events.sort_values(
        ["SEASON", "GAME_ID", "INN_CT", "BAT_HOME_ID"], kind="stable"
    ).reset_index(drop=True)

    gkey = (df["GAME_ID"].astype(str) + "|" +
            df["INN_CT"].astype(str) + "|" +
            df["BAT_HOME_ID"].astype(str)).values
    boundaries = np.flatnonzero(np.r_[True, gkey[1:] != gkey[:-1], True])
    n_groups = len(boundaries) - 1

    arr_game = df["GAME_ID"].values
    arr_inn = df["INN_CT"].values
    arr_bat_home = df["BAT_HOME_ID"].values
    arr_season = df["SEASON"].values
    arr_bat = df["RESP_BAT_ID"].values
    arr_pit = df["RESP_PIT_ID"].values
    arr_runs = df["runs_on_play"].values.astype(np.int8, copy=False)
    arr_fld = [df[c].values for c in FIELDER_COLS if c in df.columns]

    out = []
    for gi in range(n_groups):
        start, end = boundaries[gi], boundaries[gi + 1]
        runs = int(arr_runs[start:end].sum())
        bat_set, pit_set, fld_set = set(), set(), set()
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
        out.append((
            arr_game[start], int(arr_inn[start]), int(arr_bat_home[start]),
            int(arr_season[start]), runs,
            "|".join(sorted(bat_set)),
            "|".join(sorted(pit_set)),
            "|".join(sorted(fld_set)),
        ))
        if gi % 100_000 == 0 and gi > 0:
            print(f"  {gi:,}/{n_groups:,}")

    return pd.DataFrame(out, columns=HALF_INNINGS_COLS)
