"""MLB Stats API loader: pulls play-by-play via statsapi.mlb.com.

Used for current-season data that hasn't shipped through Retrosheet yet
(and as a unified source if you want to skip cwevent entirely).

Per game we hit two endpoints:
  1. schedule (once per season) -> list of gamePks for regular-season games
  2. feed/live (per game)       -> play-by-play, venue, players

Each play (a complete plate appearance, plus a few special events) becomes
one row in the normalized event frame. Runs-on-play is derived from the
post-play awayScore/homeScore deltas, not by parsing runner-movement.

MLBAM numeric player ids are translated to Retrosheet ids via the Chadwick
register so retro + statsapi data can be unioned for combined fits. Players
without a retro entry get a synthetic id (`x{mlbam:06d}`).

Fielders: v1 leaves the fielder set empty. The fit script's per-role column
scaling already handles missing fielders; this just means current-season
defensive WAR is not computed. Upgrading requires walking each game's
substitution events (TODO documented in code).
"""
import gzip
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .crosswalk import load_mlbam_people, load_mlbam_to_retro, synth_retro_id

STATSAPI = "https://statsapi.mlb.com/api/v1"
STATSAPI_V11 = "https://statsapi.mlb.com/api/v1.1"
SCHEDULE_URL = STATSAPI + "/schedule?sportId=1&season={year}&gameType=R"
FEED_URL = STATSAPI_V11 + "/game/{game_pk}/feed/live"

REQ_DELAY_S = 0.05
RETRY = 3


def _cache_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "data" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _http_json(url: str) -> dict:
    for attempt in range(RETRY):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == RETRY - 1:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def _list_game_pks(year: int) -> list:
    sched = _http_json(SCHEDULE_URL.format(year=year))
    pks = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            # Skip postponed / cancelled / future games with no plays yet.
            state = g.get("status", {}).get("abstractGameState")
            if state != "Final":
                continue
            pks.append(g["gamePk"])
    return pks


def _load_feed(game_pk: int, year: int) -> dict:
    """Fetch a game's feed/live with on-disk caching, keyed by year so that
    when Retrosheet later publishes a season we can drop that year's whole
    cache in one rmtree."""
    cache = _cache_dir() / "feeds" / str(year) / f"{game_pk}.json.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return json.load(f)
    data = _http_json(FEED_URL.format(game_pk=game_pk))
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    time.sleep(REQ_DELAY_S)
    return data


def invalidate_year_cache(year: int) -> bool:
    """Remove all cached feed JSON for a season. Called by retro.ensure_year()
    after a season's Retrosheet zip lands locally. Returns True if anything
    was removed."""
    d = _cache_dir() / "feeds" / str(year)
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True


def _events_from_feed(feed: dict, mlbam_to_retro: dict, season: int):
    """Walk one game's plays and yield normalized event rows + (park, date)."""
    game = feed.get("gameData", {})
    venue = game.get("venue", {}) or {}
    park_code = str(venue.get("id", "UNK"))
    date_iso = (game.get("datetime", {}) or {}).get("officialDate", "")
    game_pk = str(game.get("game", {}).get("pk") or feed.get("gamePk", ""))

    all_plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    rows = []
    prev_away, prev_home = 0, 0
    for play in all_plays:
        about = play.get("about", {}) or {}
        result = play.get("result", {}) or {}
        matchup = play.get("matchup", {}) or {}
        inning = about.get("inning")
        half = about.get("halfInning")  # "top" or "bottom"
        if inning is None or half is None:
            continue
        bat_home = 1 if half == "bottom" else 0
        away_score = result.get("awayScore", prev_away)
        home_score = result.get("homeScore", prev_home)
        runs_on_play = (home_score - prev_home) if bat_home else (away_score - prev_away)
        prev_away, prev_home = away_score, home_score

        bat_mlbam = (matchup.get("batter") or {}).get("id")
        pit_mlbam = (matchup.get("pitcher") or {}).get("id")
        bat_id = _resolve_id(bat_mlbam, mlbam_to_retro)
        pit_id = _resolve_id(pit_mlbam, mlbam_to_retro)

        rows.append({
            "GAME_ID": game_pk,
            "INN_CT": int(inning),
            "BAT_HOME_ID": bat_home,
            "SEASON": int(season),
            "RESP_BAT_ID": bat_id,
            "RESP_PIT_ID": pit_id,
            "runs_on_play": int(max(0, runs_on_play)),
            # TODO(fielders): walk play["playEvents"] for "Defensive Substitution"
            # events combined with boxscore starters to track who was on the
            # field for each half-inning. Until then, leave all POS*_FLD_ID
            # empty -- aggregate_to_half_innings will emit fielders="".
        })
    return rows, park_code, game_pk, date_iso


def _resolve_id(mlbam_id, mlbam_to_retro: dict) -> str:
    if mlbam_id is None:
        return ""
    try:
        mlbam_id = int(mlbam_id)
    except (TypeError, ValueError):
        return ""
    return mlbam_to_retro.get(mlbam_id) or synth_retro_id(mlbam_id)


def load_events_and_parks(years: range):
    """Return (events_df, parks_df) populated from Stats API for the requested
    years. Caches per-game feed JSON locally so reruns are cheap."""
    mlbam_to_retro = load_mlbam_to_retro()
    all_events = []
    park_rows = []
    for y in years:
        pks = _list_game_pks(y)
        print(f"  {y}: {len(pks)} final games")
        for i, pk in enumerate(pks):
            feed = _load_feed(pk, y)
            rows, park_code, game_id, date_iso = _events_from_feed(feed, mlbam_to_retro, y)
            all_events.extend(rows)
            park_rows.append((game_id, park_code, date_iso))
            if i % 200 == 0 and i > 0:
                print(f"    {i:,}/{len(pks)} games")
    events_df = pd.DataFrame(all_events)
    parks_df = pd.DataFrame(park_rows, columns=["GAME_ID", "PARK", "DATE"]).drop_duplicates("GAME_ID")
    return events_df, parks_df


def load_rosters(years: range) -> pd.DataFrame:
    """Build a roster table by hitting the per-team roster endpoint for each
    year. Player id is the retro id (or synthetic) so it matches the events
    frame. We pull positions from each team's 40-man roster snapshot."""
    mlbam_to_retro = load_mlbam_to_retro()
    people = load_mlbam_people().set_index("key_mlbam")
    out = []
    for y in years:
        teams_url = f"{STATSAPI}/teams?sportId=1&season={y}"
        teams = _http_json(teams_url).get("teams", [])
        for t in teams:
            team_id = t.get("id")
            abbr = t.get("abbreviation", "")
            roster_url = f"{STATSAPI}/teams/{team_id}/roster?season={y}&rosterType=fullSeason"
            try:
                roster = _http_json(roster_url).get("roster", [])
            except urllib.error.HTTPError:
                continue
            for entry in roster:
                person = entry.get("person", {}) or {}
                mlbam = person.get("id")
                if mlbam is None:
                    continue
                pid = mlbam_to_retro.get(int(mlbam)) or synth_retro_id(int(mlbam))
                name = person.get("fullName") or (
                    people.loc[int(mlbam), "name"] if int(mlbam) in people.index else ""
                )
                pos = (entry.get("position") or {}).get("abbreviation", "")
                out.append({"player_id": pid, "name": name, "team": abbr,
                            "pos": pos, "year": y})
            time.sleep(REQ_DELAY_S)
    return pd.DataFrame(out, columns=["player_id", "name", "team", "pos", "year"])
