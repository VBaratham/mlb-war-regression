#!/usr/bin/env bash
# Download every available Retrosheet event zip from 1910 through the current
# year into per-year directories. After successfully extracting a year, wipe
# any cached MLB Stats API feeds for that season -- once Retrosheet is the
# source of truth for the year, the feed JSON is dead weight.
set -euo pipefail
RAW=/Users/vbaratham/claude/mlb_war_regression/data/raw
FEED_CACHE=/Users/vbaratham/claude/mlb_war_regression/data/cache/feeds
mkdir -p "$RAW"
cd "$RAW"

end_year=$(date +%Y)
for y in $(seq 1910 "$end_year"); do
    if [ -d "$y" ] && ls "$y" | grep -q '\.EV[AN]$'; then
        continue  # already extracted
    fi
    url="https://www.retrosheet.org/events/${y}eve.zip"
    if curl -sf -o "${y}eve.zip" "$url"; then
        mkdir -p "$y"
        unzip -q -o "${y}eve.zip" -d "$y"
        rm "${y}eve.zip"
        echo "$y: $(ls $y | grep -c '\.EV[AN]$') event files"
        if [ -d "$FEED_CACHE/$y" ]; then
            rm -rf "$FEED_CACHE/$y"
            echo "$y: cleared statsapi feed cache"
        fi
    else
        echo "$y: not yet published"
    fi
done
