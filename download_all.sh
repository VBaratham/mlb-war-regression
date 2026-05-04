#!/usr/bin/env bash
# Download all Retrosheet event zips (1910-2025) into per-year directories.
set -euo pipefail
RAW=/Users/vbaratham/claude/mlb_war_regression/data/raw
mkdir -p "$RAW"
cd "$RAW"

for y in $(seq 1910 2025); do
    if [ -d "$y" ] && ls "$y" | grep -q '\.EV[AN]$'; then
        continue  # already extracted
    fi
    url="https://www.retrosheet.org/events/${y}eve.zip"
    if curl -sf -o "${y}eve.zip" "$url"; then
        mkdir -p "$y"
        unzip -q -o "${y}eve.zip" -d "$y"
        rm "${y}eve.zip"
        echo "$y: $(ls $y | grep -c '\.EV[AN]$') event files"
    else
        echo "$y: missing"
    fi
done
