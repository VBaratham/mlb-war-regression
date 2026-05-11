#!/usr/bin/env bash
# Cronjob entry point: refit the current season, write a daily snapshot,
# regenerate the manifest, and push the changed CSVs to GitHub so the static
# webapp picks them up. Idempotent if run multiple times per day.
#
# Designed to be cronned daily, e.g.:
#   30 9 * * * cd /path/to/mlb_war_regression && ./refresh.sh >> refresh.log 2>&1
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

SEASON="$(date -u +%Y)"
TAG="$SEASON"

echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) refresh.sh tag=$TAG ===="

git pull --rebase --autostash

# Rebuild current season half-innings (statsapi for an in-progress year, or
# retro once it ships). Skipped if nothing new -- build_dataset.py reuses the
# per-game feed cache so this is cheap on incremental days.
python3 build_dataset.py --years "$SEASON" --tag "$TAG"

# Refit and regenerate views.
python3 fit_ridge_all.py --tag "$TAG"
python3 make_views.py --tag "$TAG"

# Per-season fit. We update the current season's rows inside the all-time
# season_war table (incremental: replaces just that year's entries) so the
# webapp's season-view + player-detail panel see fresh data each day.
python3 fit_per_season.py --tag all --seasons "$SEASON"

# Drop a dated snapshot and update the manifest the webapp reads.
python3 snapshot.py --tag "$TAG"

# Stage only the lightweight CSVs / manifest. Half-inning parquets and feed
# caches are .gitignore'd. Force-add coefficients_*.csv since the
# computed-output gitignore patterns may exclude them in some configurations.
git add -f \
    data/events/coefficients_"$TAG".csv \
    data/events/coefficients_"$TAG"_enriched.csv \
    data/events/coefficients_all.csv \
    data/events/coefficients_all_enriched.csv 2>/dev/null || true
git add -f data/events/snapshots/"$TAG"/*.csv 2>/dev/null || true
git add -f data/events/season_war_all.csv 2>/dev/null || true
git add -f data/events/manifest.json

if git diff --cached --quiet; then
    echo "no changes to commit"
    exit 0
fi

git commit -m "refresh: $TAG snapshot $(date -u +%Y-%m-%d)"
git push
