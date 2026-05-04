"""Extract per-game park ID from Retrosheet event files.

Each event file has blocks like:
  id,ANA202404050
  version,2
  info,visteam,BOS
  info,hometeam,ANA
  info,site,ANA01      <-- this is the park code
  info,date,2024/04/05
  ...

We just grep these out into a (GAME_ID, SITE) lookup.
"""
from pathlib import Path
import csv

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "events" / "game_park.csv"

records = []
files = sorted(RAW.rglob("*.EV?"))
print(f"scanning {len(files):,} event files...")
for fp in files:
    current_id = None
    with open(fp, encoding="latin-1") as f:
        for line in f:
            if line.startswith("id,"):
                current_id = line[3:].strip()
            elif line.startswith("info,site,") and current_id:
                site = line[len("info,site,"):].strip()
                records.append((current_id, site))
                current_id = None  # only first site per game

print(f"extracted {len(records):,} game→park records")
unique_parks = sorted(set(s for _, s in records))
print(f"{len(unique_parks)} unique parks")
print("first 10:", unique_parks[:10])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["GAME_ID", "PARK"])
    w.writerows(records)
print(f"wrote {OUT}")
