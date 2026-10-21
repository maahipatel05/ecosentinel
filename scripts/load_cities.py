"""
Load recent PM2.5 history for a list of cities into the BigQuery warehouse.

Safe to re-run: load_city_history only appends readings newer than what
BigQuery already holds for each sensor, so a second run adds ~0 rows.

Usage (from the project root, venv active):
    python scripts/load_cities.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from warehouse import load_city_history

CITIES = [
    "Ahmedabad", "Delhi", "Mumbai", "Lahore", "Bangkok",
    "Jakarta", "Seoul", "Beijing", "Sydney", "London",
    "Paris", "Krakow", "Los Angeles", "Chicago", "Mexico City",
]


async def main() -> None:
    total = 0
    loaded_cities = 0
    # One city at a time keeps us well under OpenAQ's rate limit
    for city in CITIES:
        try:
            result = await load_city_history(city)
        except Exception as e:
            print(f"  {city:<12} ERROR {type(e).__name__}: {str(e)[:80]}")
            continue
        loaded = result.get("loaded", 0)
        total += loaded
        if result.get("fetched", 0):
            loaded_cities += 1
        print(
            f"  {result['city']:<28} sensor {result.get('sensor_id')}  "
            f"fetched {result.get('fetched', 0):>5}  loaded {loaded:>5}"
        )
    print(f"\nNew rows loaded: {total:,} | cities with data: {loaded_cities}/{len(CITIES)}")


if __name__ == "__main__":
    asyncio.run(main())
