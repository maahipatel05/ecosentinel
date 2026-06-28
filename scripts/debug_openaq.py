"""
Diagnostic: find the nearest ACTIVE PM2.5 sensor for several cities,
and show how much data it has. Helps pick working eval cities.
"""
import asyncio, os, sys, json
from datetime import datetime, timezone, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
import httpx
from anomaly import geocode

OPENAQ_BASE = "https://api.openaq.org/v3"
CUTOFF_DAYS = 30   # sensor must have reported within this many days to be "active"

TEST_CITIES = [
    # Potentially polluted cities across different OpenAQ data providers
    "Delhi", "Mumbai", "Bangkok", "Hanoi",          # Asia
    "Los Angeles", "Houston", "Chicago", "Phoenix",  # USA (EPA AirNow)
    "Krakow", "Warsaw", "London", "Madrid",          # Europe
    "Mexico City", "Santiago", "Bogota",             # Latin America
    "Seoul", "Taipei",                               # East Asia
]


async def find_active_sensor(client, lat, lon, headers):
    """Try expanding radius until an active PM2.5 sensor is found."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=CUTOFF_DAYS)
    for radius in [25_000, 50_000, 100_000]:
        r = await client.get(
            f"{OPENAQ_BASE}/locations",
            params={"coordinates": f"{lat},{lon}", "radius": radius, "limit": 50},
            headers=headers,
        )
        if r.status_code != 200:
            continue
        for loc in r.json().get("results", []):
            last_str = (loc.get("datetimeLast") or {}).get("utc")
            if not last_str:
                continue
            try:
                last_dt = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
                if last_dt < cutoff:
                    continue
            except ValueError:
                continue
            for sensor in loc.get("sensors") or []:
                if (sensor.get("parameter") or {}).get("name", "").lower() == "pm25":
                    return sensor["id"], radius // 1000, last_str[:10]
    return None, None, None


async def check_data_availability(client, sensor_id, headers):
    """Return how many distinct days of data exist in the last 60 days."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    date_from = now - timedelta(days=60)
    r = await client.get(
        f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements",
        params={
            "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to":   now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 1000, "order_by": "datetime", "sort_order": "desc",
        },
        headers=headers,
    )
    if r.status_code != 200:
        return 0, r.status_code
    results = r.json().get("results", [])
    dates = set()
    for entry in results:
        ts = (entry.get("period") or {}).get("datetimeFrom") or {}
        ts = ts.get("utc") if isinstance(ts, dict) else None
        if ts:
            dates.add(ts[:10])
    return len(dates), 200


async def main():
    api_key = os.getenv("OPENAQ_API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}
    print(f"API key present: {bool(api_key)}\n")
    print(f"{'City':<18} {'Sensor':>8} {'Radius':>8} {'Last seen':>12} {'Days (60d)':>12} {'Status'}")
    print("-" * 75)

    async with httpx.AsyncClient(timeout=30) as client:
        for city in TEST_CITIES:
            try:
                lat, lon, _ = await geocode(city)
                sensor_id, radius_km, last_seen = await find_active_sensor(client, lat, lon, headers)
                if sensor_id is None:
                    print(f"{city:<18} {'—':>8} {'—':>8} {'—':>12} {'—':>12}  NO ACTIVE SENSOR")
                    continue
                days, status = await check_data_availability(client, sensor_id, headers)
                flag = "✅ GOOD" if days >= 30 else ("⚠️ SPARSE" if days >= 10 else "❌ TOO FEW")
                print(f"{city:<18} {sensor_id:>8} {radius_km:>6}km {last_seen:>12} {days:>12}  {flag}")
            except Exception as e:
                print(f"{city:<18} ERROR: {e}")
            await asyncio.sleep(0.5)  # be polite to the API

asyncio.run(main())
