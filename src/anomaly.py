"""
EcoSentinel Anomaly Detection Engine
======================================
Detects statistically significant spikes in PM2.5 air quality
using z-score analysis over a 30-day historical baseline.

Called by server.py when the agentic router needs to determine
whether current air quality is statistically unusual for a city.

Mathematical method: Z-score
  z = (current_value - mean) / standard_deviation
  z > 2.0 = anomaly confirmed
"""

import os
import httpx
import numpy as np
from scipy import stats
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
import cache as _cache

# Load API keys using absolute path so this works
# regardless of where it is launched from
load_dotenv(Path(__file__).parent.parent / ".env")

OPENAQ_BASE      = "https://api.openaq.org/v3"
GEOCODE_BASE     = "https://geocoding-api.open-meteo.com/v1"
MIN_DATA_POINTS  = 10   # minimum readings needed for a valid z-score
Z_THRESHOLD_WARN = 1.5  # above this: elevated alert
Z_THRESHOLD_ANOM = 2.0  # above this: confirmed anomaly


# ── Helper: geocode ────────────────────────────────────────────────────────────

async def geocode(city: str) -> tuple[float, float, str]:
    """Convert city name to (lat, lon, display_name)."""
    key = f"geocode:{city.lower().strip()}"
    cached = _cache.get(key)
    if cached is not None:
        return cached
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{GEOCODE_BASE}/search",
            params={"name": city, "count": 1, "format": "json"},
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            raise ValueError(f"Location not found: '{city}'")
        hit = results[0]
        result = (
            hit["latitude"],
            hit["longitude"],
            f"{hit.get('name', city)}, {hit.get('country', '')}",
        )
        _cache.set(key, result, _cache.TTL_GEOCODE)
        return result


# ── Helper: fetch PM2.5 sensor ID near a location ─────────────────────────────

MAX_SENSOR_STALENESS_DAYS = 30  # skip locations that haven't reported in this long


async def find_pm25_sensor(client: httpx.AsyncClient, lat: float, lon: float) -> int | None:
    """
    Find the nearest ACTIVE station that has a PM2.5 sensor.

    Many stations returned by the OpenAQ /locations search are
    decommissioned (some haven't reported since 2016-2017) but still show
    up in results. We use each location's own datetimeLast field to skip
    stale ones, so we don't hand back a sensor ID with no recent data.
    Returns the sensor ID, or None if no active PM2.5 sensor is found.
    """
    key = f"sensor:{lat:.4f},{lon:.4f}"
    cached = _cache.get(key)
    if cached is not None:
        return cached
    headers = _auth_headers()

    r = await client.get(
        f"{OPENAQ_BASE}/locations",
        params={
            "coordinates": f"{lat},{lon}",
            "radius":      25000,
            "limit":       20,  # wide enough to usually find an active station even
            "order_by":    "id",  # though order_by="id" surfaces oldest-registered (often defunct) ones first
        },
        headers=headers,
    )

    if r.status_code != 200:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_SENSOR_STALENESS_DAYS)
    fallback_sensor_id = None  # used only if every location turns out stale

    for loc in r.json().get("results", []):
        last_str = (loc.get("datetimeLast") or {}).get("utc")
        is_active = False
        if last_str:
            try:
                last_dt = datetime.fromisoformat(last_str.replace("Z", "+00:00"))
                is_active = last_dt >= cutoff
            except ValueError:
                is_active = False

        for sensor in loc.get("sensors") or []:
            if (sensor.get("parameter") or {}).get("name", "").lower() == "pm25":
                if is_active:
                    _cache.set(key, sensor["id"], _cache.TTL_SENSOR_ID)
                    return sensor["id"]
                if fallback_sensor_id is None:
                    fallback_sensor_id = sensor["id"]

    # No active sensor found nearby — return the first stale one we saw so
    # callers still get *something* (and their own data-sufficiency checks
    # will correctly report "not enough data" rather than us silently
    # returning None and masking the real reason).
    if fallback_sensor_id is not None:
        _cache.set(key, fallback_sensor_id, _cache.TTL_SENSOR_ID)
    return fallback_sensor_id


# ── Helper: auth headers ───────────────────────────────────────────────────────

def _auth_headers() -> dict:
    """Return OpenAQ auth headers if a key is available."""
    key = os.getenv("OPENAQ_API_KEY", "")
    return {"X-API-Key": key} if key else {}


# ── Helper: fetch historical PM2.5 readings ────────────────────────────────────

async def fetch_pm25_history(
    client: httpx.AsyncClient,
    sensor_id: int,
    days: int = 30,
) -> list[float]:
    """
    Fetch the last N days of PM2.5 readings for a specific sensor.
    Returns a list of float values. Empty list if unavailable.
    """
    key = f"pm25_history:{sensor_id}:{days}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=days)

    r = await client.get(
        f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements",
        params={
            "date_from":  date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to":    date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit":      1000,
            "order_by":   "datetime",
            "sort_order": "asc",
        },
        headers=_auth_headers(),
    )

    if r.status_code != 200:
        return []

    results = r.json().get("results", [])
    values = [
        float(entry["value"])
        for entry in results
        if entry.get("value") is not None and float(entry["value"]) > 0
    ]
    if values:
        _cache.set(key, values, _cache.TTL_HISTORY)
    return values


# ── Helper: fetch current PM2.5 reading ───────────────────────────────────────

async def fetch_current_pm25(
    client: httpx.AsyncClient,
    sensor_id: int,
) -> float | None:
    """
    Fetch the single most recent PM2.5 reading for a sensor.
    Returns a float, or None if unavailable.
    """
    key = f"pm25_current:{sensor_id}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    r = await client.get(
        f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements",
        params={
            "limit":      1,
            "order_by":   "datetime",
            "sort_order": "desc",
        },
        headers=_auth_headers(),
    )

    if r.status_code != 200:
        return None

    results = r.json().get("results", [])
    if not results:
        return None

    val = results[0].get("value")
    result = float(val) if val is not None else None
    if result is not None:
        _cache.set(key, result, _cache.TTL_CURRENT)
    return result


# ── Core function: detect_anomaly ─────────────────────────────────────────────

async def detect_anomaly(city: str) -> dict:
    """
    Detect whether current PM2.5 is statistically anomalous
    compared to the 30-day historical baseline for a city.

    Returns a structured dictionary with:
      - city: display name
      - current_pm25: today's reading
      - mean_30day: 30-day average
      - std_30day: standard deviation
      - z_score: how many std deviations above mean
      - is_anomaly: True if z > 2.0
      - severity: NORMAL / ELEVATED / ANOMALY / CRITICAL
      - data_points: number of historical readings used
      - message: human-readable summary
      - sufficient_data: True if enough data for valid statistics
    """
    # Step 1: Geocode the city
    try:
        lat, lon, display_name = await geocode(city)
    except ValueError as e:
        return _error_result(str(e))

    async with httpx.AsyncClient(timeout=30) as client:

        # Step 2: Find nearest PM2.5 sensor
        sensor_id = await find_pm25_sensor(client, lat, lon)
        if sensor_id is None:
            return _error_result(f"No PM2.5 sensor found near {city}")

        # Step 3: Fetch historical readings and current reading concurrently
        import asyncio
        history_task = fetch_pm25_history(client, sensor_id, days=30)
        current_task = fetch_current_pm25(client, sensor_id)
        history, current_pm25 = await asyncio.gather(history_task, current_task)

        # Step 4: Check we have enough data
        if len(history) < MIN_DATA_POINTS:
            return {
                "city":            display_name,
                "current_pm25":    current_pm25,
                "sufficient_data": False,
                "data_points":     len(history),
                "severity":        "UNKNOWN",
                "message": (
                    f"Only {len(history)} historical readings available. "
                    f"Need at least {MIN_DATA_POINTS} for valid statistics. "
                    "Returning current reading without anomaly analysis."
                ),
            }

        if current_pm25 is None:
            return _error_result(f"Could not fetch current PM2.5 for {city}")

        # Step 5: Compute statistics using numpy and scipy
        history_array = np.array(history)
        mean_val      = float(np.mean(history_array))
        std_val       = float(np.std(history_array, ddof=1))

        # Guard against zero standard deviation
        # (all readings identical, cannot compute z-score)
        if std_val == 0:
            return _error_result(
                f"Standard deviation is zero for {city}. "
                "All historical readings are identical, cannot compute z-score."
            )

        # Step 6: Compute z-score
        z_score = (current_pm25 - mean_val) / std_val

        # Step 7: Assign severity label
        if z_score >= 4.0:
            severity = "CRITICAL"
        elif z_score >= 3.0:
            severity = "SEVERE"
        elif z_score >= Z_THRESHOLD_ANOM:
            severity = "ANOMALY"
        elif z_score >= Z_THRESHOLD_WARN:
            severity = "ELEVATED"
        else:
            severity = "NORMAL"

        is_anomaly = z_score >= Z_THRESHOLD_ANOM

        # Step 8: Build human-readable message
        direction = "above" if z_score >= 0 else "below"
        message = (
            f"Current PM2.5 ({current_pm25:.1f} µg/m³) is "
            f"{abs(z_score):.2f} standard deviations {direction} "
            f"the 30-day mean ({mean_val:.1f} µg/m³)."
        )

        if is_anomaly:
            message += f" STATUS: {severity}. This is a statistically significant spike."
        else:
            message += " Air quality is within normal historical range for this city."

        return {
            "city":            display_name,
            "current_pm25":    round(current_pm25, 1),
            "mean_30day":      round(mean_val, 1),
            "std_30day":       round(std_val, 1),
            "z_score":         round(z_score, 2),
            "is_anomaly":      is_anomaly,
            "severity":        severity,
            "data_points":     len(history),
            "sufficient_data": True,
            "message":         message,
            "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }


# ── Helper: error result ───────────────────────────────────────────────────────

def _error_result(reason: str) -> dict:
    """Return a standardised error dictionary."""
    return {
        "city":            "Unknown",
        "sufficient_data": False,
        "severity":        "ERROR",
        "message":         f"Anomaly detection failed: {reason}",
        "data_points":     0,
    }