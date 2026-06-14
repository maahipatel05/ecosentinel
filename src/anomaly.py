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
        return (
            hit["latitude"],
            hit["longitude"],
            f"{hit.get('name', city)}, {hit.get('country', '')}",
        )


# ── Helper: fetch PM2.5 sensor ID near a location ─────────────────────────────

async def find_pm25_sensor(client: httpx.AsyncClient, lat: float, lon: float) -> int | None:
    """
    Find the nearest station that has a PM2.5 sensor.
    Returns the sensor ID, or None if not found.
    """
    headers = _auth_headers()

    r = await client.get(
        f"{OPENAQ_BASE}/locations",
        params={
            "coordinates": f"{lat},{lon}",
            "radius":      25000,
            "limit":       5,
            "order_by":    "id",
        },
        headers=headers,
    )

    if r.status_code != 200:
        return None

    for loc in r.json().get("results", []):
        for sensor in loc.get("sensors", []):
            if sensor.get("parameter", {}).get("name", "").lower() == "pm25":
                return sensor["id"]

    return None


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
    return [
        float(entry["value"])
        for entry in results
        if entry.get("value") is not None and float(entry["value"]) > 0
    ]


# ── Helper: fetch current PM2.5 reading ───────────────────────────────────────

async def fetch_current_pm25(
    client: httpx.AsyncClient,
    sensor_id: int,
) -> float | None:
    """
    Fetch the single most recent PM2.5 reading for a sensor.
    Returns a float, or None if unavailable.
    """
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
    return float(val) if val is not None else None


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