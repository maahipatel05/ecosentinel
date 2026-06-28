"""
EcoSentinel Causal Inference Engine
=====================================
Determines whether a city's current PM2.5 reading is CAUSALLY explained by
nearby wildfire smoke, as opposed to merely correlated with it.

Method: builds a real (not simulated) dataset of paired daily observations
over the last PAIRED_WINDOW_DAYS days, then uses DoWhy to compute
P(Y | do(X)) via the backdoor criterion:

  X = wildfire intensity reaching the city that day (brightness / distance)
  Y = that day's mean PM2.5 reading
  W = that day's wind speed (confounder: wind affects both smoke transport
      and background pollution dispersal)

DAG:  W -> X,  W -> Y,  X -> Y

If fewer than MIN_PAIRED_OBSERVATIONS valid days are available, this module
gracefully falls back to anomaly.py's z-score result instead of guessing.
"""

import os
import csv
import io
import math
import asyncio
import warnings
import numpy as np
import pandas as pd
import httpx
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
from dowhy import CausalModel

from anomaly import find_pm25_sensor, fetch_current_pm25, detect_anomaly
import cache as _cache

warnings.filterwarnings("ignore")

load_dotenv(Path(__file__).parent.parent / ".env")

OPENAQ_BASE = "https://api.openaq.org/v3"
GEOCODE_BASE = "https://geocoding-api.open-meteo.com/v1"
METEO_ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1"
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area"

PAIRED_WINDOW_DAYS = 20   # how many past days we try to assemble data for
FIRMS_CHUNK_DAYS = 5    # NASA FIRMS area API's max day_range per call
MIN_PAIRED_OBSERVATIONS = 15  # minimum complete days required to run DoWhy
RETRY_DELAYS = [1, 2, 4]  # exponential backoff schedule, in seconds


# ── Helper: retry wrapper with exponential backoff ──────────────────────────

async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response | None:
    """
    GET a URL with up to 1 + len(RETRY_DELAYS) attempts. Waits 1s, 2s, 4s
    between retries. Retries on connection failures and 5xx server errors;
    does NOT retry on 4xx (a bad request will not fix itself by waiting).
    Returns the last response received, or None if every attempt failed
    to even connect.
    """
    last_response = None
    for delay in [0] + RETRY_DELAYS:
        if delay:
            await asyncio.sleep(delay)
        try:
            response = await client.get(url, params=params, headers=headers)
        except httpx.TransportError:
            continue
        if response.status_code < 500:
            return response
        last_response = response
    return last_response


# ── Helper: geocode ──────────────────────────────────────────────────────────

async def geocode(city: str) -> tuple[float, float, str]:
    """Convert a city name into (latitude, longitude, display_name)."""
    key = f"geocode:{city.lower().strip()}"
    cached = _cache.get(key)
    if cached is not None:
        return cached
    async with httpx.AsyncClient(timeout=10) as client:
        r = await _get_with_retry(
            client,
            f"{GEOCODE_BASE}/search",
            params={"name": city, "count": 1, "format": "json"},
        )
        if r is None or r.status_code != 200:
            raise ValueError(f"Could not geocode '{city}'")
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


# ── Helper: auth headers ──────────────────────────────────────────────────────

def _auth_headers() -> dict:
    key = os.getenv("OPENAQ_API_KEY", "")
    return {"X-API-Key": key} if key else {}


# ── Helper: distance between two coordinates ─────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


# ── Data fetch: wildfire hotspots, grouped by calendar day ──────────────────

async def fetch_fire_days(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    radius_km: int = 500,
    window_days: int = PAIRED_WINDOW_DAYS,
) -> dict[str, list[dict]]:
    """
    Fetch NASA FIRMS wildfire hotspots for the last window_days days,
    grouped by the calendar date they were detected (acq_date).

    NASA FIRMS' area API only allows a 5-day range per request, so we page
    backwards in 5-day chunks to cover the full window. Returns a dict
    mapping "YYYY-MM-DD" -> list of {lat, lon, brightness}. Every date in
    the window is guaranteed a key (an empty list means "no fire detected
    that day", which is itself a real, meaningful observation).
    """
    key = f"fire_days:{lat:.4f},{lon:.4f}:{radius_km}:{window_days}"
    cached = _cache.get(key)
    if cached is not None:
        return cached
    api_key = os.getenv("NASA_FIRMS_API_KEY", "DEMO_KEY")
    deg = radius_km / 111.0
    bbox = f"{lon - deg:.4f},{lat - deg:.4f},{lon + deg:.4f},{lat + deg:.4f}"
    today = datetime.now(timezone.utc).date()

    days_by_date: dict[str, list[dict]] = {}

    offset = 0
    while offset < window_days:
        chunk = min(FIRMS_CHUNK_DAYS, window_days - offset)
        end_date = today - timedelta(days=offset)
        url = f"{FIRMS_BASE}/csv/{api_key}/VIIRS_SNPP_NRT/{bbox}/{chunk}"
        if offset > 0:
            url += f"/{end_date.isoformat()}"

        r = await _get_with_retry(client, url)
        if r is not None and r.status_code == 200:
            reader = csv.DictReader(io.StringIO(r.text))
            for row in reader:
                try:
                    date_str = row["acq_date"]
                    days_by_date.setdefault(date_str, []).append({
                        "lat":        float(row["latitude"]),
                        "lon":        float(row["longitude"]),
                        "brightness": float(row["bright_ti4"]),
                    })
                except (KeyError, ValueError):
                    continue
        offset += FIRMS_CHUNK_DAYS

    for i in range(window_days):
        date_str = (today - timedelta(days=i)).isoformat()
        days_by_date.setdefault(date_str, [])

    _cache.set(key, days_by_date, _cache.TTL_FIRE)
    return days_by_date


# ── Data fetch: daily wind speed (historical archive) ────────────────────────

async def fetch_wind_daily(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    window_days: int = PAIRED_WINDOW_DAYS,
) -> dict[str, float]:
    """
    Fetch daily max wind speed (km/h) for the last window_days days from
    Open-Meteo's free historical weather archive. Returns a dict mapping
    "YYYY-MM-DD" -> wind_speed_kmh.
    """
    key = f"wind_daily:{lat:.4f},{lon:.4f}:{window_days}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=window_days - 1)

    r = await _get_with_retry(
        client,
        f"{METEO_ARCHIVE_BASE}/archive",
        params={
            "latitude":   lat,
            "longitude":  lon,
            "daily":      "wind_speed_10m_max",
            "timezone":   "UTC",
            "start_date": start.isoformat(),
            "end_date":   today.isoformat(),
        },
    )
    if r is None or r.status_code != 200:
        return {}

    daily = r.json().get("daily", {})
    dates = daily.get("time", [])
    speeds = daily.get("wind_speed_10m_max", [])
    result = {d: s for d, s in zip(dates, speeds) if s is not None}
    if result:
        _cache.set(key, result, _cache.TTL_WIND)
    return result


# ── Data fetch: daily PM2.5 readings, grouped by calendar day ───────────────

async def fetch_pm25_daily(
    client: httpx.AsyncClient,
    sensor_id: int,
    window_days: int = PAIRED_WINDOW_DAYS,
) -> dict[str, list[float]]:
    """
    Fetch raw PM2.5 measurements for a sensor over the last window_days
    days and group them by calendar date. Returns a dict mapping
    "YYYY-MM-DD" -> list of readings for that day (averaged later).
    """
    key = f"pm25_daily:{sensor_id}:{window_days}"
    cached = _cache.get(key)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    date_from = now - timedelta(days=window_days)

    # sort_order="desc" anchors this page of results to "now" and walks
    # backward. We deliberately do NOT rely on date_from/date_to alone to
    # scope the window: OpenAQ v3's date filtering has proven unreliable in
    # practice (it has returned data over a year outside the requested
    # range), and with sort_order="asc" a frequently-reporting sensor can
    # exhaust the 1000-row limit before ever reaching recent dates. Walking
    # backward from "now" guarantees whatever we get is the most recent
    # data available, even if date filtering silently does nothing.
    r = await _get_with_retry(
        client,
        f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements",
        params={
            "date_from":  date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_to":    now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit":      1000,
            "order_by":   "datetime",
            "sort_order": "desc",
        },
        headers=_auth_headers(),
    )
    if r is None or r.status_code != 200:
        return {}

    # Defensive check: OpenAQ v3's date_from/date_to filtering has proven
    # unreliable for some sensors (it has returned data over a year outside
    # the requested window in practice) — never trust it blindly, so we
    # re-filter client-side against the window we actually asked for.
    window_start_str = date_from.date().isoformat()
    window_end_str = now.date().isoformat()

    by_date: dict[str, list[float]] = {}
    for entry in r.json().get("results", []):
        value = entry.get("value")
        ts = (entry.get("period") or {}).get("datetimeFrom") or {}
        ts = ts.get("utc")
        if value is None or ts is None or float(value) <= 0:
            continue
        date_str = ts[:10]
        if not (window_start_str <= date_str <= window_end_str):
            continue
        by_date.setdefault(date_str, []).append(float(value))

    if by_date:
        _cache.set(key, by_date, _cache.TTL_HISTORY)
    return by_date


# ── Core: assemble the paired (X, W, Y) dataset ──────────────────────────────

def build_paired_dataset(
    city_lat: float,
    city_lon: float,
    fire_days: dict[str, list[dict]],
    wind_daily: dict[str, float],
    pm25_daily: dict[str, list[float]],
) -> pd.DataFrame:
    """
    Build one row per calendar day that has BOTH a wind reading and a
    PM2.5 reading. Wildfire intensity (X) defaults to 0.0 on days with no
    detected hotspot — absence of a fire is a real value, not missing data.

    X = max over that day's hotspots of brightness / (1 + distance_km)
        (closer + hotter fires contribute more)
    W = that day's max wind speed (km/h)
    Y = that day's mean PM2.5 reading
    """
    rows = []
    for date_str, pm_values in pm25_daily.items():
        if date_str not in wind_daily:
            continue

        hotspots = fire_days.get(date_str, [])
        if hotspots:
            x_val = max(
                h["brightness"] / \
                    (1.0 + haversine_km(city_lat,
                     city_lon, h["lat"], h["lon"]))
                for h in hotspots
            )
        else:
            x_val = 0.0

        rows.append({
            "date": date_str,
            "X":    x_val,
            "W":    wind_daily[date_str],
            "Y":    float(np.mean(pm_values)),
        })

    if not rows:
        return pd.DataFrame(columns=["date", "X", "W", "Y"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ── Shared causal graph — used by both estimate and refutation functions ─────

_CAUSAL_GRAPH_GML = """
graph [
    directed 1
    node [ id "X" label "X" ]
    node [ id "W" label "W" ]
    node [ id "Y" label "Y" ]
    edge [ source "W" target "X" ]
    edge [ source "W" target "Y" ]
    edge [ source "X" target "Y" ]
]
"""


# ── Core: run DoWhy on the assembled dataset ─────────────────────────────────

def run_dowhy_estimate(df: pd.DataFrame) -> float | None:
    """
    Construct the causal DAG (W -> X, W -> Y, X -> Y), identify the
    backdoor adjustment for W, and estimate the Average Treatment Effect
    of X on Y via linear regression. Returns the ATE in µg/m³ per unit
    of fire intensity, or None if estimation fails.
    """
    gml_graph = _CAUSAL_GRAPH_GML
    try:
        model = CausalModel(
            data=df[["X", "W", "Y"]],
            treatment="X",
            outcome="Y",
            graph=gml_graph,
        )
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=False,
        )
        return float(estimate.value)
    except Exception:
        return None


# ── Core: validate the ATE with DoWhy refutation tests ──────────────────────

def run_dowhy_refutations(df: pd.DataFrame, ate: float) -> dict:
    """
    Run 4 standard DoWhy refutation tests to verify the causal ATE is robust
    and not an artefact of model structure, hidden confounders, or sample size.

    Each test stress-tests a different assumption:
      random_common_cause — add a random extra confounder; ATE should barely change
      placebo_treatment   — replace X with random noise; ATE should collapse to ~0
      data_subset         — re-estimate on a random 80% subsample; ATE should be stable
      bootstrap           — resample with replacement; ATE should be stable

    Pass criteria (calibrated for small real-world datasets):
      placebo:    |new_ate| < 20% of |ate|        (effect vanishes when X is fake)
      all others: |new_ate - ate| < 30% of |ate|  (effect is stable under perturbation)

    Returns a dict keyed by refuter name. Each entry has:
      new_ate (float|None) — re-estimated ATE after the manipulation
      passed  (bool|None)  — True = passes this robustness check; None = refuter errored
    """
    try:
        model = CausalModel(
            data=df[["X", "W", "Y"]],
            treatment="X",
            outcome="Y",
            graph=_CAUSAL_GRAPH_GML,
        )
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=False,
        )
    except Exception:
        return {}

    refuter_specs = [
        ("random_common_cause", "random_common_cause"),
        ("placebo_treatment",   "placebo_treatment_refuter"),
        ("data_subset",         "data_subset_refuter"),
        ("bootstrap",           "bootstrap_refuter"),
    ]

    results = {}
    for name, method in refuter_specs:
        try:
            ref = model.refute_estimate(
                identified_estimand, estimate, method_name=method)
            new_ate = float(ref.new_effect)
            if abs(ate) < 1e-6:
                passed = True
            elif name == "placebo_treatment":
                passed = abs(new_ate) < 0.2 * abs(ate)
            else:
                passed = abs(new_ate - ate) < 0.3 * abs(ate)
            results[name] = {"new_ate": round(new_ate, 4), "passed": passed}
        except Exception:
            results[name] = {"new_ate": None, "passed": None}

    return results


# ── Pure helper: convert ATE into a 0-1 causal probability ──────────────────

MIN_MEANINGFUL_EXCESS = 1.0  # µg/m³ — below this, there's no real anomaly to explain


def compute_causal_probability(ate: float, x_now: float, excess: float) -> float:
    """
    Convert a raw ATE (µg/m³ per unit fire intensity) into a 0.0-1.0
    probability that wildfire explains the CURRENT PM2.5 excess.

    contribution = ate * x_now   (predicted µg/m³ from today's fire intensity)
    probability  = contribution / excess, clipped to [0.0, 1.0]

    Guards against excess being tiny-but-nonzero: dividing two
    near-zero, noise-dominated numbers can spuriously blow up toward
    1.0 even when there's no real signal, so below MIN_MEANINGFUL_EXCESS
    we report "nothing to explain" instead of an unstable ratio.
    """
    if excess < MIN_MEANINGFUL_EXCESS:
        return 0.0
    return float(np.clip((ate * x_now) / excess, 0.0, 1.0))


# ── Helper: standardised fallback result ─────────────────────────────────────

def _fallback(display_name: str, anomaly_result: dict, reason: str) -> dict:
    return {
        "city":              display_name,
        "causal_probability": None,
        "is_causal":          False,
        "confidence":         "N/A",
        "z_score":            anomaly_result.get("z_score"),
        "method":             "DoWhy backdoor criterion (linear regression)",
        "message":            f"{reason} {anomaly_result.get('message', '')}".strip(),
        "sufficient_data":    False,
        "timestamp":          _now(),
    }


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── Public entry point ───────────────────────────────────────────────────────

async def get_causal_attribution(city: str) -> dict:
    """
    Determine whether nearby wildfires are causally responsible for a
    city's current PM2.5 reading, using DoWhy + the backdoor criterion
    on real paired daily observations. Falls back gracefully to the
    z-score anomaly result if there isn't enough paired data.
    """
    try:
        lat, lon, display_name = await geocode(city)
    except ValueError as e:
        return {
            "city": city, "causal_probability": None, "is_causal": False,
            "confidence": "N/A", "z_score": None,
            "method": "DoWhy backdoor criterion (linear regression)",
            "message": str(e), "sufficient_data": False, "timestamp": _now(),
        }

    # The z-score check is always run: it gives us z_score for the final
    # result regardless of which path we take, and is also our fallback.
    anomaly_result = await detect_anomaly(city)

    async with httpx.AsyncClient(timeout=30) as client:
        sensor_id = await find_pm25_sensor(client, lat, lon)
        if sensor_id is None:
            return _fallback(display_name, anomaly_result, "No PM2.5 sensor found near this city.")

        fire_days, wind_daily, pm25_daily, current_pm25 = await asyncio.gather(
            fetch_fire_days(client, lat, lon, window_days=PAIRED_WINDOW_DAYS),
            fetch_wind_daily(client, lat, lon, window_days=PAIRED_WINDOW_DAYS),
            fetch_pm25_daily(client, sensor_id,
                             window_days=PAIRED_WINDOW_DAYS),
            fetch_current_pm25(client, sensor_id),
        )

    df = build_paired_dataset(lat, lon, fire_days, wind_daily, pm25_daily)

    if len(df) < MIN_PAIRED_OBSERVATIONS:
        return _fallback(
            display_name, anomaly_result,
            f"Only {len(df)} days had complete wildfire + wind + PM2.5 data "
            f"(need at least {MIN_PAIRED_OBSERVATIONS} within the last "
            f"{PAIRED_WINDOW_DAYS} days)."
        )

    ate = run_dowhy_estimate(df)
    if ate is None:
        return _fallback(display_name, anomaly_result, "DoWhy estimation failed on the available data.")

    refutation_results = run_dowhy_refutations(df, ate)
    refutations_passed = sum(
        1 for r in refutation_results.values() if r.get("passed") is True)

    baseline = float(df["Y"].mean())
    excess = max((current_pm25 or 0.0) - baseline, 0.0)

    most_recent_date = max(df["date"])
    x_now = float(df.loc[df["date"] == most_recent_date, "X"].iloc[0])

    causal_probability = compute_causal_probability(ate, x_now, excess)
    is_causal = causal_probability > 0.5

    if df["X"].std() < 1e-6:
        confidence = "LOW"
    elif len(df) >= 18:
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    pct = round(causal_probability * 100, 1)
    ref_summary = (
        f" Causal estimate passed {refutations_passed}/{len(refutation_results)} refutation tests."
        if refutation_results else ""
    )
    message = (
        f"DoWhy estimates a {pct}% causal probability that nearby wildfires explain "
        f"the current PM2.5 reading, after controlling for wind speed as a confounder "
        f"(ATE = {round(ate, 4)} µg/m³ per unit of fire intensity, based on "
        f"{len(df)} paired daily observations over the last {PAIRED_WINDOW_DAYS} days)."
        + ref_summary
    )

    return {
        "city":                display_name,
        "causal_probability":  round(causal_probability, 3),
        "is_causal":           is_causal,
        "confidence":          confidence,
        "z_score":             anomaly_result.get("z_score"),
        "method":              "DoWhy backdoor criterion (linear regression)",
        "message":             message,
        "sufficient_data":     True,
        "paired_observations": len(df),
        "ate":                 round(ate, 4),
        "refutation_results":  refutation_results,
        "refutations_passed":  refutations_passed,
        "total_refutations":   len(refutation_results),
        "timestamp":           _now(),
    }
