"""
EcoSentinel MCP Server
======================
An MCP (Model Context Protocol) server that gives AI assistants real-time
access to environmental crisis data: air quality, wildfires, and flood risk.

Tools exposed:
  - get_air_quality      : Live AQI + pollutant data for any city
  - get_wildfires        : Active wildfire hotspots near a location
  - get_weather_risk     : Current weather + flood/storm risk
  - get_crisis_summary   : Full environmental briefing for any location
"""

import os
import httpx
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import cache as _cache

from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

# MCP server instance
mcp = FastMCP(
    name="EcoSentinel",
    instructions=(
        "You are EcoSentinel, an environmental crisis monitoring assistant. "
        "You have access to real-time air quality, wildfire, and weather/flood risk data. "
        "Always cite the data source and timestamp in your answers. "
        "When risk levels are high, clearly flag them and suggest actionable steps."
    ),
)

NASA_FIRMS_KEY = os.getenv("NASA_FIRMS_API_KEY", "")
OPENAQ_BASE = "https://api.openaq.org/v3"
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area"
METEO_BASE = "https://api.open-meteo.com/v1"
GEOCODE_BASE = "https://geocoding-api.open-meteo.com/v1"


# Helpers


async def geocode(city: str) -> tuple[float, float, str]:
    """Return (lat, lon, display_name) for a city string."""
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
        name = f"{hit.get('name', city)}, {hit.get('country', '')}"
        result = hit["latitude"], hit["longitude"], name
        _cache.set(key, result, _cache.TTL_GEOCODE)
        return result


def aqi_label(aqi: float) -> str:
    if aqi <= 50:
        return "✅ Good"
    if aqi <= 100:
        return "🟡 Moderate"
    if aqi <= 150:
        return "🟠 Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "🔴 Unhealthy"
    if aqi <= 300:
        return "🟣 Very Unhealthy"
    return "🟤 Hazardous"


def wind_label(speed_kmh: float) -> str:
    if speed_kmh < 20:
        return "Calm"
    if speed_kmh < 40:
        return "Breezy"
    if speed_kmh < 60:
        return "Windy"
    if speed_kmh < 90:
        return "Strong winds"
    return "⚠️ Storm-force winds"


# Tool 1: Air Quality


@mcp.tool()
async def get_air_quality(city: str) -> str:
    """
    Get real-time air quality data for any city in the world.

    Returns AQI, PM2.5, PM10, NO2, CO levels and a health risk assessment.
    Data sourced from OpenAQ (global sensor network).

    Args:
        city: City name, e.g. 'Delhi', 'London', 'Los Angeles'
    """
    cache_key = f"tool:air_quality:{city.lower().strip()}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        lat, lon, display_name = await geocode(city)
    except ValueError as e:
        return f"❌ {e}"

    api_key = os.getenv("OPENAQ_API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}

    param_labels = {
        "pm25": "PM2.5 (fine particles)",
        "pm10": "PM10 (coarse particles)",
        "no2": "NO₂ (nitrogen dioxide)",
        "co": "CO (carbon monoxide)",
        "o3": "O₃ (ozone)",
        "so2": "SO₂ (sulphur dioxide)",
    }

    async with httpx.AsyncClient(timeout=20) as client:

        # Step 1: Find nearby stations
        r = await client.get(
            f"{OPENAQ_BASE}/locations",
            params={
                "coordinates": f"{lat},{lon}",
                "radius": 25000,
                "limit": 5,
                "order_by": "id",
            },
            headers=headers,
        )

        if r.status_code != 200:
            return f"❌ OpenAQ API error: {r.status_code}. Try a larger city nearby."

        locations = r.json().get("results", [])

        if not locations:
            return (
                f"No air quality sensors found within 25 km of **{display_name}**.\n"
                "Try a larger nearby city."
            )

        # Step 2: Collect one sensor ID per pollutant type
        sensor_ids = {}
        station_names = []

        for loc in locations[:3]:
            station_names.append(loc.get("name", "Unknown station"))
            for sensor in loc.get("sensors", []):
                param = sensor.get("parameter", {})
                pname = param.get("name", "").lower()
                if pname in param_labels and pname not in sensor_ids:
                    sensor_ids[pname] = (sensor["id"], param.get("units", ""))

        if not sensor_ids:
            return f"Stations found near **{display_name}** but no compatible sensors detected."

        # Step 3: Fetch latest measurement for each sensor concurrently
        async def fetch_sensor(pname, sid, unit):
            try:
                resp = await client.get(
                    f"{OPENAQ_BASE}/sensors/{sid}/measurements",
                    params={"limit": 1, "order_by": "datetime", "sort_order": "desc"},
                    headers=headers,
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if results:
                        val = results[0].get("value")
                        if val is not None:
                            return pname, float(val), unit
            except Exception:
                pass
            return pname, None, unit

        tasks = [fetch_sensor(p, sid, unit) for p, (sid, unit) in sensor_ids.items()]
        raw_results = await asyncio.gather(*tasks)
        pollutants = {p: (v, u) for p, v, u in raw_results if v is not None}

        if not pollutants:
            return (
                f"Stations found near **{display_name}** but readings are older than 24 hours.\n"
                "This city may have limited real-time coverage on OpenAQ right now."
            )

        # Step 4: Estimate AQI from PM2.5
        aqi_str = ""
        pm25_val = pollutants.get("pm25", (None,))[0]
        if pm25_val is not None:
            if pm25_val <= 12:
                aqi = pm25_val / 12 * 50
            elif pm25_val <= 35.4:
                aqi = 50 + (pm25_val - 12) / 23.4 * 50
            elif pm25_val <= 55.4:
                aqi = 100 + (pm25_val - 35.4) / 20 * 50
            elif pm25_val <= 150.4:
                aqi = 150 + (pm25_val - 55.4) / 95 * 50
            else:
                aqi = 200 + (pm25_val - 150.4) / 149.6 * 100
            aqi = round(aqi)
            aqi_str = f"\n**Estimated AQI**: {aqi} — {aqi_label(aqi)}"

        # Step 5: Format output
        lines = [
            f"## 🌫️ Air Quality — {display_name}",
            f"*Data from OpenAQ | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            aqi_str,
            "\n**Pollutant Levels:**",
        ]

        for param, (val, unit) in pollutants.items():
            label = param_labels.get(param, param.upper())
            lines.append(f"  • {label}: **{val:.1f} {unit}**")

        lines += [
            f"\n**Stations sampled**: {', '.join(station_names[:3])}",
            f"**Coordinates**: {lat:.4f}°N, {lon:.4f}°E",
            "\n*Source: OpenAQ global sensor network — openaq.org*",
        ]

        result = "\n".join(lines)
        _cache.set(cache_key, result, _cache.TTL_TOOL)
        return result


# Tool 2: Wildfires


@mcp.tool()
async def get_wildfires(
    city: str,
    radius_km: int = 500,
    days: int = 2,
) -> str:
    """
    Get active wildfire hotspots detected by NASA satellites near a location.

    Uses NASA FIRMS (Fire Information for Resource Management System).

    Args:
        city:      City or region to check, e.g. 'Sydney', 'California'
        radius_km: Search radius in km (default 500)
        days:      How many days back to check (1-10, default 2)
    """
    cache_key = f"tool:wildfires:{city.lower().strip()}:{radius_km}:{days}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        lat, lon, display_name = await geocode(city)
    except ValueError as e:
        return f"❌ {e}"

    days = max(1, min(days, 10))
    api_key = NASA_FIRMS_KEY or "DEMO_KEY"

    url = (
        f"{FIRMS_BASE}/csv/{api_key}/VIIRS_SNPP_NRT/"
        f"{lon - radius_km/111:.4f},{lat - radius_km/111:.4f},"
        f"{lon + radius_km/111:.4f},{lat + radius_km/111:.4f}/{days}"
    )

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url)

    if r.status_code == 400:
        return (
            "⚠️ NASA FIRMS API requires a free API key.\n"
            "Get yours free at: https://firms.modaps.eosdis.nasa.gov/api/area/\n"
            "Then add it to your .env file as NASA_FIRMS_API_KEY=your_key"
        )

    if r.status_code != 200:
        return f"❌ NASA FIRMS API error: {r.status_code}"

    lines_raw = r.text.strip().split("\n")

    if len(lines_raw) <= 1:
        result = (
            f"## 🔥 Wildfires near {display_name}\n"
            f"*NASA FIRMS | Last {days} day(s) | {radius_km} km radius*\n\n"
            f"✅ **No active fire hotspots detected** within {radius_km} km.\n"
            "*Source: NASA VIIRS SNPP satellite — firms.modaps.eosdis.nasa.gov*"
        )
        _cache.set(cache_key, result, _cache.TTL_FIRE)
        return result

    hotspots = []
    for row in lines_raw[1:]:
        cols = row.split(",")
        if len(cols) < 6:
            continue
        try:
            hotspots.append(
                {
                    "lat": float(cols[0]),
                    "lon": float(cols[1]),
                    "brightness": float(cols[2]),
                    "confidence": cols[8].strip() if len(cols) > 8 else "n/a",
                    "date": cols[5].strip() if len(cols) > 5 else "n/a",
                }
            )
        except (ValueError, IndexError):
            continue

    high_conf = [
        h for h in hotspots if h["confidence"] in ("high", "h", "100", "nominal")
    ]

    output = [
        f"## 🔥 Wildfire Alert — {display_name}",
        f"*NASA FIRMS VIIRS | Last {days} day(s) | {radius_km} km radius | "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        f"**Total hotspots detected**: {len(hotspots)}",
        f"**High-confidence fires**: {len(high_conf)}",
    ]

    if len(hotspots) > 0:
        risk = (
            "🔴 HIGH"
            if len(hotspots) > 20
            else ("🟡 MODERATE" if len(hotspots) > 5 else "🟢 LOW")
        )
        output.append(f"**Fire risk level**: {risk}")
        output.append("")
        output.append("**Recent hotspots (sample):**")
        for h in hotspots[:8]:
            output.append(
                f"  • {h['date']} | {h['lat']:.3f}°, {h['lon']:.3f}° "
                f"| Brightness: {h['brightness']:.0f}K | Confidence: {h['confidence']}"
            )
        if len(hotspots) > 8:
            output.append(f"  • ...and {len(hotspots) - 8} more hotspots")

    output.append("\n*Source: NASA FIRMS VIIRS SNPP — firms.modaps.eosdis.nasa.gov*")
    result = "\n".join(output)
    _cache.set(cache_key, result, _cache.TTL_FIRE)
    return result


# Tool 3: Weather and Flood Risk


@mcp.tool()
async def get_weather_risk(city: str) -> str:
    """
    Get current weather conditions and assess flood/storm risk for any city.

    Uses Open-Meteo (free, no API key needed).

    Args:
        city: City name, e.g. 'Mumbai', 'Houston', 'Bangladesh'
    """
    cache_key = f"tool:weather_risk:{city.lower().strip()}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        lat, lon, display_name = await geocode(city)
    except ValueError as e:
        return f"❌ {e}"

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{METEO_BASE}/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,precipitation,rain,wind_speed_10m,wind_gusts_10m,weathercode",
                "daily": "precipitation_sum,rain_sum,wind_speed_10m_max,weathercode",
                "forecast_days": 3,
                "timezone": "auto",
            },
        )
        r.raise_for_status()

    d = r.json()
    cur = d.get("current", {})
    daily = d.get("daily", {})

    temp = cur.get("temperature_2m", "N/A")
    precip = cur.get("precipitation", 0)
    wind = cur.get("wind_speed_10m", 0)
    gusts = cur.get("wind_gusts_10m", 0)
    wcode = cur.get("weathercode", 0)

    daily_precip = daily.get("precipitation_sum", [0, 0, 0])
    total_3day = sum(p for p in daily_precip if p) if daily_precip else 0

    flood_risk = "🟢 Low"
    if total_3day > 100 or precip > 20:
        flood_risk = "🔴 HIGH — significant rainfall, flooding possible"
    elif total_3day > 50 or precip > 10:
        flood_risk = "🟡 Moderate — monitor water levels"

    storm_risk = "🟢 Low"
    if gusts > 90:
        storm_risk = "🔴 HIGH — storm-force gusts, seek shelter"
    elif gusts > 60:
        storm_risk = "🟡 Moderate — strong gusts, exercise caution"

    wcode_desc = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        51: "Light drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        80: "Rain showers",
        95: "Thunderstorm",
        99: "Thunderstorm with hail",
    }.get(wcode, f"Code {wcode}")

    lines = [
        f"## 🌦️ Weather & Flood Risk — {display_name}",
        f"*Open-Meteo | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        f"**Current conditions**: {wcode_desc}",
        f"**Temperature**: {temp}°C",
        f"**Precipitation (now)**: {precip} mm",
        f"**Wind speed**: {wind} km/h ({wind_label(wind)})",
        f"**Wind gusts**: {gusts} km/h",
        "",
        "**3-Day Forecast Totals:**",
    ]

    days_label = daily.get("time", ["Day 1", "Day 2", "Day 3"])
    for day, p in zip(days_label[:3], daily_precip[:3]):
        bar = "█" * min(int((p or 0) / 5), 20)
        lines.append(f"  {day}: {p or 0:.1f} mm  {bar}")

    lines += [
        "",
        f"**Flood risk assessment**: {flood_risk}",
        f"**Storm risk assessment**: {storm_risk}",
        f"**Total 3-day rainfall**: {total_3day:.1f} mm",
        "",
        "*Source: Open-Meteo (open-meteo.com)*",
    ]

    result = "\n".join(lines)
    _cache.set(cache_key, result, _cache.TTL_TOOL)
    return result


# Tool 4: Crisis Summary


@mcp.tool()
async def get_crisis_summary(city: str) -> str:
    """
    Generate a comprehensive environmental crisis briefing for any location.

    Combines air quality, wildfire, and weather/flood risk data into a single
    synthesized report using all three data sources simultaneously.

    Args:
        city: City or region name, e.g. 'Jakarta', 'Cape Town', 'Amazon'
    """
    air_result, fire_result, weather_result = await asyncio.gather(
        get_air_quality(city),
        get_wildfires(city),
        get_weather_risk(city),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""# 🌍 EcoSentinel Crisis Briefing — {city.title()}
*Generated: {timestamp} | Powered by EcoSentinel MCP*

---

{air_result}

---

{fire_result}

---

{weather_result}

---

## 📋 Summary

Data sources:
- **OpenAQ** — global ground-level air quality sensors
- **NASA FIRMS** — satellite-detected wildfire hotspots (VIIRS SNPP)
- **Open-Meteo** — weather forecast and precipitation data

*For emergencies, always contact local authorities. This tool is for awareness and research.*
*EcoSentinel — github.com/maahipatel05/ecosentinel*
"""


# ── Tool 5: Anomaly Detection ──────────────────────────────────────────────────


@mcp.tool()
async def detect_anomaly_tool(city: str) -> str:
    """
    Detect whether current PM2.5 air quality is statistically anomalous
    for a city based on its 30-day historical baseline.

    Uses z-score analysis: compares today's reading to the mean and
    standard deviation of the last 30 days of sensor data.

    Args:
        city: City name, e.g. 'Delhi', 'London', 'Ahmedabad'
    """
    from anomaly import detect_anomaly

    result = await detect_anomaly(city)

    if result.get("severity") == "ERROR":
        return f"❌ {result['message']}"

    if not result.get("sufficient_data"):
        return (
            f"## 📊 Anomaly Detection — {result.get('city', city)}\n\n"
            f"⚠️ {result['message']}"
        )

    severity_icons = {
        "NORMAL": "✅",
        "ELEVATED": "🟡",
        "ANOMALY": "🔴",
        "SEVERE": "🟣",
        "CRITICAL": "🚨",
    }

    icon = severity_icons.get(result["severity"], "⚪")

    lines = [
        f"## 📊 Anomaly Detection — {result['city']}",
        f"*Z-score analysis | 30-day baseline | {result.get('timestamp', '')}*",
        "",
        f"**Status**: {icon} {result['severity']}",
        f"**Current PM2.5**: {result['current_pm25']} µg/m³",
        f"**30-day mean**: {result['mean_30day']} µg/m³",
        f"**30-day std dev**: {result['std_30day']} µg/m³",
        f"**Z-score**: {result['z_score']}",
        f"**Data points used**: {result['data_points']} readings",
        "",
        f"**Analysis**: {result['message']}",
        "",
        "*Method: Z-score analysis. z > 2.0 = anomaly. z > 3.0 = severe. z > 4.0 = critical.*",
        "*Source: OpenAQ historical sensor data*",
    ]

    return "\n".join(lines)


# ── Tool 6: Causal Attribution ────────────────────────────────────────────────


@mcp.tool()
async def get_causal_attribution(city: str) -> str:
    """
    Determine whether a wildfire is causally responsible for a city's
    current PM2.5 anomaly, using DoWhy causal inference (not correlation).

    Builds ~20 days of real paired daily observations (wildfire intensity,
    wind speed, PM2.5) and applies the backdoor criterion to estimate
    P(Y | do(X)): how much of today's PM2.5 is actually caused by nearby
    wildfire smoke, after controlling for wind. Falls back to a z-score
    anomaly check if there isn't enough paired data for a valid estimate.

    Args:
        city: City name, e.g. 'Delhi', 'Sydney', 'Jakarta'
    """
    from causal import get_causal_attribution as run_causal_attribution

    result = await run_causal_attribution(city)

    z_line = (
        f"**Z-score (anomaly check)**: {result['z_score']}"
        if result.get("z_score") is not None
        else ""
    )

    if not result.get("sufficient_data"):
        lines = [
            f"## 🧭 Causal Attribution — {result.get('city', city)}",
            "",
            f"⚠️ {result['message']}",
        ]
        if z_line:
            lines += ["", z_line]
        return "\n".join(lines)

    conf_icons = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴", "N/A": "⚪"}
    conf_icon = conf_icons.get(result["confidence"], "⚪")

    pct = round(result["causal_probability"] * 100, 1)
    bar_filled = int(pct / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    verdict = "🔥 CAUSAL" if result["is_causal"] else "❄️ NOT CAUSAL"

    rp = result.get("refutations_passed", 0)
    rt = result.get("total_refutations", 0)
    ref_line = ""
    if rt > 0:
        ref_icon = "🟢" if rp == rt else ("🟡" if rp >= rt // 2 else "🔴")
        ref_line = (
            f"**Causal robustness**: {ref_icon} {rp}/{rt} refutation tests passed"
        )

    lines = [
        f"## 🧭 Causal Attribution — {result['city']}",
        f"*{result['method']} | {result['timestamp']}*",
        "",
        f"**Wildfire causal probability**: {pct}% [{verdict}]",
        f"  [{bar}] {pct}%",
        "",
        f"**Average Treatment Effect (ATE)**: {result['ate']} µg/m³ per unit fire intensity",
        f"**Paired daily observations used**: {result['paired_observations']}",
        *([z_line] if z_line else []),
        "",
        f"**Confidence**: {conf_icon} {result['confidence']}",
        *([ref_line] if ref_line else []),
        "",
        f"**Verdict**: {result['message']}",
        "",
        "*Method: DoWhy causal inference, DAG with backdoor criterion.*",
        "*Variables: X = wildfire intensity (treatment), Y = PM2.5 (outcome), "
        "W = wind speed (confounder)*",
        "*Sources: OpenAQ (PM2.5), NASA FIRMS (fires), Open-Meteo (wind, archive)*",
    ]

    return "\n".join(lines)


# ── Tool 7: PM2.5 Forecast ────────────────────────────────────────────────────


@mcp.tool()
async def get_forecast(city: str) -> str:
    """
    Predict tomorrow's PM2.5 air quality for any city using a trained LSTM model.

    The LSTM was trained on 1 year of PM2.5 + weather data (Open-Meteo CAMS)
    across Delhi, Los Angeles, Seoul, and Krakow, then evaluated against
    persistence and rolling-average baselines.

    Requires the model to be trained first:
        python3 scripts/train_forecast.py

    Args:
        city: City name, e.g. 'Delhi', 'Seoul', 'London'
    """
    from forecast import predict_next_day

    try:
        result = await predict_next_day(city)
    except FileNotFoundError as e:
        return f"❌ Model not trained yet.\n{e}"
    except ValueError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ Forecast error: {e}"

    pm25  = result["predicted_pm25"]
    risk  = result["risk_level"]
    name  = result["city"]
    model = result["model"]
    seq   = result["seq_len"]

    risk_icons = {
        "Good":                              "✅",
        "Moderate":                          "🟡",
        "Unhealthy for Sensitive Groups":    "🟠",
        "Unhealthy":                         "🔴",
        "Very Unhealthy":                    "🟣",
        "Hazardous":                         "🚨",
    }
    icon = risk_icons.get(risk, "⚪")

    lines = [
        f"## 🔮 PM2.5 Forecast — {name}",
        f"*{model} | Based on last {seq} days | Open-Meteo CAMS data*",
        "",
        f"**Tomorrow's predicted PM2.5**: {pm25} µg/m³",
        f"**Risk level**: {icon} {risk}",
        "",
        "*Source: LSTM model trained on Open-Meteo CAMS historical data.*",
        "*For fine-tuned TimesFM forecasting, see Phase 6B (LoRA fine-tuning).*",
    ]
    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🌍 EcoSentinel MCP Server starting...")
    print(
        "   Tools: get_air_quality, get_wildfires, get_weather_risk, "
        "get_crisis_summary, detect_anomaly_tool, get_causal_attribution, get_forecast"
    )
    mcp.run(transport="stdio")
