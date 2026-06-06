"""
EcoSentinel MCP Server
======================
An MCP (Model Context Protocol) server that gives AI assistants real-time
access to environmental crisis data — air quality, wildfires, and flood risk.

Tools exposed:
  - get_air_quality      : Live AQI + pollutant data for any city
  - get_wildfires        : Active wildfire hotspots near a location
  - get_weather_risk     : Current weather + flood/storm risk
  - get_crisis_summary   : AI-synthesized environmental briefing for any location
"""

import os
import httpx
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ── MCP server instance ────────────────────────────────────────────────────────
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
OPENAQ_BASE    = "https://api.openaq.org/v3"
FIRMS_BASE     = "https://firms.modaps.eosdis.nasa.gov/api/area"
METEO_BASE     = "https://api.open-meteo.com/v1"
GEOCODE_BASE   = "https://geocoding-api.open-meteo.com/v1"


# ── Helpers ────────────────────────────────────────────────────────────────────

async def geocode(city: str) -> tuple[float, float, str]:
    """Return (lat, lon, display_name) for a city string."""
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
        return hit["latitude"], hit["longitude"], name


def aqi_label(aqi: float) -> str:
    """Convert numeric AQI to WHO category label."""
    if aqi <= 50:   return "✅ Good"
    if aqi <= 100:  return "🟡 Moderate"
    if aqi <= 150:  return "🟠 Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "🔴 Unhealthy"
    if aqi <= 300:  return "🟣 Very Unhealthy"
    return "🟤 Hazardous"


def wind_label(speed_kmh: float) -> str:
    if speed_kmh < 20:  return "Calm"
    if speed_kmh < 40:  return "Breezy"
    if speed_kmh < 60:  return "Windy"
    if speed_kmh < 90:  return "Strong winds"
    return "⚠️ Storm-force winds"


# ── Tool 1: Air Quality ────────────────────────────────────────────────────────

@mcp.tool()
async def get_air_quality(city: str) -> str:
    """
    Get real-time air quality data for any city in the world.

    Returns AQI, PM2.5, PM10, NO2, CO levels and a health risk assessment.
    Data sourced from OpenAQ (global sensor network).

    Args:
        city: City name, e.g. 'Delhi', 'London', 'Los Angeles'
    """
    try:
        lat, lon, display_name = await geocode(city)
    except ValueError as e:
        return f"❌ {e}"

    async with httpx.AsyncClient(timeout=15) as client:
        # Fetch nearest stations within 50 km
        r = await client.get(
            f"{OPENAQ_BASE}/locations",
            params={
                "coordinates": f"{lat},{lon}",
                "radius": 50000,
                "limit": 5,
                "order_by": "distance",
            },
            headers={"X-API-Key": os.getenv("OPENAQ_API_KEY", "")},
        )

        if r.status_code != 200:
            return f"❌ OpenAQ API error: {r.status_code}. Try a larger city nearby."

        data = r.json()
        locations = data.get("results", [])

        if not locations:
            return (
                f"No air quality sensors found within 50 km of **{display_name}**.\n"
                "Try a larger nearby city."
            )

        # Collect all latest measurements across nearby stations
        pollutants: dict[str, list[float]] = {}
        station_names = []

        for loc in locations[:3]:
            station_names.append(loc.get("name", "Unknown station"))
            for sensor in loc.get("sensors", []):
                param = sensor.get("parameter", {})
                name  = param.get("name", "").lower()
                value = sensor.get("latest", {}).get("value")
                unit  = param.get("units", "")
                if value is not None and name in ("pm25", "pm10", "no2", "co", "o3", "so2"):
                    pollutants.setdefault(name, []).append((value, unit))

        if not pollutants:
            return f"Sensors found near **{display_name}** but no recent measurements available."

        # Average across stations
        averaged = {
            k: (sum(v for v, _ in vals) / len(vals), vals[0][1])
            for k, vals in pollutants.items()
        }

        # Estimate AQI from PM2.5 if available (EPA standard approximation)
        pm25_val = averaged.get("pm25", (None,))[0]
        aqi_str = ""
        if pm25_val is not None:
            # Simple linear AQI approximation from PM2.5 µg/m³
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

        # Format output
        lines = [
            f"## 🌫️ Air Quality — {display_name}",
            f"*Data from OpenAQ | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            aqi_str,
            "\n**Pollutant Levels:**",
        ]

        param_labels = {
            "pm25": "PM2.5 (fine particles)",
            "pm10": "PM10 (coarse particles)",
            "no2":  "NO₂ (nitrogen dioxide)",
            "co":   "CO (carbon monoxide)",
            "o3":   "O₃ (ozone)",
            "so2":  "SO₂ (sulphur dioxide)",
        }

        for param, (val, unit) in averaged.items():
            label = param_labels.get(param, param.upper())
            lines.append(f"  • {label}: **{val:.1f} {unit}**")

        lines += [
            f"\n**Stations sampled**: {', '.join(station_names[:3])}",
            f"**Coordinates**: {lat:.4f}°N, {lon:.4f}°E",
            "\n*Source: OpenAQ global sensor network — openaq.org*",
        ]

        return "\n".join(lines)


# ── Tool 2: Wildfires ──────────────────────────────────────────────────────────

@mcp.tool()
async def get_wildfires(
    city: str,
    radius_km: int = 500,
    days: int = 2,
) -> str:
    """
    Get active wildfire hotspots detected by NASA satellites near a location.

    Uses NASA FIRMS (Fire Information for Resource Management System) —
    the same data used by emergency responders worldwide.

    Args:
        city:      City or region to check, e.g. 'Sydney', 'California'
        radius_km: Search radius in km (default 500)
        days:      How many days back to check (1–10, default 2)
    """
    try:
        lat, lon, display_name = await geocode(city)
    except ValueError as e:
        return f"❌ {e}"

    days = max(1, min(days, 10))

    # NASA FIRMS provides a demo key for testing; real key recommended for production
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
            "⚠️ NASA FIRMS API requires a free API key for production use.\n"
            "Get yours free at: https://firms.modaps.eosdis.nasa.gov/api/area/\n"
            "Then add it to your .env file as NASA_FIRMS_API_KEY=your_key"
        )

    if r.status_code != 200:
        return f"❌ NASA FIRMS API error: {r.status_code}"

    lines_raw = r.text.strip().split("\n")

    if len(lines_raw) <= 1:
        return (
            f"## 🔥 Wildfires near {display_name}\n"
            f"*NASA FIRMS | Last {days} day(s) | {radius_km} km radius*\n\n"
            f"✅ **No active fire hotspots detected** within {radius_km} km.\n"
            "*Source: NASA VIIRS SNPP satellite — firms.modaps.eosdis.nasa.gov*"
        )

    # Parse CSV (skip header)
    hotspots = []
    header = lines_raw[0].split(",")
    for row in lines_raw[1:]:
        cols = row.split(",")
        if len(cols) < 6:
            continue
        try:
            hotspots.append({
                "lat":         float(cols[0]),
                "lon":         float(cols[1]),
                "brightness":  float(cols[2]),
                "confidence":  cols[8].strip() if len(cols) > 8 else "n/a",
                "date":        cols[5].strip() if len(cols) > 5 else "n/a",
            })
        except (ValueError, IndexError):
            continue

    high_conf = [h for h in hotspots if h["confidence"] in ("high", "h", "100", "nominal")]

    output = [
        f"## 🔥 Wildfire Alert — {display_name}",
        f"*NASA FIRMS VIIRS | Last {days} day(s) | {radius_km} km radius | "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        f"**Total hotspots detected**: {len(hotspots)}",
        f"**High-confidence fires**: {len(high_conf)}",
    ]

    if len(hotspots) > 0:
        risk = "🔴 HIGH" if len(hotspots) > 20 else ("🟡 MODERATE" if len(hotspots) > 5 else "🟢 LOW")
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
    return "\n".join(output)


# ── Tool 3: Weather & Flood Risk ───────────────────────────────────────────────

@mcp.tool()
async def get_weather_risk(city: str) -> str:
    """
    Get current weather conditions and assess flood/storm risk for any city.

    Uses Open-Meteo (free, no API key needed). Returns temperature, rainfall,
    wind speed, and a risk assessment for flooding and severe weather.

    Args:
        city: City name, e.g. 'Mumbai', 'Houston', 'Bangladesh'
    """
    try:
        lat, lon, display_name = await geocode(city)
    except ValueError as e:
        return f"❌ {e}"

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            f"{METEO_BASE}/forecast",
            params={
                "latitude":         lat,
                "longitude":        lon,
                "current":          "temperature_2m,precipitation,rain,wind_speed_10m,wind_gusts_10m,weathercode",
                "daily":            "precipitation_sum,rain_sum,wind_speed_10m_max,weathercode",
                "forecast_days":    3,
                "timezone":         "auto",
            },
        )
        r.raise_for_status()

    d = r.json()
    cur = d.get("current", {})
    daily = d.get("daily", {})

    temp        = cur.get("temperature_2m", "N/A")
    precip      = cur.get("precipitation", 0)
    rain        = cur.get("rain", 0)
    wind        = cur.get("wind_speed_10m", 0)
    gusts       = cur.get("wind_gusts_10m", 0)
    wcode       = cur.get("weathercode", 0)

    # 3-day totals
    daily_precip = daily.get("precipitation_sum", [0, 0, 0])
    total_3day   = sum(p for p in daily_precip if p) if daily_precip else 0

    # Risk assessment
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

    # WMO weather code to description
    wcode_desc = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        51: "Light drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 80: "Rain showers", 95: "Thunderstorm", 99: "Thunderstorm with hail",
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
    for i, (day, p) in enumerate(zip(days_label[:3], daily_precip[:3])):
        bar = "█" * min(int((p or 0) / 5), 20)
        lines.append(f"  {day}: {p or 0:.1f} mm  {bar}")

    lines += [
        "",
        f"**Flood risk assessment**: {flood_risk}",
        f"**Storm risk assessment**: {storm_risk}",
        f"**Total 3-day rainfall**: {total_3day:.1f} mm",
        "",
        "*Source: Open-Meteo (open-meteo.com) + Open-Meteo Geocoding API*",
    ]

    return "\n".join(lines)


# ── Tool 4: Crisis Summary ─────────────────────────────────────────────────────

@mcp.tool()
async def get_crisis_summary(city: str) -> str:
    """
    Generate a comprehensive environmental crisis briefing for any location.

    Combines air quality, wildfire, and weather/flood risk data into a single
    synthesized report. This is the flagship EcoSentinel tool — use it when
    you need a full environmental picture of a location.

    Args:
        city: City or region name, e.g. 'Jakarta', 'Cape Town', 'Amazon'
    """
    # Run all three tools in parallel for speed
    aq_task      = get_air_quality(city)
    fire_task    = get_wildfires(city)
    weather_task = get_weather_risk(city)

    air_result, fire_result, weather_result = await asyncio.gather(
        aq_task, fire_task, weather_task
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    summary = f"""# 🌍 EcoSentinel Crisis Briefing — {city.title()}
*Generated: {timestamp} | Powered by EcoSentinel MCP*

---

{air_result}

---

{fire_result}

---

{weather_result}

---

## 📋 Summary & Recommendations

This briefing combines data from three independent sources:
- **OpenAQ** — global ground-level air quality sensors
- **NASA FIRMS** — satellite-detected wildfire hotspots (VIIRS SNPP)
- **Open-Meteo** — weather forecast and precipitation data

*For emergencies, always contact local authorities. This tool is for awareness and research.*
*EcoSentinel — github.com/maahipatel05/ecosentinel*
"""
    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🌍 EcoSentinel MCP Server starting...")
    print("   Tools: get_air_quality, get_wildfires, get_weather_risk, get_crisis_summary")
    mcp.run(transport="stdio")
