# 🌍 EcoSentinel

**An AI-powered environmental crisis monitor built on the Model Context Protocol (MCP)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-1.0-green)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

EcoSentinel gives AI assistants like Claude real-time access to environmental
crisis data — air quality, active wildfires, and flood risk — for any location
on Earth. Instead of relying on stale training data, Claude can reach out
through EcoSentinel and answer with numbers pulled live from NASA, OpenAQ, and
Open-Meteo.

Ask your AI:
- "Is the air safe to breathe in Delhi today?"
- "Are there wildfires near Sydney right now?"
- "What's the flood risk in Mumbai this week?"
- "Give me a full environmental crisis briefing for Jakarta."

---

## Table of Contents

- [Why EcoSentinel](#why-ecosentinel)
- [What Is MCP, in Plain English](#what-is-mcp-in-plain-english)
- [How It Works](#how-it-works)
- [Tools](#tools)
- [Getting Started](#getting-started)
- [Example Output](#example-output)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Why EcoSentinel

Environmental data exists, but it's scattered across dozens of agencies, each
with its own API, format, and quirks. Most people can't easily ask "is it safe
to go outside today?" and get a fast, synthesized answer grounded in real
numbers.

EcoSentinel solves that by acting as a bridge: it gives an AI assistant a
standardized way to query live air quality sensors, satellite wildfire
detection, and weather forecasts — and to combine all three into one coherent
crisis briefing for any place on the planet.

---

## What Is MCP, in Plain English

**MCP (Model Context Protocol)** is a standard, created by Anthropic, that
defines how AI assistants talk to external tools and data sources.

Think of it like this: Claude is a brilliant expert sitting in a room with no
internet connection. MCP is a slot in the door. You slide a question through;
a program on the other side (in this case, `server.py`) fetches the real
answer and slides it back. That program is what turns "Claude can only repeat
what it learned during training" into "Claude can tell you the actual AQI in
Mumbai right now."

EcoSentinel is a custom MCP **server** — it exposes a set of *tools* that
Claude can call whenever a user's question calls for live environmental data.

---

## How It Works

```
        ┌──────────┐        MCP        ┌─────────────────────┐
        │  Claude  │ ◄────────────────► │ EcoSentinel Server  │
        │ (the AI) │     (tool calls)   │     (server.py)     │
        └──────────┘                    └──────────┬──────────┘
                                                    │
                       ┌────────────────────────────┼────────────────────────────┐
                       ▼                            ▼                            ▼
               ┌───────────────┐          ┌──────────────────┐          ┌────────────────┐
               │  OpenAQ API   │          │  NASA FIRMS API  │          │ Open-Meteo API │
               │ Air quality   │          │ Wildfire hotspots│          │ Weather/flood  │
               └───────────────┘          └──────────────────┘          └────────────────┘
```

The flow for a typical question looks like this:

1. **User asks Claude** something like "what's the air quality in Lagos?"
2. **Claude inspects** the tools EcoSentinel exposes and decides which one(s)
   to call.
3. **The MCP server runs** the corresponding Python function in `server.py`.
4. **That function geocodes** the city name into latitude/longitude (every
   external API needs coordinates, not place names) and calls the relevant
   live API — often several at once using `asyncio.gather()` for speed.
5. **The data is processed and formatted** into a clean, structured report —
   AQI calculated from raw pollutant readings, flood risk derived from
   rainfall thresholds, fire hotspots filtered by distance, and so on.
6. **Claude receives the structured result** and writes a natural-language
   answer grounded in that real data — always citing its source and timestamp.

---

## Tools

EcoSentinel currently exposes four live tools to Claude:

| Tool | What it does | Source |
|---|---|---|
| `get_air_quality(city)` | Returns AQI, PM2.5, PM10, NO2, CO, O3, and SO2 levels with a health risk assessment, averaged across the nearest sensors for reliability | OpenAQ |
| `get_wildfires(city, radius_km, days)` | Lists active satellite-detected fire hotspots within a configurable radius and time window | NASA FIRMS (VIIRS SNPP) |
| `get_weather_risk(city)` | Reports current conditions plus a 3-day flood and storm risk assessment based on rainfall and wind-gust thresholds | Open-Meteo |
| `get_crisis_summary(city)` | Runs all of the above **concurrently** and synthesizes them into one full environmental briefing | All three, combined |

> **🔬 Coming soon — anomaly detection**
> A fifth tool, `detect_anomaly(city)`, is in active development
> (`src/anamoly.py`). It applies a **z-score statistical analysis** to 30 days
> of historical PM2.5 readings to answer a question raw numbers can't: *is
> today's air quality actually unusual for this city?* An AQI of 120 might be
> an ordinary Tuesday in Delhi but a major emergency in Sydney — the anomaly
> engine adds that missing context, flagging statistically significant spikes
> (`z ≥ 2.0`) with a severity rating from `NORMAL` to `CRITICAL`. It will soon
> be wired into `server.py` and folded into `get_crisis_summary`.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/maahipatel05/ecosentinel.git
cd ecosentinel
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Then edit `.env`:
- Get a **free** NASA FIRMS key at https://firms.modaps.eosdis.nasa.gov/api/area/
  and add it as `NASA_FIRMS_API_KEY`.
- Air quality (`OpenAQ`) and weather (`Open-Meteo`) tools work out of the box
  with **no API key required**.

### 5. Connect to Claude Desktop

Add EcoSentinel to your Claude Desktop config, found at:

```
~/Library/Application Support/Claude/claude_desktop_config.json
```

```json
{
  "mcpServers": {
    "ecosentinel": {
      "command": "python",
      "args": ["/absolute/path/to/ecosentinel/src/server.py"]
    }
  }
}
```

Restart Claude Desktop, and EcoSentinel will appear in the tools panel — ready
to answer questions about any location on Earth.

---

## Example Output

Asking Claude for a crisis summary triggers `get_crisis_summary`, which fans
out to all three data sources simultaneously and returns something like:

```
# 🌍 EcoSentinel Crisis Briefing — Jakarta
*Generated: 2026-06-08 14:32 UTC | Powered by EcoSentinel MCP*

---
[Air Quality report — AQI, PM2.5, PM10, NO2, health guidance]
---
[Wildfire report — active hotspots, distance, confidence]
---
[Weather & flood risk report — rainfall, wind, 3-day outlook]
---

## 📋 Summary
Data sources: OpenAQ · NASA FIRMS · Open-Meteo
For emergencies, always contact local authorities.
```

Running everything concurrently with `asyncio.gather()` means a full briefing
that would take roughly 9 seconds sequentially (3 APIs × 3 seconds each) comes
back in about 3 — a 3x speedup, for free.

---

## Data Sources

- **[OpenAQ](https://openaq.org)** — Real-time air quality from a global
  network of ground-level sensors (PM2.5, PM10, NO2, CO, O3, SO2).
- **[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov)** — Near real-time
  satellite wildfire detection via the VIIRS SNPP sensor.
- **[Open-Meteo](https://open-meteo.com)** — Free weather forecasts,
  precipitation, and wind data, no API key required.

---

## Project Structure

```
ecosentinel/
├── src/
│   ├── server.py        # MCP server — defines and exposes all tools
│   └── anamoly.py       # Z-score anomaly detection engine (in progress)
├── tests/
│   └── test_tools.py    # Async unit tests for each tool
├── docs/                # Additional documentation
├── .env.example         # Template for required environment variables
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT license
└── README.md
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

The test suite covers geocoding, air quality, wildfire, and weather-risk
tools against both valid cities and invalid/unknown locations.

---

## License

MIT. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **Anthropic** — for the Model Context Protocol
- **OpenAQ** — for open, global air quality data
- **NASA FIRMS** — for satellite-based wildfire detection
- **Open-Meteo** — for free, no-key-required weather data

---

*Built by [Maahi Patel](https://github.com/maahipatel05) — Rice University.*
*For emergencies, always contact local authorities. EcoSentinel is a tool for awareness and research, not a substitute for official alerts.*
