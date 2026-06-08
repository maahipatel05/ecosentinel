# 🌍 EcoSentinel

**A hybrid-intelligence environmental crisis monitor, built on the Model Context Protocol (MCP)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-1.0-green)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

EcoSentinel connects AI assistants like Claude to real-time environmental data
from three global sources (live air quality sensors, satellite wildfire
detection, and weather forecasting) through the Model Context Protocol. It
goes a step further than simple data-fetching: it **statistically detects**
when air quality is abnormal for a given city, and is being built out to
**mathematically attribute** *why*, separating "there's a wildfire nearby"
from "the wildfire is actually causing this spike."

Ask your AI:
- "Is the air safe to breathe in Delhi today?"
- "Are there wildfires near Sydney right now?"
- "Is Jakarta's air quality unusual for this time of year, or just normal smog?"
- "Give me a full environmental crisis briefing for Mumbai."

---

## Table of Contents

- [Why "Hybrid Intelligence"](#why-hybrid-intelligence)
- [Architecture](#architecture)
- [How a Request Flows Through the System](#how-a-request-flows-through-the-system)
- [What Is MCP, in Plain English](#what-is-mcp-in-plain-english)
- [Tools](#tools)
- [Statistical Anomaly Detection](#statistical-anomaly-detection)
- [Causal Inference Engine](#causal-inference-engine)
- [Getting Started](#getting-started)
- [Example Output](#example-output)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Why "Hybrid Intelligence"

Most AI projects let the language model do *all* of the reasoning, including
the math. That's risky, because LLMs hallucinate numbers.

EcoSentinel enforces a strict separation between two kinds of intelligence
working together:

- **Language model intelligence**: Claude handles natural-language
  understanding, decides which tools to call and when, and synthesizes the
  results into a clear answer for the user.
- **Statistical / mathematical intelligence**: Python libraries (`scipy`,
  `numpy`, and eventually `DoWhy`) handle the actual numerical computation,
  with zero hallucination risk.

Claude decides **what** to analyze; Python computes **how**. When the system
flags a pollution spike near an active wildfire, it doesn't let Claude *guess*
whether the fire is the cause. It runs real statistics and causal inference
in Python and hands Claude a precise, defensible number.

---

## Architecture

EcoSentinel is organized into six cooperating layers:

| Layer | Name | Responsibility | Technology |
|---|---|---|---|
| 1 | **Data Collection** | Fetch live data from three APIs concurrently | `httpx`, `asyncio.gather`, OpenAQ, NASA FIRMS, Open-Meteo |
| 2 | **Statistical Engine** | Z-score anomaly detection on 30-day historical baselines | Python, `scipy`, `numpy` |
| 3 | **Causal Inference** | Compute `P(Y \| do(X))`, attributing pollution cause via a DAG | `DoWhy`, `pandas` |
| 4 | **Agentic Router** | Conditionally trigger deeper analysis only when an anomaly is confirmed | async Python in `server.py` |
| 5 | **MCP Interface** | Expose every tool to Claude via the Model Context Protocol | `FastMCP`, `mcp` |
| 6 | **Visual Dashboard** | Live geospatial maps, wind vectors, time-series charts | `Streamlit`, `plotly`, `folium` |

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
               │ Air quality   │          │ Wildfire hotspots│          │ Weather/wind   │
               └───────────────┘          └──────────────────┘          └────────────────┘
                       │                            │                            │
                       └──────────────┬─────────────┴──────────────┬─────────────┘
                                       ▼                            ▼
                          ┌────────────────────────┐   ┌─────────────────────────┐
                          │ Z-Score Anomaly Engine │──▶│ Causal Inference Engine │
                          │   (scipy / numpy)      │   │  (DoWhy DAG attribution) │
                          └────────────────────────┘   └─────────────────────────┘
```

---

## How a Request Flows Through the System

A typical question moves through the layers like this:

1. A user asks Claude something like *"What's the air quality in Lagos, and is it normal?"*
2. Claude inspects the tools EcoSentinel exposes and decides which to call.
3. The **agentic router** kicks off concurrent calls to all relevant data sources.
4. `asyncio.gather()` fetches air quality, wildfire, and weather data **simultaneously**, roughly a 3x speedup over sequential calls.
5. The **z-score engine** compares today's PM2.5 reading to the city's 30-day historical baseline.
6. If the z-score crosses the anomaly threshold (`z ≥ 2.0`), the router automatically triggers the causal inference engine.
7. The **causal engine** constructs a DAG and computes `P(Y | do(X))` via the backdoor criterion to estimate how much of the spike is attributable to a nearby wildfire, after controlling for wind.
8. If there isn't enough historical data for a confident causal estimate, the system **gracefully degrades**, falling back to the z-score result alone rather than guessing.
9. Every result is returned to Claude as a structured report.
10. Claude synthesizes everything into a clear, natural-language answer that's grounded entirely in real numbers, not invented ones.

This conditional, decision-driven behavior (fetch, evaluate, and only escalate
when the evidence warrants it) is what makes the router **agentic** rather
than a simple fixed pipeline.

---

## What Is MCP, in Plain English

**MCP (Model Context Protocol)** is an open standard, created by Anthropic,
that defines how AI assistants talk to external tools and data sources.

Think of it like this: Claude is a brilliant expert sitting in a room with no
internet connection. MCP is a slot in the door. You slide a question through;
a program on the other side (`server.py`) fetches the real answer, runs the
math, and slides the result back. That's the difference between "Claude can
only repeat what it learned during training" and "Claude can tell you the
actual AQI in Mumbai right now, and whether that's unusual."

EcoSentinel is a custom MCP **server**: it exposes a set of *tools* that Claude
can call whenever a user's question calls for live environmental data or
statistical analysis.

---

## Tools

| Tool | Parameters | Source | Status |
|---|---|---|---|
| `get_air_quality` | `city` | OpenAQ v3 sensor measurements API | ✅ Live |
| `get_wildfires` | `city`, `radius_km`, `days` | NASA FIRMS (VIIRS SNPP) | ✅ Live |
| `get_weather_risk` | `city` | Open-Meteo forecast API | ✅ Live |
| `get_crisis_summary` | `city` | All three sources, fetched concurrently | ✅ Live |
| `detect_anomaly` | `city` | OpenAQ historical API + `scipy` z-score analysis | 🔬 In development |
| `get_causal_attribution` | `city` | All sources + `DoWhy` causal engine | 🧭 Planned |

> Every live tool returns a clean, structured report: AQI calculated from raw
> pollutant readings, fire hotspots filtered by distance and confidence, and
> flood/storm risk derived from rainfall and wind-gust thresholds, so Claude
> always answers with real numbers, source citations, and timestamps.

---

## Statistical Anomaly Detection

Raw numbers lack context. An AQI of 120 might be an ordinary Tuesday in Delhi
but a major emergency in Sydney. EcoSentinel adds that missing context with a
**z-score** computed against each city's own 30-day history:

```
z = (current_value - mean) / standard_deviation

current_value → today's PM2.5 reading from OpenAQ
mean          → average PM2.5 over the last 30 days for this city
std deviation → how much PM2.5 typically varies day to day
```

| Z-score | Meaning |
|---|---|
| `z < 1.5` | Normal, no further analysis triggered |
| `1.5 ≤ z < 2.0` | Elevated, alert issued |
| `z ≥ 2.0` | **Anomaly confirmed**, the agentic router escalates to the causal engine |

The engine fetches 30 days of historical PM2.5 averages, computes the mean and
standard deviation with `numpy`/`scipy`, derives the z-score, and returns a
structured result (`z_score`, `mean`, `std`, `severity`, `is_anomaly`, …) that
Claude can reason over directly.

---

## Causal Inference Engine

### Correlation isn't causation

Knowing that there's a wildfire *and* high PM2.5 near a city doesn't prove the
wildfire is the cause. Wind direction matters. If the wind is blowing away
from the city, the fire can't be responsible, no matter how close it is.
EcoSentinel uses **`DoWhy`** to mathematically isolate the wildfire's causal
contribution after controlling for wind: the actual difference between
correlation and causation.

### The causal graph (DAG)

```
  Wildfire proximity (X)  ──causes──▶  PM2.5 spike (Y)
  Wind vectors (W)        ──causes──▶  PM2.5 spike (Y)
  Wind vectors (W)        ──causes──▶  Wildfire smoke reaching the city
```

`DoWhy` reads this Directed Acyclic Graph and computes:

```
P(Y | do(X)) = the probability that PM2.5 is high BECAUSE of the wildfire,
               after mathematically removing the effect of wind (the confounder)
```

| Variable | Role | Represents | Source |
|---|---|---|---|
| `X` (Treatment) | Wildfire proximity | Distance & thermal intensity of the nearest hotspot | NASA FIRMS |
| `Y` (Outcome) | PM2.5 reading | Current fine-particulate concentration | OpenAQ |
| `W₁` (Confounder) | Wind vectors | Speed & direction carrying smoke toward/away from the city | Open-Meteo |
| `W₂` (Confounder) | Historical baseline | 30-day average PM2.5 for the city | OpenAQ |

### Observational vs. interventional probability

- **`P(Y\|X)`**: *Given that we observe a wildfire*, what is PM2.5? This
  includes all sorts of indirect, confounded effects.
- **`P(Y\|do(X))`**: *If we forced a wildfire to exist* and controlled for
  everything else, what would PM2.5 be? This isolates the **true causal
  effect**.

`DoWhy` identifies and adjusts for confounders using the **backdoor
criterion**, producing a single number (for example `0.73`, meaning *the
wildfire explains roughly 73% of this pollution spike*) that is mathematically
computed, not estimated by an LLM. If there isn't enough data for a confident
estimate, the system gracefully falls back to the z-score result alone.

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
  with **no API key required** (an OpenAQ key just raises your rate limits).

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

Restart Claude Desktop, and EcoSentinel will appear in the tools panel, ready
to answer questions about any location on Earth.

---

## Example Output

Asking Claude for a crisis summary triggers `get_crisis_summary`, which fans
out to all three data sources simultaneously and returns something like:

```
# 🌍 EcoSentinel Crisis Briefing: Jakarta
*Generated: 2026-06-08 14:32 UTC | Powered by EcoSentinel MCP*

---
[Air Quality report: AQI, PM2.5, PM10, NO2, health guidance]
---
[Wildfire report: active hotspots, distance, confidence]
---
[Weather & flood risk report: rainfall, wind, 3-day outlook]
---

## 📋 Summary
Data sources: OpenAQ · NASA FIRMS · Open-Meteo
For emergencies, always contact local authorities.
```

Running everything concurrently with `asyncio.gather()` means a full briefing
that would take roughly 9 seconds sequentially (3 APIs × 3 seconds each) comes
back in about 3, roughly a 3x speedup, for free.

---

## Data Sources

- **[OpenAQ](https://openaq.org)**: Real-time air quality from a global
  network of ground-level sensors (PM2.5, PM10, NO2, CO, O3, SO2).
- **[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov)**: Near real-time
  satellite wildfire detection via the VIIRS SNPP sensor.
- **[Open-Meteo](https://open-meteo.com)**: Free weather forecasts,
  precipitation, and wind data, no API key required.

---

## Project Structure

```
ecosentinel/
├── src/
│   ├── server.py        # MCP server: tool definitions and agentic router
│   ├── anamoly.py       # Z-score anomaly detection engine (scipy/numpy)
│   └── causal.py        # DoWhy causal inference engine: DAG + P(Y|do(X)) [planned]
├── tests/
│   ├── test_tools.py    # Async unit tests for each live tool
│   └── test_anomaly.py  # Unit tests verifying z-score math
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

The test suite covers geocoding, air quality, wildfire, weather-risk, and
anomaly-detection logic against both valid cities and invalid/unknown
locations.

---

## License

MIT. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **Anthropic**: for the Model Context Protocol
- **OpenAQ**: for open, global air quality data
- **NASA FIRMS**: for satellite-based wildfire detection
- **Open-Meteo**: for free, no-key-required weather data
- **Microsoft DoWhy**: for making rigorous causal inference accessible in Python

---

*Built by [Maahi Patel](https://github.com/maahipatel05), Rice University.*
*For emergencies, always contact local authorities. EcoSentinel is a tool for awareness and research, not a substitute for official alerts.*
