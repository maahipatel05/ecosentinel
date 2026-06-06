# 🌍 EcoSentinel

**An AI-powered environmental crisis monitor built on the Model Context Protocol (MCP)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-1.0-green)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

EcoSentinel gives AI assistants like Claude real-time access to environmental crisis data — air quality, active wildfires, and flood risk — for any location on Earth.

Ask your AI:
- Is the air safe to breathe in Delhi today?
- Are there wildfires near Sydney right now?
- What is the flood risk in Mumbai this week?
- Give me a full environmental briefing for Jakarta.

---

## Architecture

Claude (AI) via MCP connects to EcoSentinel Server, which queries:
- OpenAQ API for air quality
- NASA FIRMS for wildfire hotspots
- Open-Meteo for weather and flood risk

---

## Tools

- get_air_quality: AQI, PM2.5, PM10, NO2, CO for any city (OpenAQ)
- get_wildfires: Active satellite-detected fire hotspots (NASA FIRMS VIIRS)
- get_weather_risk: Weather and flood/storm risk assessment (Open-Meteo)
- get_crisis_summary: Full environmental briefing combining all sources

---

## Getting Started

### 1. Clone the repository

git clone https://github.com/maahipatel05/ecosentinel.git
cd ecosentinel

### 2. Create virtual environment

python -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Configure API keys

cp .env.example .env

Edit .env and add your free NASA FIRMS key from https://firms.modaps.eosdis.nasa.gov/api/area/

Air quality and weather tools work with no API key at all.

### 5. Connect to Claude Desktop

Add this to your Claude Desktop config at:
~/Library/Application Support/Claude/claude_desktop_config.json

{
  "mcpServers": {
    "ecosentinel": {
      "command": "python",
      "args": ["/absolute/path/to/ecosentinel/src/server.py"]
    }
  }
}

Restart Claude Desktop and EcoSentinel will appear in the tools panel.

---

## Data Sources

- OpenAQ (openaq.org): Real-time air quality from global ground sensors
- NASA FIRMS (firms.modaps.eosdis.nasa.gov): Satellite wildfire detection VIIRS SNPP
- Open-Meteo (open-meteo.com): Weather forecasts, precipitation, wind

---

## Project Structure

ecosentinel/
  src/
    server.py         - MCP server with all four tools
  tests/
    test_tools.py     - Unit tests
  docs/               - Additional documentation
  .env.example        - Environment variable template
  requirements.txt    - Python dependencies
  README.md

---

## License

MIT. See LICENSE for details.

---

## Acknowledgements

- Anthropic for the Model Context Protocol
- OpenAQ for open air quality data
- NASA FIRMS for satellite fire data
- Open-Meteo for free weather data
