"""
EcoSentinel — Basic tool tests
Run with: python -m pytest tests/ -v
"""

import asyncio
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from server import get_air_quality, get_weather_risk, get_wildfires


# ── Geocoding ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_air_quality_known_city():
    result = await get_air_quality("London")
    assert "Air Quality" in result
    assert "London" in result


@pytest.mark.asyncio
async def test_air_quality_invalid_city():
    result = await get_air_quality("xyznotarealplace99999")
    assert "❌" in result or "not found" in result.lower()


@pytest.mark.asyncio
async def test_weather_risk_returns_data():
    result = await get_weather_risk("Mumbai")
    assert "Weather" in result
    assert "Flood risk" in result or "flood" in result.lower()


@pytest.mark.asyncio
async def test_weather_risk_temperature_present():
    result = await get_weather_risk("Tokyo")
    assert "Temperature" in result or "°C" in result


@pytest.mark.asyncio
async def test_wildfires_returns_result():
    result = await get_wildfires("Sydney", radius_km=300, days=1)
    # Either finds fires or reports none — both valid
    assert "Wildfire" in result or "wildfire" in result.lower()


@pytest.mark.asyncio
async def test_wildfires_invalid_city():
    result = await get_wildfires("zzznowhere999")
    assert "❌" in result or "not found" in result.lower()
