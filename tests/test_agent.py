"""
Unit tests for agents/ecosentinel_agent/agent.py (Google ADK Gemini agent)

All offline: no Gemini API calls and no environmental API calls.
  - root_agent loads with the expected model and retry settings
  - the causal-claim guardrail is present in the instruction
  - ADK can launch src/server.py over stdio and discover all 7 MCP tools
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
from ecosentinel_agent.agent import root_agent

EXPECTED_TOOLS = {
    "get_air_quality",
    "get_wildfires",
    "get_weather_risk",
    "get_crisis_summary",
    "detect_anomaly_tool",
    "get_causal_attribution",
    "get_forecast",
    "get_pm25_trends",
}

# ── Agent configuration ────────────────────────────────────────────────────────


def test_agent_uses_gemini_with_retries():
    assert root_agent.name == "ecosentinel_agent"
    assert root_agent.model.model == "gemini-flash-latest"
    assert root_agent.model.retry_options.attempts == 4


def test_instruction_has_causal_guardrail():
    # Gemini must not claim wildfire causation unless the causal tool supports it
    assert "get_causal_attribution" in root_agent.instruction
    assert "unverified" in root_agent.instruction


# ── MCP tool discovery ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_discovers_all_mcp_tools():
    # Launches src/server.py as a subprocess and asks it for its tool list
    toolset = root_agent.tools[0]
    try:
        tools = await toolset.get_tools()
    finally:
        await toolset.close()

    assert {t.name for t in tools} == EXPECTED_TOOLS
