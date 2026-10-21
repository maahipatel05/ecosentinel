"""
EcoSentinel Gemini Agent (Google ADK)
=====================================
A Gemini agent that uses EcoSentinel's existing MCP server (src/server.py)
as its toolbox. ADK launches the server over stdio, discovers all its tools,
and lets Gemini call them. These are the same tools Claude Desktop uses, unchanged.

Run from the project root:
    adk web agents
"""

import os
import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters

# agents/ecosentinel_agent/agent.py -> project root is three .parent hops up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SERVER_PATH = PROJECT_ROOT / "src" / "server.py"

root_agent = LlmAgent(
    model=Gemini(
        # Overridable in .env, e.g. GEMINI_MODEL=gemini-flash-lite-latest when
        # the default model returns 503 "experiencing high demand"
        model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        # Retry transient Gemini errors (503 overloaded, 429 rate limit, ...)
        # with exponential backoff (~1s, 2s, 4s), like causal.py's FIRMS calls.
        retry_options=types.HttpRetryOptions(initial_delay=1, attempts=4),
    ),
    name="ecosentinel_agent",
    description=(
        "Environmental risk assistant with live air quality, wildfire, weather, "
        "anomaly, causal and forecast tools."
    ),
    instruction=(
        "You are EcoSentinel, an environmental crisis monitoring assistant. "
        "Always use your tools for current conditions; never guess numbers. "
        "For a general 'is it safe' question, start with get_crisis_summary. "
        "If PM2.5 looks high, use detect_anomaly_tool to check if it is unusual, "
        "and get_causal_attribution to check whether nearby wildfires are the cause. "
        "Only say wildfires are causing or contributing to pollution if "
        "get_causal_attribution returns a causal probability supporting it. "
        "If it reports insufficient data, say fire influence is unverified. "
        "Use get_forecast for questions about the coming days, and "
        "get_pm25_trends for history and whether air is getting better or worse. "
        "Cite the data source and timestamp, and give actionable safety advice "
        "when risk is high."
    ),
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    # The venv's own Python, so the server gets DoWhy, PyTorch, etc.
                    command=sys.executable,
                    args=[str(SERVER_PATH)],
                    cwd=str(PROJECT_ROOT),
                ),
                # For stdio, ADK uses this as the limit for startup AND every
                # tool call; causal attribution can take minutes on slow FIRMS.
                timeout=300,
            ),
        )
    ],
)
