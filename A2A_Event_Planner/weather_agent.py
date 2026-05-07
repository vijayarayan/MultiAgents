"""
Weather Agent — A2A HTTP server that wraps the weather-skills MCP server.

Externally: exposes A2A endpoints for other agents / coordinators.
Internally: connects to weather-skills/weather_server.py via MCP (stdio).

Usage:
    uv run weather_agent.py /absolute/path/to/weather-skills/weather_server.py
"""

import json
import re
import sys
from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
from fastapi import FastAPI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from a2a_protocol import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    TaskSendRequest,
    TaskSendResponse,
    TaskState,
    TaskStatus,
    extract_data,
    make_agent_message,
)

# ---------------------------------------------------------------------------
# City coordinate lookup (avoids needing geocoding API)
# ---------------------------------------------------------------------------

CITY_COORDS: dict[str, tuple[float, float]] = {
    "austin": (30.2672, -97.7431),
    "houston": (29.7604, -95.3698),
    "dallas": (32.7767, -96.7970),
    "san antonio": (29.4241, -98.4936),
    "miami": (25.7617, -80.1918),
    "orlando": (28.5383, -81.3792),
    "tampa": (27.9506, -82.4572),
    "denver": (39.7392, -104.9903),
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "san francisco": (37.7749, -122.4194),
    "chicago": (41.8781, -87.6298),
    "seattle": (47.6062, -122.3321),
    "phoenix": (33.4484, -112.0740),
    "atlanta": (33.7490, -84.3880),
    "boston": (42.3601, -71.0589),
    "nashville": (36.1627, -86.7816),
    "portland": (45.5152, -122.6784),
}


# ---------------------------------------------------------------------------
# MCP Bridge — connects to the existing weather-skills server
# ---------------------------------------------------------------------------

class MCPBridge:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect(self, server_path: str):
        server_params = StdioServerParameters(
            command="python",
            args=[server_path],
            env=None,
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()

        tools_resp = await self.session.list_tools()
        print(f"  MCP connected — tools: {[t.name for t in tools_resp.tools]}")

    async def call_tool(self, name: str, args: dict) -> str:
        result = await self.session.call_tool(name, args)
        text = ""
        for item in result.content:
            if hasattr(item, "text"):
                text += item.text
        return text

    async def disconnect(self):
        await self.exit_stack.aclose()


mcp_bridge = MCPBridge()

# ---------------------------------------------------------------------------
# Weather analysis logic (no LLM needed — rule-based assessment)
# ---------------------------------------------------------------------------

def _parse_temperature(forecast_text: str) -> list[int]:
    """Extract temperatures from forecast text."""
    return [int(t) for t in re.findall(r"Temperature:\s*(\d+)", forecast_text)]


def _check_precipitation(forecast_text: str) -> bool:
    """Check if precipitation is mentioned."""
    keywords = ["rain", "storm", "thunder", "shower", "snow", "sleet", "hail"]
    lower = forecast_text.lower()
    return any(kw in lower for kw in keywords)


def _check_high_wind(forecast_text: str) -> bool:
    """Check for high winds (>25 mph)."""
    wind_speeds = re.findall(r"(\d+)\s*(?:to\s*\d+\s*)?mph", forecast_text.lower())
    return any(int(w) > 25 for w in wind_speeds)


def assess_weather(alerts_text: str, forecast_text: str, time_pref: str = "") -> dict:
    """Produce a structured weather assessment from raw MCP tool outputs."""
    concerns = []
    temps = _parse_temperature(forecast_text)

    # Temperature
    high_temp = max(temps) if temps else None
    low_temp = min(temps) if temps else None
    if high_temp and high_temp > 95:
        concerns.append(f"High temperature: {high_temp}F — heat risk")
    if low_temp and low_temp < 40:
        concerns.append(f"Low temperature: {low_temp}F — cold risk")

    # Precipitation
    if _check_precipitation(forecast_text):
        concerns.append("Precipitation in forecast — rain/storm risk")

    # Wind
    if _check_high_wind(forecast_text):
        concerns.append("High winds (>25 mph) — structural/comfort risk")

    # Active alerts
    has_alerts = "no active weather alerts" not in alerts_text.lower()
    if has_alerts:
        concerns.append("Active weather alerts — check details")

    # Recommendation
    if any("heat risk" in c or "storm risk" in c for c in concerns) or has_alerts:
        recommendation = "CAUTION"
    elif len(concerns) >= 2:
        recommendation = "CAUTION"
    elif not concerns:
        recommendation = "GO"
    else:
        recommendation = "GO"

    # If time preference is evening and temp concern is afternoon heat, soften
    if time_pref == "evening" and high_temp and high_temp > 95:
        # Check if evening/night periods are cooler
        evening_ok = "tonight" in forecast_text.lower() or "night" in forecast_text.lower()
        if evening_ok:
            concerns = [c for c in concerns if "High temperature" not in c]
            concerns.append(f"Afternoon high of {high_temp}F avoided — evening temps expected lower")
            if len(concerns) <= 1 and not has_alerts:
                recommendation = "GO"

    return {
        "recommendation": recommendation,
        "concerns": concerns,
        "high_temp": high_temp,
        "low_temp": low_temp,
        "has_precipitation": _check_precipitation(forecast_text),
        "has_high_wind": _check_high_wind(forecast_text),
        "has_active_alerts": has_alerts,
        "time_preference": time_pref or "any",
        "raw_alerts": alerts_text[:500],
        "raw_forecast": forecast_text[:500],
    }


# ---------------------------------------------------------------------------
# FastAPI app with A2A endpoints
# ---------------------------------------------------------------------------

SERVER_SCRIPT_PATH = sys.argv[1] if len(sys.argv) > 1 else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not SERVER_SCRIPT_PATH:
        print("ERROR: Usage: python weather_agent.py <path_to_weather_server.py>")
        sys.exit(1)
    print("Weather Agent starting...")
    print(f"  Connecting to MCP server: {SERVER_SCRIPT_PATH}")
    await mcp_bridge.connect(SERVER_SCRIPT_PATH)
    print("  Weather Agent ready on http://localhost:5001")
    yield
    await mcp_bridge.disconnect()


app = FastAPI(lifespan=lifespan)


AGENT_CARD = AgentCard(
    name="Weather Agent",
    description="Assesses weather conditions for outdoor events using NWS data. "
                "Provides GO/CAUTION/NO-GO recommendations based on temperature, "
                "precipitation, wind, and active alerts.",
    url="http://localhost:5001",
    version="1.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="assess_weather",
            name="Weather Assessment",
            description="Evaluate weather conditions for a location and provide "
                        "a safety recommendation for outdoor events.",
            tags=["weather", "safety", "outdoor"],
            examples=[
                "Assess weather for Austin TX this Saturday",
                "Is it safe to hold an outdoor concert in Miami?",
            ],
        ),
    ],
)


@app.get("/.well-known/agent.json")
async def get_agent_card():
    return AGENT_CARD.model_dump()


@app.post("/a2a/tasks/send")
async def handle_task(request: TaskSendRequest):
    """Handle an incoming A2A task — assess weather for a location."""
    data = extract_data(request.message)

    city = data.get("city", "").lower()
    state = data.get("state", "TX")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    time_pref = data.get("time_preference", "")

    # Resolve coordinates from city name if not provided
    if (not latitude or not longitude) and city in CITY_COORDS:
        latitude, longitude = CITY_COORDS[city]

    if not latitude or not longitude:
        return TaskSendResponse(
            id=request.id,
            status=TaskStatus(state=TaskState.FAILED),
            messages=[make_agent_message(
                text=f"Cannot resolve coordinates for city: {data.get('city', 'unknown')}",
                data={"error": "unknown_city"},
            )],
        )

    # Call MCP tools via the bridge
    print(f"  Task {request.id[:8]}... | get_alerts(state={state})")
    alerts_text = await mcp_bridge.call_tool("get_alerts", {"state": state})

    print(f"  Task {request.id[:8]}... | get_forecast(lat={latitude}, lon={longitude})")
    forecast_text = await mcp_bridge.call_tool(
        "get_forecast", {"latitude": latitude, "longitude": longitude}
    )

    # Assess
    assessment = assess_weather(alerts_text, forecast_text, time_pref)
    print(f"  Task {request.id[:8]}... | recommendation={assessment['recommendation']}")

    return TaskSendResponse(
        id=request.id,
        status=TaskStatus(state=TaskState.COMPLETED),
        messages=[make_agent_message(
            text=f"Weather assessment for {data.get('city', 'unknown')}, {state}: "
                 f"{assessment['recommendation']}",
            data=assessment,
        )],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
