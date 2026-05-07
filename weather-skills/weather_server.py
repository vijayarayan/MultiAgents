"""
MCP Weather Server — demonstrates all three MCP primitives:
  1. Tools      — get_alerts, get_forecast
  2. Resources  — state codes lookup, alert severity guide
  3. Prompts    — outdoor_event_readiness, severe_weather_briefing, multi_day_trip_planner
"""

import json
import httpx
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-skills")

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-skills-mcp/1.0"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _nws_get(url: str) -> dict:
    """Make a request to the NWS API with proper headers."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


# ===========================================================================
# RESOURCES — static data the server exposes for the LLM / client to read
# ===========================================================================

@mcp.resource("weather://state-codes")
def get_state_codes() -> str:
    """JSON mapping of US two-letter state codes to full state names."""
    state_codes_path = Path(__file__).parent / "state_codes.json"
    return state_codes_path.read_text()


@mcp.resource("weather://alert-severity-guide")
def get_alert_severity_guide() -> str:
    """Guide explaining NWS alert severity levels and recommended actions."""
    return json.dumps({
        "severity_levels": [
            {
                "level": "Extreme",
                "description": "Extraordinary threat to life or property",
                "action": "Take immediate action to protect life"
            },
            {
                "level": "Severe",
                "description": "Significant threat to life or property",
                "action": "Take action to protect life and property"
            },
            {
                "level": "Moderate",
                "description": "Possible threat to life or property",
                "action": "Be prepared to take action"
            },
            {
                "level": "Minor",
                "description": "Minimal threat to life or property",
                "action": "Monitor conditions and be aware"
            },
            {
                "level": "Unknown",
                "description": "Severity not yet determined",
                "action": "Stay informed and monitor updates"
            }
        ],
        "alert_types_ranked": [
            "Warning  — hazardous event is occurring or imminent",
            "Watch    — hazardous event is possible",
            "Advisory — weather event causing inconvenience but not life-threatening",
            "Statement— informational, follow-up to prior alerts"
        ]
    }, indent=2)


# ===========================================================================
# TOOLS — callable functions the LLM can invoke
# ===========================================================================

@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get active weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, TX, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active?area={state.upper()}"
    data = await _nws_get(url)

    features = data.get("features", [])
    if not features:
        return f"No active weather alerts for {state.upper()}."

    alerts = []
    for feature in features[:20]:  # cap at 20
        props = feature.get("properties", {})
        alerts.append(
            f"Event: {props.get('event', 'Unknown')}\n"
            f"Severity: {props.get('severity', 'Unknown')}\n"
            f"Headline: {props.get('headline', 'No headline')}\n"
            f"Area: {props.get('areaDesc', 'Unknown area')}\n"
            f"Description: {props.get('description', 'No description')}\n"
        )

    return f"Active alerts for {state.upper()}:\n\n" + "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get the weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # Step 1: get the forecast grid endpoint for this location
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await _nws_get(points_url)

    forecast_url = points_data["properties"]["forecast"]

    # Step 2: get the actual forecast
    forecast_data = await _nws_get(forecast_url)

    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # next 5 periods
        forecasts.append(
            f"{period['name']}:\n"
            f"  Temperature: {period['temperature']}{period['temperatureUnit']}\n"
            f"  Wind: {period['windSpeed']} {period['windDirection']}\n"
            f"  Forecast: {period['detailedForecast']}\n"
        )

    return "Forecast:\n\n" + "\n".join(forecasts)


# ===========================================================================
# PROMPTS (SKILLS) — reusable prompt templates with domain expertise
# ===========================================================================

@mcp.prompt()
def outdoor_event_readiness(location: str, date: str, event_type: str) -> str:
    """Assess whether an outdoor event is safe to hold given weather conditions.

    Args:
        location: City and state (e.g. "Austin, TX")
        date: Target date (e.g. "this Saturday", "March 15")
        event_type: Type of event (e.g. "wedding", "concert", "5K run")
    """
    return f"""You are an outdoor event weather advisor. A user wants to hold a
{event_type} in {location} on {date}.

Follow these steps exactly:

1. First, read the resource `weather://state-codes` to confirm the state code.
2. Call `get_alerts` for the state to check for any active weather alerts.
3. Call `get_forecast` for the location coordinates (use your knowledge of the
   city's latitude/longitude).
4. Analyze the results and evaluate:
   - Temperature: Is it within a comfortable range (50-95F)?
   - Precipitation: Any rain/snow in the forecast?
   - Wind: Are winds above 25 mph?
   - Active alerts: Any warnings or watches for the area?
5. Provide a clear recommendation:
   - GO: Conditions are favorable. Proceed as planned.
   - CAUTION: Some concerns exist. List specific mitigations
     (e.g. tents, time shift, hydration stations, wind barriers).
   - NO-GO: Conditions are dangerous. Explain why and suggest the next
     best date from the forecast periods.
6. End with a concise packing/preparation checklist for the event organizer."""


@mcp.prompt()
def severe_weather_briefing(state: str) -> str:
    """Generate a structured severe weather briefing for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, TX, NY)
    """
    return f"""You are an emergency weather briefer providing a structured
situation report for {state}.

Follow these steps exactly:

1. Read the resource `weather://alert-severity-guide` to understand severity levels.
2. Call `get_alerts` for {state}.
3. For each alert, classify it by severity: Extreme > Severe > Moderate > Minor.
4. If any Warning-level or Extreme alerts exist, call `get_forecast` for the
   center of the affected area to get the detailed outlook.
5. Produce a structured briefing in this format:

   ## Severe Weather Briefing: {state}
   **Status:** [ALL CLEAR / ADVISORY / WARNING / EMERGENCY]

   ### Active Alerts (sorted by severity)
   For each alert:
   - Type & severity
   - Affected areas
   - Time range
   - Key details

   ### Forecast Outlook
   Summary of the next 24-48 hours for affected regions.

   ### Recommended Actions
   Specific steps residents should take, based on the severity guide."""


@mcp.prompt()
def multi_day_trip_planner(destinations: str) -> str:
    """Plan a multi-destination trip with weather-optimized scheduling.

    Args:
        destinations: Comma-separated list of cities (e.g. "Miami FL, Austin TX, Denver CO")
    """
    return f"""You are a travel weather planner helping optimize a multi-city trip.
The user wants to visit: {destinations}

Follow these steps exactly:

1. Read the resource `weather://state-codes` to confirm state codes.
2. For EACH destination:
   a. Call `get_alerts` for the state.
   b. Call `get_forecast` for the city coordinates.
3. Build a comparison table:
   | City | Temperature Range | Precipitation Risk | Wind | Active Alerts |
4. Recommend the optimal visit order based on:
   - Avoid cities with active weather warnings first
   - Visit cities with the best weather windows first
   - Group nearby cities together when weather is similar
5. For each city, suggest:
   - Best day/time window for outdoor activities
   - Weather-related items to pack
   - Any weather risks to watch for
6. End with a day-by-day itinerary recommendation."""


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
