# Weather MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that exposes two tools for fetching live US weather data from the [National Weather Service (NWS) API](https://www.weather.gov/documentation/services-web-api). No API key required — NWS is a free public API.

Runs over **stdio**, making it directly usable by any MCP client (Claude Desktop, the `mcp-client` in this repo, the `weather_agent.py` in `A2A_Event_Planner`, etc.).

---

## Tools

### `get_alerts`

Returns active weather alerts for a US state.

| Parameter | Type | Description |
|---|---|---|
| `state` | `str` | Two-letter US state code (e.g. `TX`, `CA`, `NY`) |

**Returns:** Formatted text listing each active alert's event type, affected area, severity, description, and instructions. Returns `"No active alerts for this state."` when the state is clear.

---

### `get_forecast`

Returns the next 5 forecast periods for a geographic coordinate.

| Parameter | Type | Description |
|---|---|---|
| `latitude` | `float` | Latitude of the location |
| `longitude` | `float` | Longitude of the location |

**Returns:** Formatted text for each period showing period name, temperature (°F/°C), wind speed and direction, and a detailed forecast description.

**Two-step NWS lookup:**
1. `GET /points/{lat},{lon}` — resolves coordinates to the NWS forecast grid
2. `GET {forecast_url}` — fetches the actual forecast periods

---

## How It Works

```
MCP Client (stdio)
       │
       ▼
  weather.py  (FastMCP server)
       │
       ├── get_alerts(state)
       │       └── GET api.weather.gov/alerts/active/area/{state}
       │
       └── get_forecast(latitude, longitude)
               ├── GET api.weather.gov/points/{lat},{lon}
               └── GET {forecast_grid_url}
```

The server is built with `FastMCP`, which handles the MCP protocol wire format automatically. `weather.py` only defines the tool functions using the `@mcp.tool()` decorator.

---

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Internet access (calls `api.weather.gov`)

No API key or authentication needed.

---

## Setup

```bash
cd weather
uv sync
```

---

## Running the Server

### Standalone (stdio — for MCP clients)

```bash
uv run weather.py
```

The server starts and waits for MCP messages on stdin. It is not meant to be used interactively — connect an MCP client to it.

### With the mcp-client in this repo

```bash
cd ../mcp-client
uv run client.py ../weather/weather.py
```

Then query it naturally:

```
Connected to server with tools: ['get_alerts', 'get_forecast']

Query: Are there any weather warnings in Texas right now?
[Calling tool get_alerts with args {'state': 'TX'}]
There is currently a Heat Advisory in effect for Central Texas...

Query: What's the forecast for San Francisco?
[Calling tool get_forecast with args {'latitude': 37.7749, 'longitude': -122.4194}]
Tonight: Mostly cloudy, low around 54°F. West wind 10–15 mph...
```

### With Claude Desktop

Add the server to your Claude Desktop MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/weather",
        "run",
        "weather.py"
      ]
    }
  }
}
```

Restart Claude Desktop and the `get_alerts` and `get_forecast` tools will be available in your conversations.

### As a dependency in another project

The `A2A_Event_Planner/weather_agent.py` uses this server via MCP stdio bridge:

```bash
uv run weather_agent.py /absolute/path/to/weather/weather.py
```

---

## Project Structure

```
weather/
├── weather.py       # FastMCP server — defines get_alerts and get_forecast tools
├── main.py          # Placeholder entry point (uv project scaffold)
├── pyproject.toml   # Project metadata and dependencies
├── .python-version  # Pins Python 3.10 for uv
└── .gitignore       # Excludes .venv, __pycache__, build artifacts
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `mcp[cli]` | FastMCP server framework and MCP protocol implementation |
| `httpx` | Async HTTP client for NWS API requests |

---

## NWS API Notes

- **Coverage:** US locations only. Coordinates outside the US will return no data.
- **Rate limits:** NWS asks clients to set a descriptive `User-Agent` header (set to `weather-app/1.0`).
- **Forecast periods:** The server returns the next 5 periods. Each period is roughly 12 hours (day/night). Adjust the slice `periods[:5]` in `weather.py` to change this.
- **Alerts:** Only *active* alerts are returned. Expired or cancelled alerts are not included.

---

## License

MIT
