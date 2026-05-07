# MCP Client

A minimal Python client that connects to any [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server over stdio and exposes its tools to OpenAI's GPT-4o via function calling. Provides an interactive CLI chat loop where the LLM can invoke MCP tools on demand.

---

## How It Works

```
User Query
    │
    ▼
MCPClient.process_query()
    │
    ├─ Fetches tool list from MCP server  (session.list_tools)
    ├─ Sends query + tools to GPT-4o      (openai.chat.completions)
    │
    └─ Tool call loop:
         ├─ GPT-4o returns tool_calls
         ├─ Client executes each via MCP  (session.call_tool)
         ├─ Results fed back to GPT-4o
         └─ Repeat until no more tool calls → return final text
```

The client acts as a bridge: MCP tools are translated into OpenAI function-calling format so GPT-4o can decide when and how to call them. The loop continues until the model produces a plain-text answer with no further tool calls.

---

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- An MCP server script (`.py` or `.js`)

---

## Setup

### 1. Install dependencies

```bash
cd mcp-client
uv sync
```

### 2. Configure environment

```bash
cp env.example .env
# Edit .env:
# OPENAI_API_KEY=sk-...
```

---

## Usage

```bash
uv run client.py <path_to_mcp_server_script>
```

**Examples:**

```bash
# Connect to a local Python MCP server
uv run client.py ../weather-skills/weather_server.py

# Connect to a JavaScript MCP server
uv run client.py /path/to/my-server/index.js
```

Once connected, the available tools are printed and you enter an interactive chat loop:

```
Connected to server with tools: ['get_alerts', 'get_forecast']

MCP Client Started!
Type your queries or 'quit' to exit.

Query: What's the weather forecast for Austin TX?
[Calling tool get_forecast with args {'latitude': 30.2672, 'longitude': -97.7431}]
The current forecast for Austin, TX shows partly cloudy skies with a high of 94°F...

Query: quit
```

---

## Project Structure

```
mcp-client/
├── client.py        # MCPClient class — core connection, query processing, chat loop
├── main.py          # Placeholder entry point (uv project scaffold)
├── pyproject.toml   # Project metadata and dependencies
├── env.example      # Environment variable template
├── .env             # Local secrets (not committed)
├── .gitignore       # Excludes .venv, __pycache__, build artifacts
└── .python-version  # Pins Python 3.14 for uv
```

### `client.py` — Key Methods

| Method | Description |
|---|---|
| `connect_to_server(path)` | Spawns the MCP server as a subprocess via stdio, initializes the MCP session, and lists available tools |
| `process_query(query)` | Sends the query to GPT-4o with MCP tools as functions, executes any tool calls, and loops until a final answer is produced |
| `chat_loop()` | Interactive REPL — reads input, calls `process_query`, prints the result |
| `cleanup()` | Closes the MCP session and stdio transport via `AsyncExitStack` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `mcp` | MCP client session and stdio transport |
| `openai` | GPT-4o for natural language reasoning and tool-call orchestration |
| `python-dotenv` | Loads `OPENAI_API_KEY` from `.env` |

---

## Extending

**Use a different model:** change `model="gpt-4o"` in the two `chat.completions.create` calls in `client.py`.

**Connect to an SSE/HTTP MCP server:** replace the `stdio_client` transport with an SSE transport from the `mcp` library. The rest of `MCPClient` (tool listing, tool calling, OpenAI loop) stays the same.

**Add system prompt context:** insert a `{"role": "system", "content": "..."}` entry at the start of the `messages` list in `process_query`.

---

## License

MIT
