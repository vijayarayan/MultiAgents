[README.md](A2A_Event_Planner/README.md) created. Here's what it covers:

- **Architecture diagram** showing the agent/coordinator topology and the MCP bridge from the Weather Agent to the NWS server
- **Agent & coordinator table** mapping each file to its port, role, and use case
- **Protocol layer** explanation of the A2A types in `a2a_protocol.py`
- **3-round negotiation flow** that both coordinators share (initial → conflict resolution → synthesis)
- **Setup instructions** with `uv sync`, `.env` config, and per-terminal startup commands for all three agents
- **Usage examples** with realistic query strings and output structure for both coordinators
- **Data file schemas** for `venue_data.json` and `travel_data.json` with scoring logic
- **Extension guide** for adding new cities and venues
- **Dependency table** and a `.gitignore` recommendation to keep `.env` and `.venv/` out of git
