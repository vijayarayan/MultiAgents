"""
Venue Agent — A2A HTTP server for venue discovery and filtering.

Serves venue data from venue_data.json and filters based on constraints
such as city, event type, time preference, and weather mitigations.

Usage:
    uv run venue_agent.py
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

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
# Venue data
# ---------------------------------------------------------------------------

VENUES: list[dict] = []


def load_venues():
    global VENUES
    venue_path = Path(__file__).parent / "venue_data.json"
    VENUES = json.loads(venue_path.read_text())
    print(f"  Loaded {len(VENUES)} venues from {venue_path.name}")


# ---------------------------------------------------------------------------
# Venue filtering logic
# ---------------------------------------------------------------------------

def find_venues(
    city: str = "",
    state: str = "",
    event_type: str = "",
    time_preference: str = "",
    require_covered: bool = False,
    require_indoor_backup: bool = False,
    min_capacity: int = 0,
) -> list[dict]:
    """Filter venues based on constraints. Returns matching venues with scores."""
    results = []

    for venue in VENUES:
        # City filter
        if city and venue["city"].lower() != city.lower():
            continue

        # State filter
        if state and venue["state"].upper() != state.upper():
            continue

        # Time slot filter
        if time_preference and time_preference not in venue["time_slots"]:
            continue

        # Covered filter
        if require_covered and not venue["covered"]:
            continue

        # Indoor backup filter
        if require_indoor_backup and "AC_indoor_backup" not in venue["amenities"]:
            continue

        # Capacity filter
        if min_capacity and venue["capacity"] < min_capacity:
            continue

        # Score the venue (higher = better match for constraints)
        score = 0
        if venue["covered"]:
            score += 2
        if "AC_indoor_backup" in venue["amenities"]:
            score += 3
        if "shade_structures" in venue["amenities"]:
            score += 1
        if "fans" in venue["amenities"] or "misting_stations" in venue["amenities"]:
            score += 1
        if "heaters" in venue["amenities"]:
            score += 1

        results.append({**venue, "match_score": score})

    # Sort by score descending
    results.sort(key=lambda v: v["match_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# FastAPI app with A2A endpoints
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Venue Agent starting...")
    load_venues()
    print("  Venue Agent ready on http://localhost:5002")
    yield


app = FastAPI(lifespan=lifespan)


AGENT_CARD = AgentCard(
    name="Venue Agent",
    description="Discovers and recommends event venues based on location, capacity, "
                "weather constraints, and event type. Can adjust recommendations when "
                "given weather-related constraints like heat, rain, or wind.",
    url="http://localhost:5002",
    version="1.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="find_venues",
            name="Venue Discovery",
            description="Find and filter venues by city, capacity, time slot, "
                        "and weather mitigation features.",
            tags=["venue", "event", "planning"],
            examples=[
                "Find outdoor venues in Austin TX",
                "Find covered evening venues in Miami for 100 guests",
            ],
        ),
        AgentSkill(
            id="adjust_venues",
            name="Venue Adjustment",
            description="Re-filter venues with additional weather constraints "
                        "(e.g. require cover, shift to evening, need indoor backup).",
            tags=["venue", "weather", "adjustment"],
            examples=[
                "Re-evaluate Austin venues: avoid afternoon heat, prefer covered",
            ],
        ),
    ],
)


@app.get("/.well-known/agent.json")
async def get_agent_card():
    return AGENT_CARD.model_dump()


@app.post("/a2a/tasks/send")
async def handle_task(request: TaskSendRequest):
    """Handle an incoming A2A task — find or adjust venue recommendations."""
    data = extract_data(request.message)

    city = data.get("city", "")
    state = data.get("state", "")
    time_preference = data.get("time_preference", "")
    require_covered = data.get("require_covered", False)
    require_indoor_backup = data.get("require_indoor_backup", False)
    min_capacity = data.get("min_capacity", 0)
    action = data.get("action", "find_venues")

    print(f"  Task {request.id[:8]}... | action={action} city={city} "
          f"time={time_preference or 'any'} covered={require_covered}")

    venues = find_venues(
        city=city,
        state=state,
        time_preference=time_preference,
        require_covered=require_covered,
        require_indoor_backup=require_indoor_backup,
        min_capacity=min_capacity,
    )

    # Build response
    venue_summaries = []
    for v in venues:
        venue_summaries.append({
            "id": v["id"],
            "name": v["name"],
            "type": v["type"],
            "covered": v["covered"],
            "capacity": v["capacity"],
            "time_slots": v["time_slots"],
            "amenities": v["amenities"],
            "price_range": v["price_range"],
            "description": v["description"],
            "match_score": v["match_score"],
        })

    adjustments = ""
    if action == "adjust_venues":
        filters_applied = []
        if time_preference:
            filters_applied.append(f"time={time_preference}")
        if require_covered:
            filters_applied.append("covered/shaded required")
        if require_indoor_backup:
            filters_applied.append("indoor backup required")
        adjustments = f"Adjusted with constraints: {', '.join(filters_applied)}"

    text = f"Found {len(venue_summaries)} venue(s) in {city}, {state}"
    if adjustments:
        text += f". {adjustments}"

    print(f"  Task {request.id[:8]}... | found {len(venue_summaries)} venues")

    return TaskSendResponse(
        id=request.id,
        status=TaskStatus(state=TaskState.COMPLETED),
        messages=[make_agent_message(
            text=text,
            data={
                "venues": venue_summaries,
                "count": len(venue_summaries),
                "filters": {
                    "city": city,
                    "state": state,
                    "time_preference": time_preference,
                    "require_covered": require_covered,
                    "require_indoor_backup": require_indoor_backup,
                },
                "adjustments": adjustments,
            },
        )],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
