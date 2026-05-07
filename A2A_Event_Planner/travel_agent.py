"""
Travel Advisor Agent — A2A HTTP server for travel cost and logistics data.

Serves flight costs, hotel prices, travel times, and event impact data
from travel_data.json. Filters and ranks cities by total trip cost.

Usage:
    uv run travel_agent.py
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
# Travel data
# ---------------------------------------------------------------------------

TRAVEL_DATA: list[dict] = []
CITY_INDEX: dict[str, dict] = {}  # lowercase city name → data


def load_travel_data():
    global TRAVEL_DATA, CITY_INDEX
    data_path = Path(__file__).parent / "travel_data.json"
    TRAVEL_DATA = json.loads(data_path.read_text())
    CITY_INDEX = {entry["city"].lower(): entry for entry in TRAVEL_DATA}
    print(f"  Loaded {len(TRAVEL_DATA)} cities from {data_path.name}")


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def compute_city_cost(city_data: dict, origin: str, trip_days: int) -> dict:
    """Compute total trip cost for a city from a given origin."""
    origin_key = _match_origin(origin, city_data["flights"])

    flight_cost = city_data["flights"].get(origin_key)
    if flight_cost is None:
        return {
            "city": city_data["city"],
            "state": city_data["state"],
            "error": f"No flight data from {origin}",
        }

    hotel_base = city_data["avg_hotel_per_night"]
    surcharge = city_data.get("peak_season_surcharge_pct", 0)
    hotel_adjusted = hotel_base * (1 + surcharge / 100) if city_data.get("peak_season") else hotel_base

    total_hotel = round(hotel_adjusted * trip_days, 2)
    total_cost = round(flight_cost + total_hotel, 2)

    travel_hours = city_data["travel_time_hours"].get(origin_key, None)

    return {
        "city": city_data["city"],
        "state": city_data["state"],
        "flight_cost": flight_cost,
        "hotel_per_night_base": hotel_base,
        "hotel_per_night_adjusted": round(hotel_adjusted, 2),
        "peak_season": city_data.get("peak_season", False),
        "peak_surcharge_pct": surcharge,
        "total_hotel": total_hotel,
        "total_estimated_cost": total_cost,
        "travel_time_hours": travel_hours,
        "trip_days": trip_days,
    }


def _match_origin(origin: str, flights: dict) -> str | None:
    """Fuzzy-match origin city name to flight data keys."""
    origin_lower = origin.lower().strip()
    for key in flights:
        if key.lower() == origin_lower:
            return key
    # Partial match
    for key in flights:
        if origin_lower in key.lower() or key.lower() in origin_lower:
            return key
    return origin


def get_events_for_city(city_data: dict) -> list[dict]:
    """Return upcoming events for a city."""
    return city_data.get("upcoming_events", [])


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def handle_compare_costs(data: dict) -> dict:
    """Compare travel costs across multiple cities from an origin."""
    cities_requested = data.get("cities", [])
    origin = data.get("origin", "New York")
    trip_days = data.get("trip_duration_days", 5)

    comparisons = []
    for city_req in cities_requested:
        city_name = city_req.get("city", "") if isinstance(city_req, dict) else city_req
        city_data = CITY_INDEX.get(city_name.lower())

        if not city_data:
            comparisons.append({
                "city": city_name,
                "error": f"City '{city_name}' not found in travel database",
            })
            continue

        cost_info = compute_city_cost(city_data, origin, trip_days)
        events = get_events_for_city(city_data)
        high_impact_events = [e for e in events if e["impact"] == "high"]

        comparisons.append({
            **cost_info,
            "events": events,
            "high_impact_event_warning": len(high_impact_events) > 0,
            "event_warnings": [
                f"{e['name']} ({e['date_range']}): {e['description']}"
                for e in high_impact_events
            ],
        })

    # Rank by total cost (exclude errors)
    valid = [c for c in comparisons if "error" not in c]
    valid.sort(key=lambda c: c["total_estimated_cost"])
    for rank, c in enumerate(valid, 1):
        c["cost_rank"] = rank

    cheapest = valid[0]["city"] if valid else None
    most_expensive = valid[-1]["city"] if valid else None

    return {
        "comparisons": comparisons,
        "cheapest": cheapest,
        "most_expensive": most_expensive,
        "origin": origin,
        "trip_days": trip_days,
        "cities_compared": len(valid),
    }


def handle_check_events(data: dict) -> dict:
    """Check event impact for a specific city."""
    city_name = data.get("city", "")
    city_data = CITY_INDEX.get(city_name.lower())

    if not city_data:
        return {"city": city_name, "error": f"City '{city_name}' not found"}

    events = get_events_for_city(city_data)
    high_impact = [e for e in events if e["impact"] == "high"]
    moderate_impact = [e for e in events if e["impact"] == "moderate"]

    price_impact = "none"
    if high_impact:
        price_impact = "severe — expect 2-3x hotel pricing"
    elif moderate_impact:
        price_impact = "moderate — slight price increases and crowding"

    return {
        "city": city_name,
        "state": city_data["state"],
        "events": events,
        "high_impact_events": high_impact,
        "moderate_impact_events": moderate_impact,
        "price_impact": price_impact,
        "peak_season": city_data.get("peak_season", False),
        "recommendation": "avoid" if high_impact else ("plan around" if moderate_impact else "clear"),
    }


# ---------------------------------------------------------------------------
# FastAPI app with A2A endpoints
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Travel Advisor Agent starting...")
    load_travel_data()
    print("  Travel Advisor Agent ready on http://localhost:5003")
    yield


app = FastAPI(lifespan=lifespan)


AGENT_CARD = AgentCard(
    name="Travel Advisor Agent",
    description="Provides travel logistics data including flight costs, "
                "hotel prices, travel times, and upcoming events or festivals "
                "that may affect trip planning and pricing. Compares multiple "
                "cities and ranks by total estimated trip cost.",
    url="http://localhost:5003",
    version="1.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="compare_travel_costs",
            name="Travel Cost Comparison",
            description="Compare flight, hotel, and total trip costs across "
                        "multiple cities from a given origin.",
            tags=["travel", "cost", "flights", "hotels", "comparison"],
            examples=[
                "Compare Austin, Miami, Denver travel costs from NYC for 5 days",
            ],
        ),
        AgentSkill(
            id="check_events_impact",
            name="Event Impact Check",
            description="Check if upcoming events or festivals in a city affect "
                        "pricing, availability, or crowding.",
            tags=["events", "festivals", "pricing", "impact"],
            examples=[
                "Check events impact in Austin mid-March",
                "Are there any festivals in Miami next week?",
            ],
        ),
    ],
)


@app.get("/.well-known/agent.json")
async def get_agent_card():
    return AGENT_CARD.model_dump()


@app.post("/a2a/tasks/send")
async def handle_task(request: TaskSendRequest):
    """Route A2A tasks to the appropriate handler."""
    data = extract_data(request.message)
    action = data.get("action", "compare_travel_costs")

    print(f"  Task {request.id[:8]}... | action={action}")

    if action == "check_events_impact":
        result = handle_check_events(data)
        city = data.get("city", "unknown")
        rec = result.get("recommendation", "unknown")
        print(f"  Task {request.id[:8]}... | city={city} events_recommendation={rec}")
    else:
        result = handle_compare_costs(data)
        n = result.get("cities_compared", 0)
        cheapest = result.get("cheapest", "?")
        print(f"  Task {request.id[:8]}... | compared {n} cities, cheapest={cheapest}")

    return TaskSendResponse(
        id=request.id,
        status=TaskStatus(state=TaskState.COMPLETED),
        messages=[make_agent_message(
            text=json.dumps(result, indent=2),
            data=result,
        )],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5003)
