"""
Travel Coordinator — orchestrates Weather Agent and Travel Advisor Agent
via A2A to compare cities and recommend the best travel destination.

3-round negotiation:
  Round 1: Parallel weather + cost assessment for all cities
  Round 2: Conflict detection and adjusted queries
  Round 3: Final synthesis with comparison table and recommendation

Usage:
    uv run travel_coordinator.py
"""

import asyncio
import json

import httpx
from openai import OpenAI
from dotenv import load_dotenv

from a2a_protocol import (
    AgentCard,
    DataPart,
    TaskSendRequest,
    TaskSendResponse,
    make_user_message,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_URLS = {
    "weather": "http://localhost:5001",
    "travel": "http://localhost:5003",
}


# ---------------------------------------------------------------------------
# A2A Client
# ---------------------------------------------------------------------------

class A2AClient:
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=60.0)
        self.agents: dict[str, AgentCard] = {}

    async def discover(self, name: str, base_url: str) -> AgentCard:
        resp = await self.http.get(f"{base_url}/.well-known/agent.json")
        resp.raise_for_status()
        card = AgentCard(**resp.json())
        self.agents[name] = card
        return card

    async def send_task(self, name: str, data: dict) -> TaskSendResponse:
        base_url = AGENT_URLS[name]
        message = make_user_message(data=data)
        request = TaskSendRequest(message=message)
        resp = await self.http.post(
            f"{base_url}/a2a/tasks/send",
            json=request.model_dump(),
        )
        resp.raise_for_status()
        return TaskSendResponse(**resp.json())

    async def close(self):
        await self.http.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_data(response: TaskSendResponse) -> dict:
    """Extract the DataPart from an A2A task response."""
    for msg in response.messages:
        for part in msg.parts:
            if isinstance(part, DataPart):
                return part.data
    return {}


def _llm_json(client: OpenAI, system: str, user: str) -> dict:
    """Call OpenAI and parse the response as JSON."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


def _llm_text(client: OpenAI, system: str, user: str) -> str:
    """Call OpenAI and return the raw text response."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=3000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Travel Coordinator
# ---------------------------------------------------------------------------

class TravelCoordinator:
    def __init__(self):
        self.a2a = A2AClient()
        self.openai = OpenAI()

    # --- Phase 2: Parse ---

    def parse_query(self, user_input: str) -> dict:
        return _llm_json(
            self.openai,
            system=(
                "Extract travel comparison parameters from the user query. "
                "Return ONLY valid JSON with these fields:\n"
                '  "cities": [{"city": string, "state": string (2-letter code)}],\n'
                '  "origin": string (city name, e.g. "New York"),\n'
                '  "travel_dates": string (date range or relative description),\n'
                '  "trip_duration_days": integer (default 5 if not specified),\n'
                '  "preferences": string (any mentioned preferences like budget, '
                'activities, etc., or empty string)\n'
                "Always resolve city names to their common form "
                '(e.g. "NYC" → "New York", "LA" → "Los Angeles", "SF" → "San Francisco").'
            ),
            user=user_input,
        )

    # --- Phase 4: Conflict detection ---

    def detect_conflicts(self, weather_results: dict, travel_data: dict) -> dict:
        return _llm_json(
            self.openai,
            system=(
                "You are analyzing weather and travel cost data for a multi-city "
                "trip comparison. Identify conflicts and suggest resolutions.\n\n"
                "Return ONLY valid JSON:\n"
                '  "has_conflicts": boolean,\n'
                '  "conflicts": [string descriptions of each conflict],\n'
                '  "cities_to_eliminate": [city names that are NO-GO due to '
                "dangerous weather (warnings, flooding, hurricanes)],\n"
                '  "best_weather_city": string,\n'
                '  "best_value_city": string,\n'
                '  "weather_cost_conflict": boolean (true if best weather != cheapest),\n'
                '  "event_conflict_cities": [cities where events inflate costs],\n'
                '  "adjustments": {\n'
                '    "recheck_weather_for": [{"city": string, "state": string, '
                '"time_preference": string}] or [],\n'
                '    "recheck_events_for": [string city names] or []\n'
                "  },\n"
                '  "explanation": string\n\n'
                "Conflicts exist when:\n"
                "- Best weather city is significantly more expensive than cheapest\n"
                "- Cheapest city has weather concerns (CAUTION or worse)\n"
                "- High-impact events inflate costs in otherwise good cities\n"
                "- Any city has dangerous weather (active warnings) → eliminate it\n"
                "If no meaningful conflicts exist, set has_conflicts to false."
            ),
            user=json.dumps({
                "weather_per_city": weather_results,
                "travel_costs": travel_data,
            }),
        )

    # --- Phase 5: Synthesis ---

    def synthesize(
        self,
        user_query: str,
        parsed: dict,
        weather_r1: dict,
        travel_r1: dict,
        conflicts: dict,
        weather_r2: dict | None,
        travel_r2: dict | None,
    ) -> str:
        context = {
            "user_request": user_query,
            "parsed_parameters": parsed,
            "round_1": {
                "weather_per_city": weather_r1,
                "travel_costs": travel_r1,
            },
            "conflict_analysis": conflicts,
        }
        if weather_r2:
            context["round_2_adjusted_weather"] = weather_r2
        if travel_r2:
            context["round_2_adjusted_travel"] = travel_r2

        return _llm_text(
            self.openai,
            system=(
                "You are an expert travel advisor. Based on the multi-agent "
                "negotiation data, produce a final travel recommendation.\n\n"
                "Format your response EXACTLY as:\n\n"
                "## Travel Recommendation\n\n"
                "### Winner: [City, State]\n"
                "One paragraph on why this city was selected, balancing weather, "
                "cost, and logistics.\n\n"
                "### Comparison Table\n"
                "| City | Weather | High Temp | Alerts | Flight | Hotel/Night | "
                "Total Est. | Travel Time | Events |\n"
                "(fill for all compared cities)\n\n"
                "### City-by-City Breakdown\n"
                "For each city: 2-3 bullet points covering weather, cost, pros/cons.\n\n"
                "### Eliminated Cities\n"
                "List any cities removed and why. Write 'None' if all cities remained.\n\n"
                "### Negotiation Summary\n"
                "Explain what the agents initially reported, what conflicts were "
                "found, and how they were resolved between rounds.\n\n"
                "### Travel Tips for [Winner City]\n"
                "- Best days for outdoor activities based on forecast\n"
                "- Weather-appropriate packing items\n"
                "- Events to be aware of\n"
                "- Any cost-saving suggestions\n\n"
                "Be specific. Use actual numbers from the data. Do not invent data."
            ),
            user=json.dumps(context, indent=2),
        )

    # --- Main negotiation loop ---

    async def negotiate(self, user_input: str):
        # ===== DISCOVERY =====
        print("\n" + "=" * 52)
        print("       A2A TRAVEL WEATHER COMPARATOR")
        print("=" * 52)

        print("\n[Discovery] Contacting agents...")
        weather_card = await self.a2a.discover("weather", AGENT_URLS["weather"])
        travel_card = await self.a2a.discover("travel", AGENT_URLS["travel"])
        print(f"  Weather Agent:  {weather_card.name} ({weather_card.url})")
        for s in weather_card.skills:
            print(f"    - {s.name}: {s.description[:60]}...")
        print(f"  Travel Agent:   {travel_card.name} ({travel_card.url})")
        for s in travel_card.skills:
            print(f"    - {s.name}: {s.description[:60]}...")

        # ===== PARSE =====
        print("\n[Parse] Extracting travel parameters...")
        parsed = self.parse_query(user_input)
        cities = parsed.get("cities", [])
        origin = parsed.get("origin", "New York")
        trip_days = parsed.get("trip_duration_days", 5)

        print(f"  Cities:    {', '.join(c['city'] + ' ' + c['state'] for c in cities)}")
        print(f"  Origin:    {origin}")
        print(f"  Dates:     {parsed.get('travel_dates', 'not specified')}")
        print(f"  Duration:  {trip_days} days")
        if parsed.get("preferences"):
            print(f"  Prefs:     {parsed['preferences']}")

        if len(cities) < 2:
            print("\n  ERROR: Need at least 2 cities to compare.")
            return

        # ===== ROUND 1: INITIAL ASSESSMENT =====
        print("\n" + "-" * 52)
        print("  ROUND 1: Initial Parallel Assessment")
        print("-" * 52)

        # Build parallel tasks: one weather task per city + one travel cost task
        weather_tasks = []
        for city_info in cities:
            weather_tasks.append(
                self.a2a.send_task("weather", {
                    "action": "assess_weather",
                    "city": city_info["city"],
                    "state": city_info["state"],
                })
            )

        travel_task = self.a2a.send_task("travel", {
            "action": "compare_travel_costs",
            "cities": cities,
            "origin": origin,
            "trip_duration_days": trip_days,
        })

        # Execute all in parallel
        print(f"  Sending {len(cities)} weather tasks + 1 travel task in parallel...")
        all_results = await asyncio.gather(*weather_tasks, travel_task)

        # Separate results
        weather_responses = all_results[:-1]
        travel_response = all_results[-1]

        # Extract structured data
        weather_r1: dict[str, dict] = {}
        for city_info, resp in zip(cities, weather_responses):
            city_name = city_info["city"]
            weather_r1[city_name] = _extract_data(resp)

        travel_r1 = _extract_data(travel_response)

        # Print Round 1 summary
        print("\n  Weather Results:")
        for city_name, w in weather_r1.items():
            rec = w.get("recommendation", "?")
            concerns = w.get("concerns", [])
            temp_hi = w.get("high_temp", "?")
            print(f"    {city_name}: {rec} (high: {temp_hi}F)")
            for c in concerns[:2]:
                print(f"      - {c}")

        print("\n  Travel Cost Results:")
        for comp in travel_r1.get("comparisons", []):
            if "error" in comp:
                print(f"    {comp['city']}: {comp['error']}")
                continue
            rank = comp.get("cost_rank", "?")
            total = comp.get("total_estimated_cost", "?")
            flight = comp.get("flight_cost", "?")
            hotel = comp.get("hotel_per_night_adjusted", "?")
            warnings = comp.get("event_warnings", [])
            print(f"    #{rank} {comp['city']}: ${total} total "
                  f"(flight ${flight} + hotel ${hotel}/night x {trip_days}d)")
            for warn in warnings:
                print(f"       EVENT: {warn}")

        # ===== ROUND 2: CONFLICT RESOLUTION =====
        print("\n" + "-" * 52)
        print("  ROUND 2: Conflict Detection & Resolution")
        print("-" * 52)

        print("  Analyzing weather vs travel data for conflicts...")
        conflicts = self.detect_conflicts(weather_r1, travel_r1)

        weather_r2: dict[str, dict] | None = None
        travel_r2: dict | None = None

        if not conflicts.get("has_conflicts"):
            print("  No conflicts — agents agree. Proceeding to synthesis.")
        else:
            print(f"\n  CONFLICTS FOUND:")
            for conflict in conflicts.get("conflicts", []):
                print(f"    - {conflict}")

            eliminated = conflicts.get("cities_to_eliminate", [])
            if eliminated:
                print(f"\n  ELIMINATING: {', '.join(eliminated)} (weather too dangerous)")

            best_weather = conflicts.get("best_weather_city", "?")
            best_value = conflicts.get("best_value_city", "?")
            if conflicts.get("weather_cost_conflict"):
                print(f"  Best weather: {best_weather}  |  Best value: {best_value}")

            print(f"  Explanation: {conflicts.get('explanation', '')}")

            # Send adjusted tasks based on conflict analysis
            adjusted_tasks = []
            adjusted_labels = []

            # Re-check weather with time preferences
            recheck_weather = conflicts.get("adjustments", {}).get("recheck_weather_for", [])
            for item in recheck_weather:
                adjusted_tasks.append(
                    self.a2a.send_task("weather", {
                        "action": "assess_weather",
                        "city": item["city"],
                        "state": item["state"],
                        "time_preference": item.get("time_preference", ""),
                    })
                )
                adjusted_labels.append(("weather", item["city"]))

            # Re-check events for specific cities
            recheck_events = conflicts.get("adjustments", {}).get("recheck_events_for", [])
            for city_name in recheck_events:
                adjusted_tasks.append(
                    self.a2a.send_task("travel", {
                        "action": "check_events_impact",
                        "city": city_name,
                    })
                )
                adjusted_labels.append(("travel_events", city_name))

            # If cities were eliminated, recompute costs for remaining
            if eliminated:
                remaining_cities = [c for c in cities if c["city"] not in eliminated]
                if remaining_cities:
                    adjusted_tasks.append(
                        self.a2a.send_task("travel", {
                            "action": "compare_travel_costs",
                            "cities": remaining_cities,
                            "origin": origin,
                            "trip_duration_days": trip_days,
                        })
                    )
                    adjusted_labels.append(("travel_recompute", "remaining"))

            if adjusted_tasks:
                print(f"\n  Sending {len(adjusted_tasks)} adjusted task(s)...")
                adjusted_results = await asyncio.gather(*adjusted_tasks)

                weather_r2 = {}
                for label, result in zip(adjusted_labels, adjusted_results):
                    agent_type, city_or_label = label
                    data = _extract_data(result)

                    if agent_type == "weather":
                        weather_r2[city_or_label] = data
                        rec = data.get("recommendation", "?")
                        print(f"    Weather (adjusted) {city_or_label}: {rec}")

                    elif agent_type == "travel_events":
                        if travel_r2 is None:
                            travel_r2 = {}
                        travel_r2[f"events_{city_or_label}"] = data
                        rec = data.get("recommendation", "?")
                        impact = data.get("price_impact", "?")
                        print(f"    Events {city_or_label}: {rec} (impact: {impact})")

                    elif agent_type == "travel_recompute":
                        if travel_r2 is None:
                            travel_r2 = {}
                        travel_r2["recomputed_costs"] = data
                        cheapest = data.get("cheapest", "?")
                        print(f"    Recomputed costs (excl. eliminated): cheapest={cheapest}")
            else:
                print("  No adjusted tasks needed — conflicts are informational only.")

        # ===== ROUND 3: SYNTHESIS =====
        print("\n" + "-" * 52)
        print("  ROUND 3: Final Synthesis")
        print("-" * 52)

        print("  Generating final travel recommendation...")
        plan = self.synthesize(
            user_query=user_input,
            parsed=parsed,
            weather_r1=weather_r1,
            travel_r1=travel_r1,
            conflicts=conflicts,
            weather_r2=weather_r2,
            travel_r2=travel_r2,
        )

        print("\n" + "=" * 52)
        print(plan)
        print("=" * 52)

    async def cleanup(self):
        await self.a2a.close()


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

async def main():
    coordinator = TravelCoordinator()

    print("=" * 52)
    print("  A2A Travel Weather Comparator — Coordinator")
    print("=" * 52)
    print()
    print("  Agents expected:")
    print("    Weather Agent        → localhost:5001")
    print("    Travel Advisor Agent → localhost:5003")
    print()
    print("  Example queries:")
    print("    Compare Austin, Miami, Denver for a trip from NYC")
    print("    Best city to visit from Chicago: Nashville, Phoenix, or Seattle?")
    print("    Budget trip from LA to Denver, Phoenix, or Austin next week")
    print()
    print("  Type 'quit' to exit.")
    print()

    try:
        while True:
            try:
                query = input("Query: ").strip()
                if not query:
                    continue
                if query.lower() == "quit":
                    break
                await coordinator.negotiate(query)
                print()
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                print("\nMake sure both agents are running "
                      "(weather on :5001, travel on :5003)")
    finally:
        await coordinator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
