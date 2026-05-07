"""
Coordinator Agent — orchestrates multi-agent negotiation via A2A protocol.

Discovers Weather Agent and Venue Agent, sends tasks, detects conflicts
between their responses, and negotiates adjusted plans.

Uses OpenAI for:
  1. Parsing user queries into structured parameters
  2. Detecting conflicts between agent responses
  3. Synthesizing the final event plan

Usage:
    uv run coordinator.py
"""

import asyncio
import json
import sys

import httpx
from openai import OpenAI
from dotenv import load_dotenv

from a2a_protocol import (
    AgentCard,
    TaskSendRequest,
    TaskSendResponse,
    TaskState,
    DataPart,
    make_user_message,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_URLS = {
    "weather": "http://localhost:5001",
    "venue": "http://localhost:5002",
}


# ---------------------------------------------------------------------------
# A2A Client — talks to agents over HTTP
# ---------------------------------------------------------------------------

class A2AClient:
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=60.0)
        self.agents: dict[str, AgentCard] = {}

    async def discover(self, name: str, base_url: str) -> AgentCard:
        """Fetch an agent's card from its well-known URL."""
        resp = await self.http.get(f"{base_url}/.well-known/agent.json")
        resp.raise_for_status()
        card = AgentCard(**resp.json())
        self.agents[name] = card
        return card

    async def send_task(self, name: str, data: dict, text: str = "") -> TaskSendResponse:
        """Send a task to a named agent and return the response."""
        base_url = AGENT_URLS[name]
        message = make_user_message(text=text, data=data)
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
# Coordinator — negotiation logic
# ---------------------------------------------------------------------------

class Coordinator:
    def __init__(self):
        self.a2a = A2AClient()
        self.openai = OpenAI()

    def _llm(self, system: str, user: str) -> str:
        """Call OpenAI with a system + user message and return the text."""
        resp = self.openai.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    # --- Step 1: Parse user query ---

    def parse_query(self, user_input: str) -> dict:
        """Extract structured event parameters from natural language."""
        raw = self._llm(
            system=(
                "Extract event planning parameters from the user query. "
                "Return ONLY valid JSON with these fields:\n"
                '  "city": string (city name),\n'
                '  "state": string (2-letter state code),\n'
                '  "date": string (the date or relative description),\n'
                '  "event_type": string (type of event),\n'
                '  "guest_count": integer or null,\n'
                '  "time_preference": string ("morning","afternoon","evening") or ""\n'
                "If the user does not specify a field, use reasonable defaults. "
                "Always include city and state."
            ),
            user=user_input,
        )
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        return json.loads(raw)

    # --- Step 2: Detect conflicts ---

    def detect_conflicts(self, weather_data: dict, venue_data: dict) -> dict:
        """Use OpenAI to detect conflicts between weather and venue responses."""
        raw = self._llm(
            system=(
                "You are analyzing weather and venue data for an outdoor event. "
                "Determine if there are conflicts that require adjustments.\n\n"
                "Return ONLY valid JSON:\n"
                '  "has_conflicts": boolean,\n'
                '  "concerns": [list of specific concern strings],\n'
                '  "suggested_time_preference": "morning"|"afternoon"|"evening"|"",\n'
                '  "require_covered": boolean,\n'
                '  "require_indoor_backup": boolean,\n'
                '  "explanation": string (brief explanation of reasoning)\n\n'
                "Conflicts exist if:\n"
                "- Temperature > 95F or < 40F for outdoor event\n"
                "- Active weather alerts (warnings/watches)\n"
                "- Rain/storm in forecast\n"
                "- High winds > 25 mph\n"
                "If weather is CAUTION or worse, suggest mitigations."
            ),
            user=json.dumps({
                "weather_assessment": weather_data,
                "venue_options": venue_data,
            }),
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()
        return json.loads(raw)

    # --- Step 3: Synthesize final plan ---

    def synthesize_plan(
        self,
        user_query: str,
        parsed: dict,
        weather_initial: dict,
        venues_initial: dict,
        conflicts: dict,
        weather_adjusted: dict | None,
        venues_adjusted: dict | None,
    ) -> str:
        """Produce the final event plan using all gathered information."""
        context = {
            "user_request": user_query,
            "parsed_parameters": parsed,
            "round_1_weather": weather_initial,
            "round_1_venues": venues_initial,
            "conflict_analysis": conflicts,
        }
        if weather_adjusted:
            context["round_2_weather_adjusted"] = weather_adjusted
        if venues_adjusted:
            context["round_2_venues_adjusted"] = venues_adjusted

        return self._llm(
            system=(
                "You are an expert outdoor event planner. Based on the multi-agent "
                "negotiation data below, produce a final event plan.\n\n"
                "Format your response as:\n"
                "## Event Plan: [Event Type] in [City]\n"
                "### Decision: GO / CAUTION / NO-GO\n"
                "### Recommended Venue\n"
                "  - Venue name and why it was selected\n"
                "### Timing\n"
                "  - Recommended time slot and reasoning\n"
                "### Weather Summary\n"
                "  - Key weather factors\n"
                "### Preparation Checklist\n"
                "  - Actionable items for the organizer\n"
                "### Negotiation Summary\n"
                "  - Brief explanation of how the agents negotiated "
                "(what changed between rounds)\n\n"
                "Be specific and actionable. Reference actual venue names and data."
            ),
            user=json.dumps(context, indent=2),
        )

    # --- Main negotiation loop ---

    async def negotiate(self, user_input: str):
        """Run the full 3-round negotiation for an event planning query."""

        # ===== DISCOVERY =====
        print("\n╔══════════════════════════════════════════╗")
        print("║       A2A MULTI-AGENT NEGOTIATION        ║")
        print("╚══════════════════════════════════════════╝")

        print("\n[Discovery] Contacting agents...")
        weather_card = await self.a2a.discover("weather", AGENT_URLS["weather"])
        venue_card = await self.a2a.discover("venue", AGENT_URLS["venue"])
        print(f"  Weather Agent: {weather_card.name} — {len(weather_card.skills)} skill(s)")
        print(f"  Venue Agent:   {venue_card.name} — {len(venue_card.skills)} skill(s)")

        # ===== PARSE QUERY =====
        print("\n[Parse] Extracting event parameters...")
        parsed = self.parse_query(user_input)
        print(f"  City:       {parsed.get('city')}")
        print(f"  State:      {parsed.get('state')}")
        print(f"  Date:       {parsed.get('date')}")
        print(f"  Event:      {parsed.get('event_type')}")
        print(f"  Guests:     {parsed.get('guest_count', 'not specified')}")
        print(f"  Time pref:  {parsed.get('time_preference') or 'any'}")

        # ===== ROUND 1: INITIAL ASSESSMENT =====
        print("\n┌──────────────────────────────────────────┐")
        print("│        ROUND 1: Initial Assessment       │")
        print("└──────────────────────────────────────────┘")

        print("  Sending tasks to both agents in parallel...")

        weather_task, venue_task = await asyncio.gather(
            self.a2a.send_task("weather", {
                "action": "assess_weather",
                "city": parsed.get("city", ""),
                "state": parsed.get("state", ""),
                "latitude": parsed.get("latitude"),
                "longitude": parsed.get("longitude"),
            }),
            self.a2a.send_task("venue", {
                "action": "find_venues",
                "city": parsed.get("city", ""),
                "state": parsed.get("state", ""),
                "min_capacity": parsed.get("guest_count") or 0,
                "time_preference": parsed.get("time_preference", ""),
            }),
        )

        # Extract response data
        weather_data = {}
        venue_data = {}
        for msg in weather_task.messages:
            for part in msg.parts:
                if isinstance(part, DataPart):
                    weather_data = part.data
        for msg in venue_task.messages:
            for part in msg.parts:
                if isinstance(part, DataPart):
                    venue_data = part.data

        w_rec = weather_data.get("recommendation", "UNKNOWN")
        v_count = venue_data.get("count", 0)
        print(f"  Weather Agent: {w_rec}")
        if weather_data.get("concerns"):
            for c in weather_data["concerns"]:
                print(f"    - {c}")
        print(f"  Venue Agent:   {v_count} venue(s) found")
        for v in venue_data.get("venues", [])[:3]:
            print(f"    - {v['name']} ({v['type']}, {'covered' if v['covered'] else 'open'})")

        # ===== ROUND 2: CONFLICT RESOLUTION =====
        print("\n┌──────────────────────────────────────────┐")
        print("│      ROUND 2: Conflict Resolution        │")
        print("└──────────────────────────────────────────┘")

        print("  Analyzing for conflicts...")
        conflicts = self.detect_conflicts(weather_data, venue_data)

        weather_adjusted = None
        venues_adjusted = None

        if conflicts.get("has_conflicts"):
            print(f"  CONFLICTS DETECTED:")
            for concern in conflicts.get("concerns", []):
                print(f"    - {concern}")
            print(f"  Resolution: {conflicts.get('explanation', '')}")

            # Send adjusted tasks
            print("  Sending adjusted tasks...")
            adjusted_tasks = []

            # Adjusted venue request
            adjusted_venue_data = {
                "action": "adjust_venues",
                "city": parsed.get("city", ""),
                "state": parsed.get("state", ""),
                "time_preference": conflicts.get("suggested_time_preference", ""),
                "require_covered": conflicts.get("require_covered", False),
                "require_indoor_backup": conflicts.get("require_indoor_backup", False),
                "min_capacity": parsed.get("guest_count") or 0,
            }
            adjusted_tasks.append(
                self.a2a.send_task("venue", adjusted_venue_data)
            )

            # Adjusted weather request (with time preference)
            adjusted_weather_data = {
                "action": "assess_weather",
                "city": parsed.get("city", ""),
                "state": parsed.get("state", ""),
                "time_preference": conflicts.get("suggested_time_preference", ""),
            }
            adjusted_tasks.append(
                self.a2a.send_task("weather", adjusted_weather_data)
            )

            adjusted_results = await asyncio.gather(*adjusted_tasks)

            # Extract adjusted data
            venues_adjusted_data = {}
            weather_adjusted_data = {}
            for msg in adjusted_results[0].messages:
                for part in msg.parts:
                    if isinstance(part, DataPart):
                        venues_adjusted_data = part.data
            for msg in adjusted_results[1].messages:
                for part in msg.parts:
                    if isinstance(part, DataPart):
                        weather_adjusted_data = part.data

            venues_adjusted = venues_adjusted_data
            weather_adjusted = weather_adjusted_data

            w_adj_rec = weather_adjusted.get("recommendation", "UNKNOWN")
            v_adj_count = venues_adjusted.get("count", 0)
            print(f"  Weather Agent (adjusted): {w_adj_rec}")
            print(f"  Venue Agent (adjusted):   {v_adj_count} venue(s)")
            if venues_adjusted.get("adjustments"):
                print(f"    {venues_adjusted['adjustments']}")
            for v in venues_adjusted.get("venues", [])[:3]:
                print(f"    - {v['name']} ({v['type']}, score={v['match_score']})")
        else:
            print("  No conflicts detected — Round 1 results are acceptable.")

        # ===== ROUND 3: SYNTHESIS =====
        print("\n┌──────────────────────────────────────────┐")
        print("│        ROUND 3: Final Synthesis          │")
        print("└──────────────────────────────────────────┘")

        print("  Generating final event plan...")
        plan = self.synthesize_plan(
            user_query=user_input,
            parsed=parsed,
            weather_initial=weather_data,
            venues_initial=venue_data,
            conflicts=conflicts,
            weather_adjusted=weather_adjusted,
            venues_adjusted=venues_adjusted,
        )

        print("\n" + "=" * 50)
        print(plan)
        print("=" * 50)

    async def cleanup(self):
        await self.a2a.close()


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

async def main():
    coordinator = Coordinator()

    print("╔══════════════════════════════════════════╗")
    print("║     A2A Event Planner — Coordinator      ║")
    print("╠══════════════════════════════════════════╣")
    print("║  Agents expected:                        ║")
    print("║    Weather Agent → localhost:5001         ║")
    print("║    Venue Agent   → localhost:5002         ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("Type your event planning query, or 'quit' to exit.")
    print("Example: Plan an outdoor wedding in Austin TX this Saturday")
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
                print("Make sure both agents are running (weather on :5001, venue on :5002)")
    finally:
        await coordinator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
