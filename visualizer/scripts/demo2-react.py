"""Demo 2: ReAct agent loop — tool calling conversation step by step."""

import httpx
import time

BASE = "http://localhost:8080"


def inject(msg: dict, label: str, delay: float = 2.0):
    httpx.post(f"{BASE}/inject", json=msg)
    print(f"  -> {label}")
    time.sleep(delay)


def main():
    httpx.post(f"{BASE}/reset")
    time.sleep(0.5)
    httpx.post(f"{BASE}/mode", json={"mode": "tool"})
    time.sleep(0.3)

    print("=== Demo 2: ReAct Agent Loop ===\n")

    # System message
    inject({
        "role": "system",
        "content": """You are a Trip.com travel agent with access to the following tools:

- search_flights(origin, destination, date, passengers): Search available flights
- search_hotels(city, checkin, checkout, guests): Search available hotels
- check_weather(city, date): Get weather forecast
- calculate_budget(items): Calculate total trip budget

Use the ReAct pattern: Think about what to do, take an action, observe the result, and repeat until you have enough information to give a final answer.""",
    }, "System prompt with tools", 1.5)

    # User query
    inject({
        "role": "user",
        "content": "Plan a 3-day trip to Bangkok from Shanghai starting May 1st for 2 people. I want flights, hotel, and weather info. Budget under ¥8000 per person.",
    }, "User query", 2.0)

    # ReAct Loop - Round 1: Think + search flights
    print("\n--- Round 1: Search Flights ---")
    inject({
        "role": "assistant",
        "content": "Let me help plan your Bangkok trip. First, I'll search for flights from Shanghai to Bangkok.",
        "tool_calls": [{
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "search_flights",
                "arguments": '{"origin": "SHA", "destination": "BKK", "date": "2025-05-01", "passengers": 2}'
            }
        }]
    }, "Assistant: Thought + Action (search_flights)", 1.5)

    inject({
        "role": "tool",
        "tool_call_id": "call_001",
        "name": "search_flights",
        "content": '{"flights": [{"airline": "Spring Airlines", "flight": "9C8951", "depart": "08:00", "arrive": "12:30", "price": 1280, "currency": "CNY"}, {"airline": "Thai Airways", "flight": "TG665", "depart": "10:30", "arrive": "15:00", "price": 2100, "currency": "CNY"}, {"airline": "China Eastern", "flight": "MU547", "depart": "14:00", "arrive": "18:30", "price": 1650, "currency": "CNY"}]}'
    }, "Tool response: flight results", 2.0)

    # Round 2: Search hotels
    print("\n--- Round 2: Search Hotels ---")
    inject({
        "role": "assistant",
        "content": "Good, I found several flight options. Now let me search for hotels in Bangkok.",
        "tool_calls": [{
            "id": "call_002",
            "type": "function",
            "function": {
                "name": "search_hotels",
                "arguments": '{"city": "Bangkok", "checkin": "2025-05-01", "checkout": "2025-05-04", "guests": 2}'
            }
        }]
    }, "Assistant: Thought + Action (search_hotels)", 1.5)

    inject({
        "role": "tool",
        "tool_call_id": "call_002",
        "name": "search_hotels",
        "content": '{"hotels": [{"name": "Ibis Bangkok Riverside", "stars": 3, "price_per_night": 320, "currency": "CNY"}, {"name": "Novotel Bangkok Sukhumvit", "stars": 4, "price_per_night": 580, "currency": "CNY"}, {"name": "Centara Grand", "stars": 5, "price_per_night": 950, "currency": "CNY"}]}'
    }, "Tool response: hotel results", 2.0)

    # Round 3: Check weather
    print("\n--- Round 3: Check Weather ---")
    inject({
        "role": "assistant",
        "content": "Great hotel options. Let me also check the weather forecast for Bangkok in early May.",
        "tool_calls": [{
            "id": "call_003",
            "type": "function",
            "function": {
                "name": "check_weather",
                "arguments": '{"city": "Bangkok", "date": "2025-05-01"}'
            }
        }]
    }, "Assistant: Thought + Action (check_weather)", 1.5)

    inject({
        "role": "tool",
        "tool_call_id": "call_003",
        "name": "check_weather",
        "content": '{"forecast": "Hot and humid. High 35°C, Low 27°C. 40% chance of afternoon thunderstorms. UV index: Very High. Recommend sunscreen and umbrella."}'
    }, "Tool response: weather forecast", 2.0)

    # Round 4: Calculate budget
    print("\n--- Round 4: Calculate Budget ---")
    inject({
        "role": "assistant",
        "content": "Now let me calculate the total budget with the best value options.",
        "tool_calls": [{
            "id": "call_004",
            "type": "function",
            "function": {
                "name": "calculate_budget",
                "arguments": '{"items": [{"name": "Spring Airlines round-trip", "cost": 2560, "per_person": true}, {"name": "Novotel 3 nights", "cost": 1740, "per_person": false}, {"name": "Daily meals estimate", "cost": 600, "per_person": true}, {"name": "Activities & transport", "cost": 400, "per_person": true}]}'
            }
        }]
    }, "Assistant: Thought + Action (calculate_budget)", 1.5)

    inject({
        "role": "tool",
        "tool_call_id": "call_004",
        "name": "calculate_budget",
        "content": '{"total_per_person": 4430, "total_for_group": 8860, "breakdown": {"flights": 2560, "hotel": 870, "meals": 600, "activities": 400}, "under_budget": true}'
    }, "Tool response: budget calculation", 2.0)

    # Final answer
    print("\n--- Final Answer ---")
    inject({
        "role": "assistant",
        "content": """Here's your optimized Bangkok trip plan:

**Flights (¥2,560/person round-trip):**
- Outbound: Spring Airlines 9C8951, May 1, 08:00→12:30
- Return: Spring Airlines 9C8952, May 4, 13:00→18:00

**Hotel (¥870/person for 3 nights):**
- Novotel Bangkok Sukhumvit (4-star), ¥580/night
- Great location near BTS Skytrain, pool, breakfast included

**Weather:** Hot (35°C), bring sunscreen and umbrella for afternoon showers.

**Budget Breakdown (per person):**
| Item | Cost |
|------|------|
| Flights | ¥2,560 |
| Hotel (3 nights) | ¥870 |
| Meals | ¥600 |
| Activities | ¥400 |
| **Total** | **¥4,430** |

Well under your ¥8,000 budget! You have ¥3,570 of buffer per person for shopping, upgrades, or extra activities. Shall I book the flights and hotel?"""
    }, "Final answer with complete plan", 1.0)

    print("\nDemo 2 complete! Check the visualizer at http://localhost:3000")
    print("Notice how the context window grew with each ReAct cycle.")


if __name__ == "__main__":
    main()
