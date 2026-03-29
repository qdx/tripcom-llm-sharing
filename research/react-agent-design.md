# Demo 2: ReAct Agent Workflow Design

**Task:** 2.2 Design ReAct Agent workflow — Trip.com relevant scenario

## Chosen Scenario: Travel Itinerary Planning Assistant

**Why this resonates with Trip.com engineers:**
- Directly maps to their product domain (travel booking)
- Shows real tool calling patterns they'd build in production
- Demonstrates context window growth in a familiar business context
- Multi-step reasoning with observable Thought→Action→Observation cycles

## Scenario

**User prompt:** "I'm planning a 3-day trip to Tokyo next month. I need flights from Shanghai, a hotel near Shinjuku, and restaurant recommendations. Budget is ¥15,000/day."

## ReAct Loop Design

### Round 1: Flight Search
```
💭 Thought: The user wants to travel from Shanghai to Tokyo. Let me search for available flights first, as this determines the travel dates and remaining budget.

🔧 Action: search_flights(origin="PVG", destination="NRT", date_range="2026-04-15 to 2026-04-20", max_price=5000)

📋 Observation: Found 3 flights:
  - MU523: PVG→NRT, Apr 16 08:30, ¥2,890 (economy)
  - CA921: PVG→HND, Apr 16 10:15, ¥3,200 (economy)  
  - 9C8515: PVG→NRT, Apr 16 06:00, ¥1,650 (economy, Spring Airlines)
```

### Round 2: Hotel Search
```
💭 Thought: MU523 looks like a good balance of price and timing. Now I need a hotel near Shinjuku within budget. With ¥15,000/day budget minus meals, hotel should be under ¥8,000/night.

🔧 Action: search_hotels(location="Shinjuku, Tokyo", checkin="2026-04-16", checkout="2026-04-19", max_price_per_night=8000)

📋 Observation: Found 5 hotels:
  - Hotel Sunroute Plaza Shinjuku: ¥7,200/night, 4.2★, 3min walk from station
  - Tokyu Stay Shinjuku: ¥6,800/night, 4.0★, has kitchenette
  - Shinjuku Washington Hotel: ¥5,900/night, 3.8★, business hotel
```

### Round 3: Restaurant Search
```
💭 Thought: Hotel Sunroute is ideal — close to station, good reviews, within budget. For restaurants, I should find options near Shinjuku that match the remaining daily budget (~¥5,000-6,000 for meals).

🔧 Action: search_restaurants(location="Shinjuku, Tokyo", cuisine="mixed", price_range="moderate", rating_min=4.0)

📋 Observation: Top recommendations:
  - Fuunji (つけ麺 風雲児): tsukemen, ¥1,200, 4.6★
  - Omoide Yokocho: yakitori alley, ¥2,000-3,000, 4.3★
  - Tsunahachi (つな八): tempura since 1924, ¥3,500, 4.5★
```

### Round 4: Budget Summary
```
💭 Thought: Let me compile the complete itinerary with budget breakdown to ensure it fits the ¥15,000/day constraint.

🔧 Action: calculate_budget(flights=2890, hotel_per_night=7200, meals_per_day=5000, days=3)

📋 Observation: 
  Total: ¥2,890 (flight) + ¥21,600 (hotel 3 nights) + ¥15,000 (meals 3 days) = ¥39,490
  Per day average: ¥13,163 ✅ Under ¥15,000/day budget
```

### Final Response
The agent synthesizes all tool results into a structured itinerary recommendation.

## Tools Definition (for Dify)

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_flights` | Search available flights | origin, destination, date_range, max_price |
| `search_hotels` | Search hotels by location | location, checkin, checkout, max_price_per_night |
| `search_restaurants` | Find restaurants | location, cuisine, price_range, rating_min |
| `calculate_budget` | Budget breakdown | flights, hotel_per_night, meals_per_day, days |
| `get_weather` | Weather forecast | city, dates |

## Implementation in Dify

### Option A: Mock API Tools (Recommended for Demo)
- Create a simple FastAPI server with endpoints that return pre-crafted realistic data
- **Advantages:** Reliable, fast responses, no API key dependencies, perfect for live demo
- Endpoints mirror the tools above, return hardcoded but realistic results
- Can add random latency (500ms-2s) to simulate real API calls

### Option B: Real APIs (Risky for Live Demo)
- Amadeus API for flights (free sandbox)
- Booking.com or Google Places for hotels
- **Disadvantage:** Network dependency, rate limits, unpredictable responses

**Recommendation:** Option A with mock APIs. The point of the demo is showing the ReAct loop + context window growth, not real travel data.

## Dify Workflow Structure

```
[User Input]
    │
    ▼
[LLM Node — System Prompt with ReAct instructions]
    │
    ├──▶ [Tool: search_flights]
    │         │
    │         ▼
    ├──▶ [Tool: search_hotels]  
    │         │
    │         ▼
    ├──▶ [Tool: search_restaurants]
    │         │
    │         ▼
    ├──▶ [Tool: calculate_budget]
    │
    ▼
[Final Response — Synthesized Itinerary]
```

## System Prompt for the Agent

```
You are a Trip.com travel planning assistant. Help users plan trips by searching for flights, hotels, and restaurants within their budget.

For each step:
1. Think about what information you need next
2. Use the appropriate tool to search
3. Analyze the results
4. Decide the next action

Always verify the total cost stays within the user's daily budget.
Show your reasoning at each step — the audience should understand why you're making each choice.
```

## Context Window Growth (Key Visual)

This is what the visualizer shows during Demo 2:

| Step | Messages in Context | Approx Tokens | % of 128K |
|------|-------------------|---------------|-----------|
| Start | system + user | ~500 | 0.4% |
| After Round 1 | + assistant(thought+tool_call) + tool_response | ~1,200 | 0.9% |
| After Round 2 | + assistant + tool_response | ~2,000 | 1.6% |
| After Round 3 | + assistant + tool_response | ~2,800 | 2.2% |
| After Round 4 | + assistant + tool_response | ~3,400 | 2.7% |
| Final response | + assistant (full itinerary) | ~4,200 | 3.3% |

**Key insight for audience:** Even a simple 4-tool conversation fills thousands of tokens. With GPT-4's 128K context, this is fine — but older models (4K, 8K) would struggle. This is why context engineering matters.

## Presentation Talking Points

1. **"Notice how the agent THINKS before acting"** — each round starts with explicit reasoning
2. **"Watch the context window grow"** — visualizer shows real-time token accumulation  
3. **"Every tool call adds ~400-600 tokens"** — tool responses are verbose
4. **"The agent maintains state across rounds"** — references flight cost when budgeting hotel
5. **"This is what your product could do"** — Trip.com engineers can picture this in their app
6. **"Now imagine 10 tools, 20 rounds"** — context management becomes critical

## Fallback Plan

If Dify or LLM is slow/down during demo:
1. Pre-recorded video of the conversation (screen capture with visualizer)
2. Use the `/inject` endpoint on the proxy to replay pre-scripted messages
3. Step through manually, explaining each round with static slides
