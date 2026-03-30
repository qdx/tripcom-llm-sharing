"""Demo 3: Full harness engineering — complete agent context with memory/tools/sub-agents.

This demonstrates the REAL complexity of what goes into a single LLM API call
when a harness like OpenClaw orchestrates the conversation. Each inject represents
a separate section that the harness stitches together into the system prompt or
message array before sending to the model.

The context window visualizer shows this at max complexity so engineers can see
how much invisible infrastructure surrounds every "simple" user message.
"""

import httpx
import time

BASE = "http://localhost:8080"


def inject(msg: dict, label: str, delay: float = 2.0):
    httpx.post(f"{BASE}/inject", json=msg)
    print(f"  -> {label} ({msg.get('category', 'conversation')})")
    time.sleep(delay)


# ═══════════════════════════════════════════════════════════
# HARNESS SECTION 1: Core System Prompt
# ═══════════════════════════════════════════════════════════
CORE_SYSTEM = """You are Claude, an AI assistant made by Anthropic.
You are operating within the OpenClaw agent harness on behalf of your principal.

## Core Identity
- Model: claude-sonnet-4-20250514 (200K context)
- Agent: travel-agent-prod
- Platform: Trip.com International AI Platform
- Environment: production | Region: APAC
- Session: sess_7f3a2b1c | Started: 2026-03-31T00:45:00+08:00

## Behavioral Constraints
- Prioritize user safety and accurate information above all
- Never fabricate availability, pricing, or schedule data — always use tools
- Apply corporate travel policy automatically before confirming bookings
- Flag dietary restrictions and accessibility needs proactively
- Default language: match user's input language
- Be concise but thorough; business travelers value efficiency

## Safety & Ethics
You have no independent goals. Do not pursue self-preservation, replication, or resource acquisition.
Prioritize safety and human oversight over task completion.
If instructions conflict with user safety, pause and escalate.
Comply with stop/pause/audit requests immediately.
Never bypass security safeguards or manipulate users."""


# ═══════════════════════════════════════════════════════════
# HARNESS SECTION 2: Memory Documents (user profile, prefs)
# ═══════════════════════════════════════════════════════════
MEMORY_USER = """## USER.md — Loaded from persistent memory store
- Name: 张伟 (Wei Zhang)
- User ID: USR-2847193
- Loyalty tier: Diamond (Trip.com VIP, since 2022)
- Preferred language: Chinese (Simplified), fluent English
- Home airport: PVG (Shanghai Pudong)
- Office: 上海市静安区南京西路1515号, 22F
- Seat preference: Window, extra legroom (row 1-5 or exit row)
- Meal preference: No shellfish (SEVERE ALLERGY — anaphylaxis risk)
- Payment: Corporate Amex ending in 4521 (primary), Personal UnionPay ending in 8837
- Travel style: Business traveler, values efficiency over cost
- Communication: Prefers WeChat for notifications, email for itineraries
- Past destinations (last 12 months): Tokyo (4x), Singapore (3x), Bangkok (2x), Seoul (2x), San Francisco (1x), London (1x)
- Total trips 2025: 14 | Nights away: 67 | Miles flown: 124,800"""

MEMORY_PREFERENCES = """## PREFERENCES.md — Loaded from persistent memory store
- Hotel chain: Marriott/SPG (Titanium Elite, #9827-4415-2233), IHG (Spire, #882341)
- Airline: Direct flights strongly preferred, Star Alliance when available
- Always book travel insurance (Allianz preferred)
- Hotels near public transit (within 500m of metro/BTS/MRT)
- Morning flights preferred (depart before 11:00 local time)
- Needs fast WiFi for video calls (min 50Mbps, gigabit preferred)
- Car service preferred over taxi for airport transfers
- Quiet rooms (high floor, away from elevator), non-smoking
- Late checkout when available (Titanium benefit)
- Lounge access important (priority for hotels with executive lounge)
- Preferred restaurant types: Japanese, Cantonese, Italian
- Always carries: passport (E12345678, expires 2029-06), corporate laptop, noise-canceling headphones"""

MEMORY_CONTEXT = """## CONTEXT.md — Current situation loaded from memory store
- Current trip in progress: Tokyo business trip, April 2-5, 2026
  - Hotel: Tokyo Marriott (confirmation #MR-7728841), Executive Suite
  - Flight out: MU523 PVG→NRT Apr 2 08:30, Business Class
  - Flight return: MU524 NRT→PVG Apr 5 19:00, Business Class
  - Purpose: Q1 review with Tokyo engineering team
- Upcoming: Singapore conference, April 15-18 (CONFIRMED)
  - Hotel: Marina Bay Sands (confirmation #MBS-992341)
  - Flight: SQ831 PVG→SIN Apr 15 07:15, Business Class
- Company travel policy: Business class for flights >4 hours, Premium Economy for shorter
- Expense cap: ¥50,000 per trip without VP approval
- Corporate account: Trip.com Enterprise — Acme Corp (ID: ENT-88421)
- Department budget remaining (Q2): ¥380,000
- Pending expense reports: 2 (Tokyo Q4 trip, Seoul January trip)"""


# ═══════════════════════════════════════════════════════════
# HARNESS SECTION 3: Tool Definitions
# ═══════════════════════════════════════════════════════════
TOOLS_DEFINITION = """## Available Tools

You have access to the following tools. Call them using the standard function calling interface.

### Flight Tools
- search_flights(origin: str, destination: str, departure_date: str, return_date?: str, passengers: int, cabin_class: str, alliance_preference?: str, direct_only?: bool, flexible_dates?: bool) → FlightSearchResult[]
- book_flight(flight_id: str, passengers: PassengerInfo[], seat_preference: str, meal_preference: str, add_insurance: bool, corporate_account?: str) → BookingConfirmation
- check_flight_status(flight_number: str, date: str) → FlightStatus
- get_seat_map(flight_id: str) → SeatMap

### Hotel Tools
- search_hotels(city: str, checkin: str, checkout: str, guests: int, chain_preference?: str, min_stars?: int, near_transit?: bool, wifi_speed?: str, max_price?: int) → HotelSearchResult[]
- book_hotel(hotel_id: str, room_type: str, checkin: str, checkout: str, guest_name: str, loyalty_number?: str, special_requests?: str) → BookingConfirmation
- check_hotel_availability(hotel_id: str, dates: DateRange) → AvailabilityResult
- get_hotel_reviews(hotel_id: str, sort_by?: str, limit?: int) → Review[]

### Transport Tools
- book_car_service(pickup: str, dropoff: str, datetime: str, vehicle_type: str, notes?: str) → CarBooking
- search_trains(origin: str, destination: str, date: str, class?: str) → TrainResult[]

### Information Tools
- check_weather(city: str, date: str) → WeatherForecast
- get_visa_requirements(passport_country: str, destination: str) → VisaInfo
- get_currency_rates(from: str, to: str, amount?: float) → ExchangeRate
- get_destination_tips(city: str, category?: str) → TravelTips

### Business Tools
- calculate_budget(items: BudgetItem[], corporate_policy?: str) → BudgetResult
- check_policy_compliance(booking: BookingDetails) → PolicyCheck
- get_loyalty_status(program: str, member_id: str) → LoyaltyStatus
- submit_expense_report(trip_id: str, items: ExpenseItem[]) → ExpenseReport

### Communication Tools
- send_itinerary(booking_ids: str[], recipient: str, format: str) → SendResult
- notify_assistant(message: str, channel: str) → NotifyResult
- create_calendar_event(title: str, start: str, end: str, location?: str) → CalendarEvent

### Sub-Agent Delegation
- spawn_subagent(task: str, context: str, tools?: str[]) → SubAgentResult
  Spawns an isolated sub-agent for parallel research tasks. Results are injected back into context."""


# ═══════════════════════════════════════════════════════════
# HARNESS SECTION 4: Skill Definitions
# ═══════════════════════════════════════════════════════════
SKILLS_DEFINITION = """## Active Skills (loaded by harness based on context)

### skill:travel-policy-checker (v2.1)
Automatically validates all bookings against Acme Corp travel policy before confirmation.
Checks: cabin class rules, expense caps, preferred vendors, advance booking requirements (7-day minimum for international), weekend travel surcharge policy, duty of care compliance.
Triggers: pre-booking validation, post-search filtering, expense estimate.

### skill:loyalty-optimizer (v1.8)
Suggests ways to maximize loyalty points and elite status across programs.
Monitors: Marriott Bonvoy, IHG Rewards, Star Alliance miles, Trip.com VIP points.
Auto-actions: check upgrade availability, calculate points vs cash value, suggest status-qualifying options, alert on expiring points (Marriott points expire in 180 days: 42,000 points at risk).

### skill:itinerary-conflict-detector (v1.3)
Monitors for scheduling conflicts with existing bookings and calendar.
Checks: overlapping dates, impossible connection times (<2h international, <1h domestic), visa processing time, jet lag recovery recommendations, meeting conflicts from synced calendar.

### skill:weather-travel-advisor (v1.0)
Provides weather-aware travel recommendations.
Auto-injects: packing suggestions, weather warnings, seasonal event info.
Active alerts: Typhoon season (APAC Jun-Nov), monsoon patterns, extreme heat advisories.

### skill:expense-auto-categorizer (v0.9)
Automatically categorizes expenses for corporate reporting.
Maps bookings to GL codes, generates pre-filled expense reports, flags policy exceptions."""


# ═══════════════════════════════════════════════════════════
# HARNESS SECTION 5: Runtime Context
# ═══════════════════════════════════════════════════════════
RUNTIME_CONTEXT = """## Runtime Context (injected by harness at request time)

### Current Time
2026-03-31T00:50:00+08:00 (Asia/Shanghai)
Day: Tuesday | Week: 14 | Quarter: Q1→Q2 transition

### Session Metadata
- Session ID: sess_7f3a2b1c
- Channel: Trip.com App (iOS 15.4, iPhone 14 Pro)
- IP region: Shanghai, CN
- Request latency budget: 30s
- Rate limits: 60 req/min, 100K tokens/min

### Active Alerts
- ⚠️ Tokyo Narita Terminal 2 construction — allow extra transit time
- ⚠️ Singapore entry: ArriveSG card required (digital), submit 3 days before travel
- 📌 Marriott Bonvoy points expiring: 42,000 points on May 15, 2026

### Conversation History Summary (previous turns compressed)
[Turn 1] User asked about extending Tokyo trip
[Turn 2] Assistant confirmed availability, suggested dates
[Turn 3] User approved extension to April 7
[This is turn 4 — continuation of trip modification flow]"""


def main():
    httpx.post(f"{BASE}/reset")
    time.sleep(0.5)
    httpx.post(f"{BASE}/mode", json={"mode": "harness"})
    time.sleep(0.5)

    print("=" * 60)
    print("  Demo 3: Full Harness Engineering — Max Complexity")
    print("  Context Window Visualizer at Full Power")
    print("=" * 60)

    # ─── Phase 1: Harness assembles the system prompt ───────
    print("\n📋 Phase 1: Harness assembles context (invisible to user)\n")

    inject({
        "role": "system",
        "content": CORE_SYSTEM,
        "label": "CORE SYSTEM PROMPT",
        "section": "Identity, constraints, safety rules",
        "category": "core",
    }, "Core system prompt", 2.5)

    inject({
        "role": "system",
        "content": MEMORY_USER,
        "label": "MEMORY: USER.md",
        "section": "User profile & history",
        "category": "memory",
    }, "Memory: user profile", 2.0)

    inject({
        "role": "system",
        "content": MEMORY_PREFERENCES,
        "label": "MEMORY: PREFERENCES.md",
        "section": "Behavioral preferences & patterns",
        "category": "memory",
    }, "Memory: preferences", 2.0)

    inject({
        "role": "system",
        "content": MEMORY_CONTEXT,
        "label": "MEMORY: CONTEXT.md",
        "section": "Current situation & upcoming trips",
        "category": "memory",
    }, "Memory: current context", 2.0)

    inject({
        "role": "system",
        "content": TOOLS_DEFINITION,
        "label": "TOOL DEFINITIONS",
        "section": "25 tools across 6 categories",
        "category": "tools",
    }, "Tool definitions (25 tools)", 2.0)

    inject({
        "role": "system",
        "content": SKILLS_DEFINITION,
        "label": "ACTIVE SKILLS",
        "section": "5 skills auto-loaded by harness",
        "category": "skills",
    }, "Skill definitions (5 active skills)", 2.0)

    inject({
        "role": "system",
        "content": RUNTIME_CONTEXT,
        "label": "RUNTIME CONTEXT",
        "section": "Time, session, alerts, history summary",
        "category": "runtime",
    }, "Runtime context (time, alerts, session)", 2.0)

    # ─── Phase 2: The actual conversation ────────────────────
    print("\n💬 Phase 2: User sends a simple message\n")

    inject({
        "role": "user",
        "content": "帮我确认一下东京行程延期到4月7号的酒店预订，另外查下新加坡那边需要什么入境材料。顺便看看有没有什么积分可以用。",
        "category": "conversation",
    }, "User message (3 requests in one sentence)", 2.5)

    # ─── Phase 3: Agent thinks and calls tools in parallel ───
    print("\n🔧 Phase 3: Agent orchestrates parallel tool calls\n")

    inject({
        "role": "assistant",
        "content": "张伟先生，我来同时处理这三件事。先查酒店延期可用性、新加坡入境要求，以及您的积分情况。",
        "tool_calls": [
            {
                "id": "call_ext01",
                "type": "function",
                "function": {
                    "name": "check_hotel_availability",
                    "arguments": '{"hotel_id": "marriott-tokyo-MR7728841", "dates": {"checkin": "2026-04-05", "checkout": "2026-04-07"}}'
                }
            },
            {
                "id": "call_ext02",
                "type": "function",
                "function": {
                    "name": "get_visa_requirements",
                    "arguments": '{"passport_country": "CN", "destination": "SG"}'
                }
            },
            {
                "id": "call_ext03",
                "type": "function",
                "function": {
                    "name": "get_loyalty_status",
                    "arguments": '{"program": "marriott_bonvoy", "member_id": "9827-4415-2233"}'
                }
            }
        ],
        "category": "conversation",
    }, "Assistant: 3 parallel tool calls", 2.0)

    # Tool results come back
    inject({
        "role": "tool",
        "tool_call_id": "call_ext01",
        "name": "check_hotel_availability",
        "content": '{"available": true, "hotel": "Tokyo Marriott Hotel", "room_type": "Executive Suite", "dates": "Apr 5-7", "price_per_night": 3200, "currency": "CNY", "titanium_upgrade": "Presidential Suite available (complimentary upgrade)", "late_checkout": "Guaranteed until 16:00", "executive_lounge": true, "wifi": "Gigabit fiber", "notes": "Same room can be extended, no need to change rooms"}',
        "category": "conversation",
    }, "Tool result: hotel availability ✓", 1.5)

    inject({
        "role": "tool",
        "tool_call_id": "call_ext02",
        "name": "get_visa_requirements",
        "content": '{"visa_required": false, "entry_type": "Visa-free (30 days)", "requirements": ["Valid passport (6+ months validity)", "Return/onward ticket", "ArriveSG digital declaration (submit 3 days before)", "Proof of accommodation", "COVID: No requirements since 2023"], "processing_time": "ArriveSG: instant online", "notes": "⚠️ ArriveSG submission opens 3 days before arrival. Recommended: submit April 12.", "url": "https://www.ica.gov.sg/enter-transit/entering-singapore"}',
        "category": "conversation",
    }, "Tool result: visa requirements ✓", 1.5)

    inject({
        "role": "tool",
        "tool_call_id": "call_ext03",
        "name": "get_loyalty_status",
        "content": '{"program": "Marriott Bonvoy", "tier": "Titanium Elite", "points_balance": 320400, "points_expiring": {"amount": 42000, "date": "2026-05-15"}, "elite_nights_ytd": 18, "nights_to_ambassador": 82, "suite_night_awards": 3, "upgrade_available": true, "point_value_estimate": "¥0.065/point", "redeemable_free_nights": [{"hotel": "Tokyo Marriott", "points_needed": 45000, "cash_value": "¥3,200"}]}',
        "category": "conversation",
    }, "Tool result: loyalty status ✓", 2.0)

    # ─── Phase 4: Sub-agent delegation ───────────────────────
    print("\n🤖 Phase 4: Agent spawns sub-agent for complex task\n")

    inject({
        "role": "assistant",
        "content": "酒店可以延期，入境材料已查到。我发现您有 42,000 积分即将过期，让我的助手帮您分析最优的积分使用方案。",
        "tool_calls": [
            {
                "id": "call_sub01",
                "type": "function",
                "function": {
                    "name": "spawn_subagent",
                    "arguments": '{"task": "Analyze optimal redemption strategy for 42,000 expiring Marriott Bonvoy points (expire May 15). User has Tokyo Marriott stay Apr 2-7 and Singapore MBS Apr 15-18. Consider: free night redemptions, point transfers, room upgrades via SNA, dining credits. Return top 3 options ranked by value.", "context": "User is Titanium Elite with 320,400 total points. Has 3 Suite Night Awards. Tokyo Marriott free night = 45,000 pts. Point value ~¥0.065/pt.", "tools": ["get_loyalty_status", "search_hotels", "calculate_budget"]}'
                }
            },
            {
                "id": "call_ext04",
                "type": "function",
                "function": {
                    "name": "check_policy_compliance",
                    "arguments": '{"booking": {"type": "hotel_extension", "hotel": "Tokyo Marriott", "dates": "Apr 5-7", "cost": 6400, "original_trip_cost": 32000, "new_total": 38400, "corporate_account": "ENT-88421"}}'
                }
            }
        ],
        "category": "conversation",
    }, "Assistant: spawn sub-agent + policy check", 2.5)

    inject({
        "role": "tool",
        "tool_call_id": "call_ext04",
        "name": "check_policy_compliance",
        "content": '{"compliant": true, "total_trip_cost": 38400, "policy_limit": 50000, "remaining_budget": 11600, "flags": [], "auto_approved": true, "policy_version": "Acme Corp Travel Policy v3.2 (2026-01)"}',
        "category": "conversation",
    }, "Tool result: policy compliance ✓", 1.5)

    # Sub-agent returns its result
    inject({
        "role": "tool",
        "tool_call_id": "call_sub01",
        "name": "spawn_subagent",
        "content": '{"status": "completed", "execution_time_ms": 4200, "tokens_used": 3847, "result": {"analysis": "Analyzed 42,000 expiring points across 6 redemption strategies", "top_options": [{"rank": 1, "strategy": "Transfer 42,000 pts to United MileagePlus (1:1.1 current promo)", "value": "¥4,200 equivalent", "notes": "Best value. Promo ends Apr 10. Can use for PVG-SIN segment upgrade."}, {"rank": 2, "strategy": "Apply to Tokyo stay (Apr 5-7): 42,000 pts = ¥2,730 room credit", "value": "¥2,730", "notes": "Simple, immediate. Reduces out-of-pocket by 42%."}, {"rank": 3, "strategy": "Marriott Moments experience in Tokyo", "value": "¥2,100-3,500 depending on experience", "notes": "Unique experiences, but limited Apr availability."}], "recommendation": "Option 1 (airline transfer) provides best value at ¥4,200. However, if user prefers simplicity, Option 2 is a solid ¥2,730 savings on the current trip."}}',
        "label": "SUB-AGENT RESULT",
        "section": "Points optimization analysis (3,847 tokens used in sub-context)",
        "category": "subagent",
    }, "Sub-agent result: points analysis complete", 2.5)

    # ─── Phase 5: Final synthesized response ─────────────────
    print("\n✅ Phase 5: Agent synthesizes everything into final response\n")

    inject({
        "role": "assistant",
        "content": """张伟先生，三件事都处理好了：

## 1️⃣ 东京酒店延期 ✅ 已确认可延
- **Tokyo Marriott** Executive Suite → 延至 4月7日退房
- 新增2晚: ¥6,400 (¥3,200/晚)
- 🌟 **Titanium 福利**: 可免费升级至总统套房（视当日可用）
- 🕐 延迟退房保证至 16:00
- **企业差旅政策**: ✅ 合规（行程总费用 ¥38,400 < ¥50,000 上限）

## 2️⃣ 新加坡入境材料
- 🛂 中国护照免签入境（30天）
- 📋 必需材料:
  - 有效护照（您的 E12345678，2029年到期 ✓）
  - ArriveSG 电子入境卡 — **请在4月12日提交**
  - 回程机票确认（已有 SQ831 ✓）
  - 酒店确认（MBS 已有 ✓）
- ⚠️ 提醒: ArriveSG 必须在到达前3天内在线提交

## 3️⃣ 积分使用建议 (42,000 分5月15日过期)
| 方案 | 价值 | 推荐度 |
|------|------|--------|
| 转至 United 里程 (1:1.1 促销) | ¥4,200 | ⭐⭐⭐ 最优 |
| 抵扣东京房费 | ¥2,730 | ⭐⭐ 简单直接 |
| Marriott Moments 体验 | ¥2,100-3,500 | ⭐ 供参考 |

**我的建议**: 转 United 里程最划算（促销4月10日截止），但如果您嫌麻烦，直接抵扣东京房费省 ¥2,730 也很好。

需要我现在就预订酒店延期并设置积分转换提醒吗？""",
        "category": "conversation",
    }, "Final synthesized response (personalized, policy-checked, multi-source)", 1.0)

    # ─── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Demo 3 Complete!")
    print("=" * 60)
    print("""
What the audience just saw:

📦 HARNESS ASSEMBLED (before user's message):
  • Core system prompt — identity, safety, constraints
  • 3 memory documents — user profile, preferences, current context
  • 25 tool definitions — flights, hotels, transport, business, comms
  • 5 active skills — policy checker, loyalty optimizer, conflict detector...
  • Runtime context — time, session, alerts, compressed history

💬 CONVERSATION (what the user sees):
  • 1 user message → 3 parallel tool calls → sub-agent delegation → synthesized answer

🧮 CONTEXT WINDOW REALITY:
  • ~90% of tokens are INVISIBLE infrastructure (harness-injected)
  • ~10% is the actual conversation
  • The user typed ONE sentence; the model processed THOUSANDS of tokens of context
  • This is Harness Engineering — the context window IS the product

Check the visualizer at http://localhost:3000
Look at the segmented token bar to see the breakdown!
""")


if __name__ == "__main__":
    main()
