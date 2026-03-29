"""Demo 1: Simple conversation — system + user + assistant messages."""

import httpx
import time

BASE = "http://localhost:8080"


def main():
    # Reset
    httpx.post(f"{BASE}/reset")
    time.sleep(0.5)

    # Set mode
    httpx.post(f"{BASE}/mode", json={"mode": "simple"})
    time.sleep(0.3)

    # System message
    httpx.post(f"{BASE}/inject", json={
        "role": "system",
        "content": "You are a helpful travel assistant for Trip.com. Help users find the best flights, hotels, and travel packages. Be concise and provide specific recommendations with prices when possible.",
    })
    print("[1/3] System message injected")
    time.sleep(1.5)

    # User message
    httpx.post(f"{BASE}/inject", json={
        "role": "user",
        "content": "I need to book a round-trip flight from Shanghai to Tokyo for 2 adults, departing April 15 and returning April 20. What are the best options?",
    })
    print("[2/3] User message injected")
    time.sleep(2.0)

    # Assistant response
    httpx.post(f"{BASE}/inject", json={
        "role": "assistant",
        "content": """Based on your search for Shanghai (PVG) to Tokyo (NRT), April 15-20, 2 adults, here are the top options:

**Best Value:**
- China Eastern MU523 — PVG→NRT, depart 08:30, arrive 12:00 (2h30m)
- Return: MU524 — NRT→PVG, depart 13:30, arrive 15:30 (2h)
- Price: ¥2,480 per person round-trip

**Fastest Schedule:**
- ANA NH922 — PVG→NRT, depart 10:15, arrive 13:45 (2h30m)
- Return: ANA NH921 — NRT→PVG, depart 17:00, arrive 19:00 (2h)
- Price: ¥3,150 per person round-trip

**Premium Option:**
- JAL JL086 — PVG→HND, depart 09:00, arrive 12:30 (2h30m)
- Return: JAL JL087 — HND→PVG, depart 14:00, arrive 16:30 (2h30m)
- Price: ¥3,680 per person round-trip (flies into Haneda, closer to city center)

All prices include taxes and fees. Would you like me to check hotel availability in Tokyo for these dates as well?""",
    })
    print("[3/3] Assistant response injected")
    print("\nDemo 1 complete! Check the visualizer at http://localhost:3000")


if __name__ == "__main__":
    main()
