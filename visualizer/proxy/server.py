"""Context Window Visualizer — Proxy Server.

Proxies OpenAI-compatible chat completions, captures the messages array,
and pushes real-time updates to the visualizer frontend via WebSocket.
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from token_counter import count_message_tokens

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MAX_TOKENS = 128_000  # GPT-4 context window

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
conversation_messages: list[dict] = []
current_mode: str = "simple"
connected_clients: set[WebSocket] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def enrich_message(msg: dict) -> dict:
    """Add tokenCount to a message dict."""
    enriched = dict(msg)
    enriched["tokenCount"] = count_message_tokens(msg)
    return enriched


def build_state_payload() -> dict:
    total_tokens = sum(m.get("tokenCount", 0) for m in conversation_messages)
    return {
        "type": "state_update",
        "messages": conversation_messages,
        "tokenCount": total_tokens,
        "maxTokens": DEFAULT_MAX_TOKENS,
        "mode": current_mode,
        "metadata": {
            "model": "gpt-4",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


async def broadcast(payload: dict):
    data = json.dumps(payload)
    disconnected = set()
    for ws in connected_clients:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.add(ws)
    connected_clients.difference_update(disconnected)


async def broadcast_state():
    await broadcast(build_state_payload())


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        base_url=LLM_BASE_URL, timeout=120.0
    )
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Context Window Visualizer Proxy", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    # Send current state on connect
    try:
        await ws.send_text(json.dumps(build_state_payload()))
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(ws)


# ---------------------------------------------------------------------------
# Proxy: POST /v1/chat/completions
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    global conversation_messages

    body = await request.json()
    messages = body.get("messages", [])

    # Capture incoming messages
    conversation_messages = [enrich_message(m) for m in messages]
    await broadcast_state()

    # Forward to actual LLM
    headers = {}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    headers["Content-Type"] = "application/json"

    client: httpx.AsyncClient = request.app.state.http_client
    t0 = time.time()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json=body,
            headers=headers,
        )
        resp_json = resp.json()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    latency_ms = int((time.time() - t0) * 1000)

    # Capture assistant response
    choices = resp_json.get("choices", [])
    if choices:
        assistant_msg = choices[0].get("message", {})
        conversation_messages.append(enrich_message(assistant_msg))

    # Update metadata
    await broadcast_state()
    return JSONResponse(resp_json, status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Manual injection: POST /inject
# ---------------------------------------------------------------------------
@app.post("/inject")
async def inject_messages(request: Request):
    global conversation_messages, current_mode

    body = await request.json()

    # Support both single message and array
    if "messages" in body:
        new_msgs = body["messages"]
    elif "role" in body:
        new_msgs = [body]
    else:
        return JSONResponse({"error": "Provide 'messages' array or a single message with 'role'"}, status_code=400)

    if body.get("replace", False):
        conversation_messages = []

    if body.get("mode"):
        current_mode = body["mode"]

    for msg in new_msgs:
        conversation_messages.append(enrich_message(msg))

    await broadcast_state()
    return {"ok": True, "messageCount": len(conversation_messages)}


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------
@app.get("/state")
async def get_state():
    return build_state_payload()


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------
@app.post("/reset")
async def reset():
    global conversation_messages, current_mode
    conversation_messages = []
    current_mode = "simple"
    await broadcast_state()
    return {"ok": True}


# ---------------------------------------------------------------------------
# POST /mode
# ---------------------------------------------------------------------------
@app.post("/mode")
async def set_mode(request: Request):
    global current_mode
    body = await request.json()
    current_mode = body.get("mode", "simple")
    await broadcast_state()
    return {"ok": True, "mode": current_mode}


# ---------------------------------------------------------------------------
# GET /pca-data — serve pre-computed PCA trajectory data
# ---------------------------------------------------------------------------
PCA_DATA_PATH = Path(__file__).parent.parent / "outputs" / "pca_trajectories.json"

@app.get("/pca-data")
async def get_pca_data():
    if not PCA_DATA_PATH.exists():
        return JSONResponse(
            {"error": "PCA data not found. Run scripts/generate_pca_data.py first."},
            status_code=404,
        )
    with open(PCA_DATA_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
