# Context Window Visualizer — Architecture Design

**Task:** 3.2 Design visualizer architecture — proxy + WebSocket + React frontend

## Overview

A real-time visualization tool that shows the **context window as a growing memory stack** during LLM conversations. Used across all 3 demos in the presentation.

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  LLM Client │───▶│  Proxy Server    │───▶│  LLM API        │
│  (Demo UI)  │    │  (Node.js/Python)│    │  (OpenAI, etc.) │
└─────────────┘    └────────┬─────────┘    └─────────────────┘
                            │
                     WebSocket push
                            │
                   ┌────────▼─────────┐
                   │  Visualizer UI   │
                   │  (React SPA)     │
                   └──────────────────┘
```

## Components

### 1. LLM Proxy Server (Python/FastAPI)
- **Purpose:** Intercept all LLM API calls, capture the full messages array, and push to visualizer
- **Why Python:** Consistent with the project stack (TransformerLens, Dify). FastAPI for async WebSocket support
- **Endpoints:**
  - `POST /v1/chat/completions` — OpenAI-compatible proxy endpoint
  - `WS /ws` — WebSocket for real-time message stream to visualizer
  - `POST /inject` — Manual message injection (for scripted demos)
  - `GET /state` — Current conversation state snapshot

**Proxy flow:**
1. Client sends chat completion request
2. Proxy captures the `messages` array
3. Pushes full message list via WebSocket to all connected visualizer clients
4. Forwards request to actual LLM API
5. Captures response, appends assistant message
6. Pushes updated messages array via WebSocket
7. Returns response to client

### 2. React Visualizer Frontend
- **Purpose:** Render the context window as a visual "memory stack"
- **Tech:** React + Vite + TailwindCSS
- **Layout:** Single-page vertical stack, messages rendered as colored blocks

**Visual design:**
- Each message = a colored block in a vertical stack (bottom-up or top-down)
- Role-based coloring:
  - 🔵 `system` — blue/indigo (always at top)
  - 🟢 `user` — green
  - 🟡 `assistant` — yellow/amber
  - 🔴 `tool_call` (within assistant) — red/rose
  - 🟠 `tool` (response) — orange
- Block height proportional to token count
- Live token counter showing total / max context window
- Progress bar showing context window utilization
- Smooth animation when new messages appear

### 3. Three Modes (matching 3 demos)

#### Mode 1: Simple (Demo 1 — CoT Visualization)
- Just system + user + assistant messages
- Shows context window for simple prompt vs CoT prompt
- Highlight: CoT prompt is physically longer → more "thinking space"
- Side-by-side: Direct prompt (short) vs CoT prompt (longer)

#### Mode 2: Tool Calling (Demo 2 — Dify ReAct Agent)
- system + user + assistant (with tool_calls) + tool responses
- Messages appear in real-time as the ReAct loop executes
- Shows Thought→Action→Observation cycle building up in context
- Context window grows visibly with each tool call round-trip

#### Mode 3: Full Harness (Demo 3 — Harness Engineering)
- system (massive, with memory/skills/tools injected) + user + assistant + tools
- System message is expanded to show injected components:
  - Core instructions
  - Memory docs (SOUL.md, USER.md, etc.)
  - Skill definitions
  - Tool schemas
- This is the "wow" moment — showing how much the harness contributes

## Data Protocol (WebSocket)

```typescript
interface VisualizerMessage {
  type: 'state_update' | 'token_count' | 'mode_change';
  messages: ChatMessage[];        // Full messages array
  tokenCount: number;             // Total tokens
  maxTokens: number;              // Model's context window size
  mode: 'simple' | 'tool' | 'harness';
  metadata?: {
    model: string;
    timestamp: string;
    latency_ms?: number;
  };
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: ToolCall[];        // For assistant messages
  tool_call_id?: string;          // For tool response messages
  name?: string;                  // Tool name
  tokenCount: number;             // Estimated tokens for this message
  // Visual metadata
  label?: string;                 // Display label (e.g., "SOUL.md", "Weather Tool")
  section?: string;               // For grouped display in harness mode
}
```

## Token Counting

Use `tiktoken` (Python) for accurate OpenAI token counts. For the visualizer, approximate with `len(text) / 4` as fallback for real-time display (correct on next full state update).

## File Structure

```
visualizer/
├── proxy/
│   ├── server.py           # FastAPI proxy + WebSocket server
│   ├── token_counter.py    # tiktoken wrapper
│   └── requirements.txt    # fastapi, uvicorn, tiktoken, httpx
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── ContextStack.tsx      # Main visualization
│   │   │   ├── MessageBlock.tsx      # Individual message block
│   │   │   ├── TokenCounter.tsx      # Token usage display
│   │   │   ├── ModeSelector.tsx      # Switch between 3 modes
│   │   │   └── SystemExpander.tsx    # Expand system msg sections
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts       # WebSocket connection
│   │   └── types.ts
│   ├── package.json
│   └── vite.config.ts
├── scripts/
│   ├── demo1-simple.py      # Pre-scripted Demo 1 messages
│   ├── demo2-react.py       # Pre-scripted Demo 2 messages
│   └── demo3-harness.py     # Pre-scripted Demo 3 messages
└── README.md
```

## Demo-Day Specifics

- **Runs on DChar's MacBook** — no cloud dependency for the visualizer itself
- Proxy binds to `localhost:8080`, visualizer at `localhost:3000`
- LLM API calls go through the proxy (set `OPENAI_BASE_URL=http://localhost:8080/v1`)
- Pre-scripted demo scripts as backup (inject messages without actual LLM calls)
- Dark theme for presentation visibility (matches PM Portal aesthetic)

## Dependencies

**Proxy:**
- Python 3.11+
- fastapi, uvicorn, httpx (async HTTP client)
- tiktoken
- websockets

**Frontend:**
- Node.js 18+
- React 18, Vite, TailwindCSS
- reconnecting-websocket (npm)

## Key Design Decisions

1. **Proxy pattern (not browser extension)** — more reliable for live demo, works with any LLM client
2. **WebSocket (not polling)** — real-time feel, messages appear as they happen
3. **Full state push (not diffs)** — simpler, more robust for demo, small payload sizes
4. **Pre-scripted fallback** — live demo can fail; inject scripts ensure the show goes on
5. **Dark theme** — projector-friendly, professional look
6. **Token count as primary metric** — engineers understand "token budget" intuitively
