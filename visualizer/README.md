# Context Window Visualizer

Real-time visualization of the LLM context window as a growing memory stack. Used for the Trip.com LLM sharing session demos.

## Quick Start

### 1. Start the Proxy Server

```bash
cd visualizer/proxy
pip3 install -r requirements.txt
python3 server.py
```

Proxy runs on `http://localhost:8080`.

### 2. Start the Frontend

```bash
cd visualizer/frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`.

### 3. Run a Demo Script

In a third terminal:

```bash
cd visualizer/scripts

# Demo 1: Simple conversation
python3 demo1-simple.py

# Demo 2: ReAct agent with tool calling
python3 demo2-react.py

# Demo 3: Full harness engineering (massive system prompt)
python3 demo3-harness.py
```

## Architecture

```
Browser (localhost:3000)  <--WebSocket-->  Proxy (localhost:8080)  -->  LLM API
```

- **Proxy** intercepts chat completions, captures messages, broadcasts via WebSocket
- **Frontend** renders messages as colored blocks proportional to token count
- **Demo scripts** inject pre-scripted conversations via `/inject` endpoint

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible proxy |
| `/ws` | WebSocket | Real-time state updates |
| `/inject` | POST | Manual message injection |
| `/state` | GET | Current conversation snapshot |
| `/reset` | POST | Clear conversation |
| `/mode` | POST | Switch visualization mode |

## Modes

- **Simple** — Basic system/user/assistant conversation
- **Tool Calling** — ReAct agent loop with tool calls and responses
- **Harness** — Full harness engineering with expandable system prompt

## Using as a Proxy

Point your LLM client at the proxy to visualize real API calls:

```bash
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=your-key-here
# Now any OpenAI-compatible client will route through the visualizer
```

Configure the upstream LLM endpoint:

```bash
export LLM_BASE_URL=https://api.openai.com  # default
```
