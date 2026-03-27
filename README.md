# From Prompt Engineering to Harness Engineering

Knowledge sharing session for Trip.com software engineers (~30-40 people, 1 hour).

## Structure

Three demos connected by an AI Agent evolution timeline:

1. **Prompt Engineering → CoT Visualization** — GPT-2/Phi-2 hidden state trajectories showing how Chain-of-Thought affects model computation
2. **Context Engineering → Dify ReAct Agent** — Live demo of tool calling, context window management with a travel-relevant scenario
3. **Harness Engineering → Context Window Visualizer** — Real-time visualization of the full context window as an "memory stack", showing harness-injected context growing in real-time

## Unified Visual Language

A "Context Window as Memory Stack" view is used from start to finish — starting simple in Demo 1 and growing in complexity through Demo 3.

## Project Management

Tracked via [PM Portal](https://pm.rahcd.com) — project `tripcom-llm-sharing`, 3 epics, 15 tasks.

## Research

- `research/timeline.md` — 22 milestones from GPT-2 (2019) to Harness Engineering (2026)
- `research/harness-engineering.md` — Deep dive into harness engineering origin, definitions, key players
- `research/model-selection.md` — GPT-2 vs Phi-2 analysis for CoT demo

## Tech Stack (planned)

- **Demo 1:** TransformerLens + GPT-2/Phi-2 + Jupyter notebook
- **Demo 2:** Dify platform + custom tool integrations
- **Demo 3:** React + WebSocket + LLM API proxy
