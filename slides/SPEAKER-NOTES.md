# Speaker Notes — Slide-by-Slide Guide

## Slide 1: Title
- Welcome everyone, quick intro
- "Today we're covering the evolution of how engineers work with LLMs"
- "Not a research talk — a practical guide with live demos"

## Slide 2: Opening Hook — "What do you do next?"
- Ask the audience directly. Pause for 3 seconds.
- Click to reveal the three eras — 2023/2024/2026 boxes
- "In 2023, we'd rewrite the prompt. In 2024, add RAG or tools. In 2026, we engineer the whole environment."
- This sets up the entire narrative arc

## Slide 3: The Evolution Table
- Walk through each row briefly
- Emphasize: "The human's job got MORE complex, not less"
- The quote sets the theme

## Slide 4: Agenda
- Quick walkthrough, don't dwell
- "We have 3 live demos, each showing the same tool — the context window visualizer — in increasingly complex modes"
- Set expectations: interactive, questions welcome anytime

## Slide 5: Timeline Section Divider
- "Let's start with how we got here. 7 years, 22 milestones."

## Slide 6: Era 1 — Prompt Engineering
- GPT-2: "this was only 7 years ago, and we thought 1.5B parameters was impressive"
- ChatGPT: "the iPhone moment — 100M users in 2 months"
- CoT: "the peak of prompt engineering — discovering that asking the model to think step by step actually changes the computation"
- Fragment quote: pause, let it sink in

## Slide 7: Era 2 — Context Engineering
- This is the era most of us are currently in
- Karpathy quote — he coined "context engineering" June 2025
- Key message: "the prompt is just a small part — what matters is EVERYTHING in the context window"
- Mention Dify/Coze — relevant for Chinese market, Trip.com engineers

## Slide 8: Era 3 — Harness Engineering
- Hashimoto: "creator of Terraform — when HE names something, people listen"
- OpenAI: "0 lines of hand-written code, 1M generated — that's not a toy demo"
- The shift quote is the key message — click fragment

## Slide 9: Context Window Unifier
- "One concept ties all three eras together"
- Build suspense, then reveal: "The Context Window"
- "Every interaction = filling a fixed-size memory buffer. Let's see what's inside."
- This transitions naturally to Demo 1

## Slide 10: Demo 1 Section Divider
- Quick setup: "First demo — simplest possible mode"

## Slide 11: Context Window — Simple Mode
- Walk through the visual: system (blue), user (green), assistant (amber)
- Point out the token meter: "187 out of 128,000 — almost empty"
- Fragment: CoT comparison — "adding 'think step by step' literally makes the context bigger"
- This is the setup for the live demo

## Slide 12: Live Demo 1
- Switch to localhost:3000
- Run demo1-simple.py script
- Show: basic conversation filling the context
- Then show: CoT version — longer context, better reasoning
- ~5 minutes live interaction
- Switch back to slides

## Slide 13: Transition to Demo 2
- "Simple prompts are nice, but what if the model needs to DO things?"
- Travel scenario resonates with Trip.com engineers
- Set up the Context Engineering era connection

## Slide 14: Demo 2 Section Divider
- ReAct = Reasoning + Acting
- "Watch the context window grow with each tool call"

## Slide 15: ReAct Pattern Breakdown
- Walk through each colored block in the stack
- Emphasize: each round-trip ADDS to the context
- Key insight fragment: "long agent loops can fill the context window"
- Trip.com-relevant scenario makes it concrete

## Slide 16: Live Demo 2
- Switch to localhost:3000, Mode: Tool Calling
- Run demo2-react.py
- Watch messages appear in real-time: thought → tool_call → tool_response → thought
- Point out the token meter growing
- "After 4 rounds, most of the context is tool responses, not thinking"
- ~7 minutes live interaction
- Switch back to slides

## Slide 17: Transition to Demo 3
- "Tool calling agents work, BUT..."
- Walk through the bullet points: memory, identity, guardrails
- Quote: "The model is already smart. The harness gives it hands, eyes, and a workspace."
- This is the big reveal coming

## Slide 18: Demo 3 Section Divider
- "Now we see what a REAL production agent looks like inside"

## Slide 19: Full Harness — What Gets Injected
- THIS IS THE WOW SLIDE
- Walk through every block in the system prompt stack
- "All of this is injected BEFORE the user says anything"
- Token meter: 18,000 tokens just for the system prompt
- "The user says 6 tokens. The model sees 18,000."
- Fragment: "This is harness engineering"

## Slide 20: Harness Architecture Grid
- Quick walkthrough of 6 components
- Relate to Trip.com: "Your booking APIs = ready-made tools, your domain docs = RAG content"
- OpenAI quote fragment: "Humans steer. Agents execute."

## Slide 21: OpenAI Codex Numbers
- Let the numbers speak: 0 hand-written, 1M generated, 1,500 PRs
- Walk through the HOW bullets
- "AGENTS.md as table of contents, not a manual" — this is counterintuitive
- "Linter error messages that teach" — brilliant detail

## Slide 22: Live Demo 3
- Switch to localhost:3000, Mode: Harness
- Run demo3-harness.py
- Show the massive system prompt expanding
- Click to expand sections: SOUL.md, USER.md, skills, tools
- "Look at the token meter — the system prompt alone takes 9% of a 200K context"
- ~8 minutes live interaction
- Switch back to slides

## Slide 23: Takeaways Section Divider
- "Let's bring it home"

## Slide 24: Five Takeaways
- Click through each fragment one at a time
- #1: Context window is the key abstraction — "if you remember one thing..."
- #2: Tools vs guardrails — "most teams get tools right but skip guardrails"
- #3: Harness = DevOps for AI — "same rigor, new domain"
- #4: Human role elevated — "you need MORE engineering, not less"
- #5: Trip.com specific — "your domain is perfect for this"

## Slide 25: Practical Next Steps
- "Things you can literally do this week"
- Getting started: AGENTS.md is free and takes 30 minutes
- Intermediate: MCP tool wrapping is the highest-ROI activity
- Advanced: for teams ready to go all-in
- Reading list: 4 must-read articles

## Slide 26: Closing
- Read the three lines slowly, with emphasis on ask/send/build
- Quote fragment: let it land
- Final message: "The model is smart enough. Your job is to build the world it operates in."
- Pause before moving to Q&A

## Slide 27: Q&A
- Open the floor
- If no questions, have 2-3 prepared:
  1. "What's the biggest context engineering challenge in your team?"
  2. "Has anyone tried Claude Code or Codex on real work?"
  3. "What Trip.com APIs would make the best agent tools?"

## Slide 28: References (Appendix)
- Show briefly, "these are all in the repo"
- Don't read through them
