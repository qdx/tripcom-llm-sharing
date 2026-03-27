# Harness Engineering — Research Reference

> **Purpose:** Reference document for the Trip.com LLM Knowledge Sharing presentation: *"From Prompt Engineering to Harness Engineering"*
> **Last Updated:** 2026-03-27

---

## Table of Contents

1. [The Three Eras: Prompt → Context → Harness](#1-the-three-eras-prompt--context--harness)
2. [Origin: Mitchell Hashimoto](#2-origin-mitchell-hashimoto)
3. [OpenAI's Codex Harness](#3-openais-codex-harness)
4. [Anthropic's Agent Harness](#4-anthropics-agent-harness)
5. [Community Consensus & Key Articles](#5-community-consensus--key-articles)
6. [What a Harness Actually Contains](#6-what-a-harness-actually-contains)
7. [Key Quotes for Slides](#7-key-quotes-for-slides)
8. [Sources](#8-sources)

---

## 1. The Three Eras: Prompt → Context → Harness

| Era | Time Period | Focus | Key Question |
|-----|-----------|-------|-------------|
| **Prompt Engineering** | 2023–2024 | Crafting the right question | *"What do I ask the model?"* |
| **Context Engineering** | Mid-2025 | Filling the context window with the right information | *"What do I send the model so it can answer well?"* |
| **Harness Engineering** | Feb 2026– | Building the entire system around the model | *"How do I make agents work reliably, continuously, at scale?"* |

### Prompt Engineering (2023–2024)
- One question, one answer. Optimize the instruction.
- Techniques: few-shot, chain-of-thought, system prompts, role-play
- Limitation: You're still doing all the work — the model just generates text

### Context Engineering (Mid-2025)
- **Popularized by Andrej Karpathy** (June 2025): *"+1 for 'context engineering' over 'prompt engineering'. Context engineering is the delicate art and science of filling the context window with just the right information for the next step."*
- Also championed by Shopify CEO Tobi Lütke and formalized by LangChain and Anthropic
- Focus: RAG, MCP, memory systems, structured context injection
- Key insight: The prompt is just a small part — what matters is **everything** you put in the context window (retrieved docs, tool outputs, conversation history, system instructions)
- Limitation: Still focused on a single model call or conversation. Doesn't address what happens when agents run for hours across sessions.

### Harness Engineering (Feb 2026–)
- **The model is already smart. The harness gives it hands, eyes, and a workspace.**
- Goes beyond the context window to encompass: tool execution, state persistence, feedback loops, guardrails, observability, orchestration, memory across sessions
- **Harness engineering includes context engineering** (it's a superset), but operates at the system level, not the interaction level

> **The cleanest distinction:**
> - **Prompt engineering** = what to ask
> - **Context engineering** = what to send
> - **Harness engineering** = everything around the model that makes it work reliably in production

---

## 2. Origin: Mitchell Hashimoto

### The Blog Post
- **Title:** "My AI Adoption Journey"
- **Author:** Mitchell Hashimoto (co-founder of HashiCorp, creator of Terraform, Vagrant, Ghostty)
- **Date:** February 5, 2026
- **URL:** https://mitchellh.com/writing/my-ai-adoption-journey

### The Six-Stage AI Adoption Journey

| Step | Name | Description |
|------|------|-------------|
| 1 | Drop the Chatbot | Use agents, not chat interfaces |
| 2 | Reproduce Your Own Work | Force yourself to recreate manual work with agents to build intuition |
| 3 | End-of-Day Agents | Kick off agents in the last 30 min of day for overnight progress |
| 4 | Outsource the Slam Dunks | Let agents handle tasks you're confident they'll do well; work on other things |
| 5 | **Engineer the Harness** | When an agent makes a mistake, engineer a solution so it never makes that mistake again |
| 6 | Always Have an Agent Running | Goal: have an agent doing useful work at all times |

### Step 5: The Key Quote

> *"I don't know if there is a broad industry-accepted term for this yet, but I've grown to calling this 'harness engineering.' It is the idea that anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again."*
> — Mitchell Hashimoto, Feb 5, 2026

### Two Forms of Harness Engineering (per Hashimoto)
1. **Better implicit prompting (AGENTS.md):** For simple issues — wrong commands, wrong APIs — update the AGENTS.md. Each line corresponds to a past agent failure that's now prevented. [Example from Ghostty](https://github.com/ghostty-org/ghostty/blob/ca07f8c3f775fe437d46722db80a755c2b6e6399/src/inspector/AGENTS.md).
2. **Actual programmed tools:** Scripts to take screenshots, run filtered tests, validate output. Paired with AGENTS.md updates to teach the agent about them.

### Impact
- The term spread rapidly. Days after Hashimoto's post, OpenAI published "Harness engineering: leveraging Codex in an agent-first world"
- Martin Fowler / Thoughtworks picked it up within 2 weeks
- By March 2026, "harness engineering" became the dominant framing in the industry

---

## 3. OpenAI's Codex Harness

### The Blog Post
- **Title:** "Harness engineering: leveraging Codex in an agent-first world"
- **Author:** Ryan Lopopolo, Member of the Technical Staff
- **Date:** February 11, 2026
- **URL:** https://openai.com/index/harness-engineering/

### The Experiment
- **A team of 3 engineers built an internal product over 5 months with 0 lines of manually written code**
- ~1,500 PRs merged → ~1 million lines of code
- Throughput: 3.5 PRs per engineer per day (increased as team grew to 7)
- Product has real daily internal users and external alpha testers
- Estimated 10x faster than hand-writing the code

### Core Philosophy: "Humans steer. Agents execute."

### Key Harness Components

#### A. Context Engineering — Progressive Disclosure
- **Rejected the "one big AGENTS.md" approach** — it failed because:
  - Context is a scarce resource; a giant file crowds out the actual task
  - Too much guidance becomes non-guidance ("when everything is important, nothing is")
  - Monolithic docs rot instantly
  - Hard to verify for freshness/accuracy
- **Solution: AGENTS.md as table of contents (~100 lines)** pointing to deeper sources:
  - `docs/design-docs/` — architecture, core beliefs, design history
  - `docs/exec-plans/active/` — current execution plans
  - `docs/references/` — framework docs, llms.txt files
  - `ARCHITECTURE.md`, `QUALITY_SCORE.md`, `SECURITY.md`, etc.
- **"Give Codex a map, not a 1,000-page instruction manual"**

#### B. Architectural Constraints — Enforced Mechanically
- Rigid layered architecture: Types → Config → Repo → Service → Runtime → UI
- Custom linters enforce dependency directions and architectural rules
- **Brilliant detail: Linter error messages double as remediation instructions** — when an agent violates a constraint, the error tells it how to fix it. The tooling teaches the agent.
- Structural tests validate module boundaries
- "Boring" tech preferred — composable, stable APIs, well-represented in training data

#### C. Agent Legibility
- Codebase optimized for **agent readability**, not human readability
- **"What Codex can't see doesn't exist"** — Slack discussions, Google Docs, tacit knowledge must be encoded into the repo
- Repository-local, versioned artifacts are the only source of truth

#### D. Tool Access & Observability
- Chrome DevTools Protocol wired into agent runtime → agents can navigate, screenshot, and test UI
- Full observability stack per worktree: logs (LogQL), metrics (PromQL), traces (TraceQL)
- Single Codex runs work on tasks for up to 6 hours (often overnight)

#### E. Garbage Collection
- Periodic "doc-gardening" agents scan for stale documentation
- Background Codex tasks scan for architectural deviations
- "Golden principles" encoded in repo → continuous enforcement
- **"Technical debt is like a high-interest loan — better to pay it down continuously"**

#### F. Levels of Autonomy Achieved
Single prompt → agent can: validate codebase → reproduce bug → record video → implement fix → validate fix → record resolution video → open PR → respond to feedback → detect/fix build failures → escalate only when judgment required → merge

### Also Published
- **"Unlocking the Codex harness: how we built the App Server"** (Feb 4, 2026) — technical deep dive on execution environment
- **"Unrolling the Codex agent loop"** — details on the agent execution loop

---

## 4. Anthropic's Agent Harness

### The Research Post
- **Title:** "Effective harnesses for long-running agents"
- **URL:** https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- **Focus:** Making the Claude Agent SDK work across many context windows for hours/days

### The Core Problem
> *"Imagine a software project staffed by engineers working in shifts, where each new engineer arrives with no memory of what happened on the previous shift."*

### Two Key Failure Modes
1. **One-shotting:** Agent tries to do too much at once → runs out of context mid-implementation → next session inherits a mess
2. **Premature victory:** After some features are built, a later agent session looks around, sees progress, and declares the job done

### The Two-Part Solution
1. **Initializer Agent:** First session uses a specialized prompt to:
   - Set up `init.sh` script
   - Generate comprehensive `feature_list.json` with 200+ features (all initially marked `passes: false`)
   - Create `claude-progress.txt` for session handoff
   - Make initial git commit

2. **Coding Agent:** Every subsequent session:
   - Reads progress files and git log
   - Runs basic health check on dev server
   - Picks ONE feature to work on
   - Tests end-to-end with browser automation (Puppeteer MCP)
   - Commits with descriptive messages
   - Updates progress file
   - Leaves environment in clean, mergeable state

### Key Insights
- **JSON > Markdown** for structured tracking — models are less likely to inappropriately edit JSON
- **Incremental progress is critical** — one feature at a time
- **Browser automation dramatically improved testing** — catches bugs invisible from code alone
- Limitations remain: e.g., can't see browser-native alert modals through Puppeteer

### Anthropic's Claude Code as a Harness
- **Claude Code and the Claude Agent SDK** provide a harness with:
  - Built-in permission model
  - Hooks system
  - Support for long-running multi-session agents
  - Agent Skills system with progressive disclosure:
    - Level 1: YAML frontmatter only (name + description) loaded at startup
    - Level 2: Full SKILL.md body loaded when relevant
    - Level 3: Additional reference files accessed for deep work
  - MCP tool integration with code execution approach (replacing tool definitions with executable code)

---

## 5. Community Consensus & Key Articles

### Martin Fowler / Birgitta Böckeler (Thoughtworks)
- **URL:** https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html
- **Date:** Feb 17, 2026
- Harness = three categories:
  1. **Context engineering** — knowledge base + dynamic context
  2. **Architectural constraints** — linters + structural tests (deterministic)
  3. **Garbage collection** — periodic agents fighting entropy
- Insight: *"Increasing trust and reliability required constraining the solution space, not expanding it"*
- Prediction: Harnesses may become the new "service templates" for organizations
- Open question: How to retrofit harnesses on brownfield (legacy) codebases?

### The Ignorance.ai Synthesis
- **URL:** https://www.ignorance.ai/p/the-emerging-harness-engineering
- **Date:** Feb 22, 2026
- Best synthesis of Hashimoto + OpenAI + Anthropic + Stripe patterns
- Key framing: Engineer's job splits into two halves:
  1. **Building the environment** (harness engineering)
  2. **Managing the work** (directing agent tasks, reviews, parallelization)
- Convergence across solo (Steinberger/OpenClaw), small team (OpenAI), and enterprise (Stripe) scales

### LangChain: "The Anatomy of an Agent Harness"
- **URL:** https://blog.langchain.com/the-anatomy-of-an-agent-harness/
- Counterpoint: *"Models will get better at planning, self-verification, and long-horizon coherence natively, requiring less context injection. Harnesses should matter less over time."*
- But: for now, harness engineering is essential

### Phil Schmid (Hugging Face)
- **URL:** https://www.philschmid.de/agent-harness-2026
- Distinction: **Framework** provides the inference loop; **Harness** provides prompt presets, tool call handling, lifecycle hooks, planning, filesystem access, sub-agent management
- Harness = opinionated layer above the framework

### Other Key References
| Article | Author | Key Contribution |
|---------|--------|-----------------|
| [What Is an Agent Harness?](https://www.firecrawl.dev/blog/what-is-an-agent-harness) | Firecrawl | Detailed breakdown of harness components |
| [What Is Harness Engineering? Complete Guide](https://www.nxcode.io/resources/news/what-is-harness-engineering-complete-guide-2026) | NxCode | Comprehensive overview with Anthropic/OpenAI comparison |
| [The Rise of AI Harness Engineering](https://cobusgreyling.medium.com/the-rise-of-ai-harness-engineering-5f5220de393e) | Cobus Greyling | Practitioner perspective |
| [Harness Engineering: The Missing Layer](https://www.louisbouchard.ai/harness-engineering/) | Louis Bouchard | Accessible explanation |
| [From Prompts → Context → Harness Engineering](https://softmaxdata.com/blog/from-prompt-engineering-to-harness-engineering-the-three-eras-of-building-with-ai/) | SoftmaxData | The three-era evolution narrative |

---

## 6. What a Harness Actually Contains

### The Components of a Production Harness

```
┌─────────────────────────────────────────────────────┐
│                   AGENT HARNESS                      │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Context     │  │    Tools     │  │  Memory &  │ │
│  │  Engineering  │  │  & Execution │  │   State    │ │
│  │              │  │              │  │            │ │
│  │ • AGENTS.md  │  │ • File R/W   │  │ • Progress │ │
│  │ • Docs/specs │  │ • Shell/CLI  │  │   files    │ │
│  │ • RAG/search │  │ • Browser    │  │ • Git log  │ │
│  │ • Prog.      │  │ • HTTP/API   │  │ • Feature  │ │
│  │   disclosure │  │ • MCP tools  │  │   lists    │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Guardrails  │  │ Observability│  │ Orchestr.  │ │
│  │  & Safety    │  │  & Feedback  │  │ & Lifecycle│ │
│  │              │  │              │  │            │ │
│  │ • Linters    │  │ • Logs       │  │ • Session  │ │
│  │ • Arch tests │  │ • Metrics    │  │   mgmt     │ │
│  │ • Permissions│  │ • Traces     │  │ • Compactio│ │
│  │ • Sandboxing │  │ • Screenshots│  │ • Multi-   │ │
│  │ • Rate limits│  │ • CI results │  │   agent    │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                      │
│               ┌──────────────────┐                   │
│               │   LLM / Model    │                   │
│               │  (the "engine")  │                   │
│               └──────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

### Concrete Examples

| Component | What It Looks Like In Practice |
|-----------|-------------------------------|
| **AGENTS.md / CLAUDE.md** | Short map (~100 lines) pointing to deeper docs; updated every time agent makes a mistake |
| **Execution plans** | Versioned markdown files with task breakdowns, progress tracking, decision logs |
| **Feature lists** | JSON files with pass/fail status per feature, updated by coding agents |
| **Custom linters** | Enforce architectural boundaries; error messages include fix instructions |
| **Browser automation** | Puppeteer/CDP for UI testing — agent can screenshot, navigate, verify visually |
| **Observability** | Per-worktree log/metric/trace stack agents can query (LogQL, PromQL) |
| **Git discipline** | Descriptive commits, clean state after each session, revert capability |
| **Doc gardening** | Background agents that scan for stale docs and open cleanup PRs |
| **Progressive disclosure** | Skills/docs loaded in layers — headers first, full content on demand |
| **Sandboxed environments** | Isolated devboxes (Stripe), per-worktree instances (OpenAI) |

---

## 7. Key Quotes for Slides

### Mitchell Hashimoto (Origin)
> *"Anytime you find an agent makes a mistake, you take the time to engineer a solution such that the agent never makes that mistake again."*

### OpenAI (Scale)
> *"Humans steer. Agents execute."*

> *"Give Codex a map, not a 1,000-page instruction manual."*

> *"Our most difficult challenges now center on designing environments, feedback loops, and control systems."*

> *"When everything is 'important,' nothing is."*

### Anthropic (Reliability)
> *"Imagine a software project staffed by engineers working in shifts, where each new engineer arrives with no memory of what happened on the previous shift."*

### Community
> *"The model is already smart. The harness gives it hands, eyes, and a workspace."*
> — shareAI-lab/learn-claude-code

> *"Increasing trust and reliability required constraining the solution space, not expanding it."*
> — Birgitta Böckeler, Thoughtworks

> *"2025 was agents. 2026 is agent harnesses."*
> — Aakash Gupta

> *"Claude Code is not a better model. It's a better harness wrapped around the same model."*
> — Aakash Gupta

### The Evolution
> *"Prompt engineering is what to ask. Context engineering is what to send. Harness engineering is everything around the model."*
> — Louis Bouchard (paraphrased)

---

## 8. Sources

### Primary Sources
| Source | Date | URL |
|--------|------|-----|
| Mitchell Hashimoto — "My AI Adoption Journey" | Feb 5, 2026 | https://mitchellh.com/writing/my-ai-adoption-journey |
| OpenAI — "Harness engineering: leveraging Codex in an agent-first world" | Feb 11, 2026 | https://openai.com/index/harness-engineering/ |
| Anthropic — "Effective harnesses for long-running agents" | 2026 | https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents |
| OpenAI — "Unlocking the Codex harness: how we built the App Server" | Feb 4, 2026 | https://openai.com/index/unlocking-the-codex-harness/ |

### Key Analysis
| Source | Date | URL |
|--------|------|-----|
| Martin Fowler / Birgitta Böckeler — "Harness Engineering" | Feb 17, 2026 | https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html |
| Ignorance.ai — "The Emerging Harness Engineering Playbook" | Feb 22, 2026 | https://www.ignorance.ai/p/the-emerging-harness-engineering |
| LangChain — "The Anatomy of an Agent Harness" | Mar 2026 | https://blog.langchain.com/the-anatomy-of-an-agent-harness/ |
| Phil Schmid — "The importance of Agent Harness in 2026" | Jan 5, 2026 | https://www.philschmid.de/agent-harness-2026 |

### Community Articles
| Source | URL |
|--------|-----|
| Louis Bouchard — "Harness Engineering: The Missing Layer Behind AI Agents" | https://www.louisbouchard.ai/harness-engineering/ |
| Cobus Greyling — "The Rise of AI Harness Engineering" | https://cobusgreyling.medium.com/the-rise-of-ai-harness-engineering-5f5220de393e |
| Firecrawl — "What Is an Agent Harness?" | https://www.firecrawl.dev/blog/what-is-an-agent-harness |
| NxCode — "What Is Harness Engineering? Complete Guide 2026" | https://www.nxcode.io/resources/news/what-is-harness-engineering-complete-guide-2026 |
| SoftmaxData — "From Prompt Engineering to Harness Engineering" | https://softmaxdata.com/blog/from-prompt-engineering-to-harness-engineering-the-three-eras-of-building-with-ai/ |
| Epsilla — "The Third Evolution" | https://www.epsilla.com/blogs/harness-engineering-evolution-prompt-context-autonomous-agents |
| MadPlay — "Beyond Prompts and Context" | https://madplay.github.io/en/post/harness-engineering |
| Aakash Gupta — "2025 Was Agents. 2026 Is Agent Harnesses." | https://aakashgupta.medium.com/2025-was-agents-2026-is-agent-harnesses-heres-why-that-changes-everything-073e9877655e |
| HumanLayer — "Skill Issue: Harness Engineering for Coding Agents" | https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents |
| Generative Inc — "Harness Engineering: The Most Important Skill" | https://www.generative.inc/harness-engineering-the-most-important-skill-in-the-agentic-ai-era |

### Context Engineering Background
| Source | URL |
|--------|-----|
| Andrej Karpathy tweet (June 2025) | https://x.com/karpathy/status/1937902205765607626 |

---

## Appendix: Timeline

```
Jun 2025   Karpathy popularizes "context engineering"
           → Focus on filling context window with right information

Jan 5, 2026  Phil Schmid writes about "agent harness" importance

Feb 4, 2026  OpenAI publishes "Unlocking the Codex harness"

Feb 5, 2026  ★ Mitchell Hashimoto publishes "My AI Adoption Journey"
             → Coins "harness engineering" (Step 5: Engineer the Harness)

Feb 11, 2026 ★ OpenAI publishes "Harness engineering: leveraging Codex 
             in an agent-first world" (Ryan Lopopolo)
             → 0 manually-written code, 1M lines, 1500 PRs, 3 engineers

Feb 17, 2026 Birgitta Böckeler (Thoughtworks) writes analysis on 
             martinfowler.com

Feb 22, 2026 Ignorance.ai synthesizes the emerging playbook
             → Maps Hashimoto + OpenAI + Stripe + Anthropic convergence

Mar 2026     Term goes mainstream: LangChain, Firecrawl, NxCode, 
             dozens of blog posts, Reddit threads, newsletters
             → Community consensus: harness engineering is THE 
             discipline for 2026
```
