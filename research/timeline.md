# AI Agent Evolution Timeline
## From Prompt Engineering to Harness Engineering (2019–2026)

> Narrative backbone for Trip.com LLM Knowledge Sharing presentation.
> All dates verified via web search (2026-03-27). Sources: Wikipedia, arXiv, official announcements.

---

## Era 1: Prompt Engineering (2019–2022)

*The era of learning to talk to models. Humans craft the perfect input; the model generates output. No tools, no memory, no autonomy.*

### 📅 February 2019 — GPT-2 Released
- **What:** OpenAI releases GPT-2 (1.5B parameters). Initially withheld the full model due to misuse concerns; full release November 5, 2019.
- **Paradigm:** Prompt Engineering
- **Why it matters:** First demonstration that scaling up language models produces surprisingly coherent text. Established the "too dangerous to release" narrative and kicked off the scaling era. The model had no instruction-following — you had to engineer prompts to coax useful output.

### 📅 May 2020 — GPT-3 Paper / June 2020 — API Access
- **What:** OpenAI publishes "Language Models are Few-Shot Learners" (arXiv: May 28, 2020). GPT-3 (175B parameters) demonstrates that large models can perform tasks via few-shot prompting — just show examples in the prompt. API beta access rolled out June 2020.
- **Paradigm:** Prompt Engineering
- **Why it matters:** Birth of "prompt engineering" as a discipline. For the first time, you could get a model to do tasks (translation, Q&A, code) without fine-tuning — just by writing better prompts. The 175B scale was 100× GPT-2.

### 📅 January 28, 2022 — Chain-of-Thought Prompting Paper
- **What:** Wei et al. publish "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (arXiv: 2201.11903). Shows that adding "Let's think step by step" dramatically improves reasoning.
- **Paradigm:** Prompt Engineering (peak)
- **Why it matters:** The pinnacle of prompt engineering — discovering that *how* you prompt matters as much as *what* you prompt. CoT became the standard technique and foreshadowed the reasoning models that would come later (o1, R1).

### 📅 March 4, 2022 — InstructGPT / RLHF Paper
- **What:** Ouyang et al. publish "Training Language Models to Follow Instructions with Human Feedback" (arXiv: 2203.02155). InstructGPT model deployed January 27, 2022. Introduces RLHF (Reinforcement Learning from Human Feedback).
- **Paradigm:** Prompt Engineering → transition
- **Why it matters:** Models that *follow instructions* rather than just *complete text*. This is the bridge from prompt engineering to conversational AI. RLHF became the standard alignment technique and made ChatGPT possible.

### 📅 November 30, 2022 — ChatGPT Launch
- **What:** OpenAI launches ChatGPT (based on GPT-3.5 with RLHF). Reaches 100M users in 2 months — fastest consumer app adoption in history.
- **Paradigm:** Prompt Engineering (mainstream moment)
- **Why it matters:** The "iPhone moment" for AI. Everyone suddenly understands prompt engineering because everyone is doing it. But the limitations are immediately obvious: no internet access, no tools, knowledge cutoff, hallucinations. This creates the demand for the next era.

---

## Era 2: Context Engineering (2023–2024)

*The era of giving models superpowers through external context. RAG, tools, plugins, agents, multi-agent systems. The prompt is no longer just text — it's an orchestrated context window.*

### 📅 March 14, 2023 — GPT-4 Released
- **What:** OpenAI releases GPT-4, a multimodal model with dramatically improved reasoning. Available via API and ChatGPT Plus.
- **Paradigm:** Context Engineering (foundation)
- **Why it matters:** GPT-4's capability jump made tool use and complex agents viable. It could reliably follow multi-step instructions, making the agent frameworks that followed actually work.

### 📅 March 23, 2023 — ChatGPT Plugins Announced
- **What:** OpenAI launches plugin support for ChatGPT, including web browsing and Code Interpreter. Third-party plugins let ChatGPT access external services.
- **Paradigm:** Context Engineering
- **Why it matters:** First mainstream attempt at "LLM + tools." Models could now browse the web, run code, and call APIs. The plugin ecosystem eventually failed (deprecated late 2023 in favor of GPTs), but it proved the concept that models need external tools.

### 📅 March 2023 — AutoGPT & BabyAGI
- **What:** AutoGPT (by Toran Bruce Richards) and BabyAGI (by Yohei Nakajima) release within days of each other. Both are autonomous agent loops: LLM → plan → execute → observe → repeat.
- **Paradigm:** Context Engineering (agent era begins)
- **Why it matters:** The "autonomous agent" meme explodes. AutoGPT becomes the fastest-growing GitHub repo ever. While these early agents were unreliable and often got stuck in loops, they established the agent paradigm: models that act, not just respond. Every agent framework that followed owes a debt to this moment.

### 📅 Mid-2023 — LangChain Explosion
- **What:** LangChain (launched October 2022 by Harrison Chase) becomes the fastest-growing open-source project on GitHub by June 2023. Raises $25M+ in funding. Provides chains, agents, and retrieval abstractions.
- **Paradigm:** Context Engineering
- **Why it matters:** LangChain became the default "glue" for LLM applications. It popularized patterns like chains (sequential LLM calls), agents (LLM + tools), and retrieval (RAG pipelines). Even if many teams later moved away from it, it defined the vocabulary of context engineering.

### 📅 June 13, 2023 — OpenAI Function Calling
- **What:** OpenAI introduces Function Calling in the API — models can output structured JSON to invoke functions defined by the developer.
- **Paradigm:** Context Engineering
- **Why it matters:** The standardization moment for tool use. Instead of hacking tool use into prompts, function calling gave models a native way to interact with external systems. This became the foundation for every agent framework and enterprise integration.

### 📅 October 2023 — AutoGen Released (Microsoft)
- **What:** Microsoft Research releases AutoGen, a framework for building multi-agent conversational systems. Agents can be composed into group chats with different roles.
- **Paradigm:** Context Engineering (multi-agent)
- **Why it matters:** Brought academic rigor to multi-agent systems. Introduced the concept of agents-as-conversation-participants, which influenced CrewAI and subsequent frameworks.

### 📅 November 6, 2023 — GPT-4 Turbo (128K Context) at DevDay
- **What:** OpenAI announces GPT-4 Turbo at its first DevDay conference. Features 128K context window (300+ pages), lower pricing, JSON mode, and reproducible outputs.
- **Paradigm:** Context Engineering
- **Why it matters:** 128K context meant you could fit entire codebases, long documents, or rich agent histories into a single prompt. This was a 16× increase over GPT-4's original 8K context. The "just put everything in the context window" approach became viable.

### 📅 November 2023 — CrewAI Released
- **What:** CrewAI launches as an open-source multi-agent framework with role-based agent design. Agents have specific roles, goals, and backstories.
- **Paradigm:** Context Engineering (multi-agent)
- **Why it matters:** Made multi-agent systems accessible to mainstream developers. The "crew" metaphor (assemble a team of specialists) became the dominant mental model for building agent applications in 2024.

### 📅 Early 2024 — RAG Becomes Standard Practice
- **What:** Retrieval-Augmented Generation (concept from Lewis et al., May 2020 paper) becomes the default architecture for enterprise LLM applications. Vector databases (Pinecone, Weaviate, Chroma) hit mainstream adoption.
- **Paradigm:** Context Engineering
- **Why it matters:** RAG solved the knowledge cutoff and hallucination problems by grounding LLM responses in retrieved documents. Every enterprise chatbot, search system, and knowledge base now uses some form of RAG. It's the defining pattern of context engineering.

### 📅 Early 2024 — Dify & Coze Platforms
- **What:** Dify (launched May 2023, 100K+ GitHub stars) and Coze (ByteDance, launched February 1, 2024) represent the "no-code/low-code AI platform" wave. Visual builders for AI workflows, RAG pipelines, and agent systems.
- **Paradigm:** Context Engineering (democratization)
- **Why it matters:** Context engineering moves from developer-only to business-user-accessible. These platforms abstract away LangChain-level complexity into drag-and-drop interfaces, making LLM application building accessible to non-engineers. Especially relevant in the Chinese market.

---

## Era 3: Harness Engineering (2024–2026)

*The era of engineering the environment, not the prompt. Models reason internally. Humans build harnesses — the scaffolding, guardrails, tool ecosystems, and feedback loops that let autonomous agents operate reliably. The human's job shifts from "telling the model what to think" to "building the world the model operates in."*

### 📅 September 12, 2024 — OpenAI o1 Preview (Reasoning Models)
- **What:** OpenAI releases o1-preview, the first "reasoning model" that uses internal chain-of-thought before responding. Full o1 release: December 5, 2024.
- **Paradigm:** Harness Engineering (beginning)
- **Why it matters:** The paradigm shift. o1 *internalizes* chain-of-thought — you no longer need to prompt "think step by step" because the model does it automatically. This means prompt engineering for reasoning becomes less important. The human's job shifts from crafting prompts to providing the right environment (context, tools, constraints) for the model to reason within.

### 📅 October 2024 — Claude Computer Use (Beta)
- **What:** Anthropic introduces Computer Use in public beta with Claude 3.5 Sonnet — the model can control a computer desktop by moving the cursor, clicking, and typing.
- **Paradigm:** Harness Engineering
- **Why it matters:** First general-purpose model that can interact with arbitrary computer interfaces. This is a qualitative shift from "model calls predefined tools" to "model uses any tool a human could use." The harness becomes the entire computer environment.

### 📅 November 25, 2024 — Anthropic MCP (Model Context Protocol)
- **What:** Anthropic introduces the Model Context Protocol (MCP) as an open standard for connecting AI models to external tools and data sources. Think "USB-C for AI."
- **Paradigm:** Harness Engineering
- **Why it matters:** MCP standardizes how agents connect to tools, replacing bespoke integrations. Instead of each agent framework inventing its own tool protocol, MCP provides a universal interface. Rapidly adopted across the industry (OpenAI, Google, Microsoft all support it by early 2025). This is harness infrastructure.

### 📅 January 20, 2025 — DeepSeek-R1
- **What:** Chinese AI lab DeepSeek releases R1, an open-source reasoning model that matches o1's performance. Uses reinforcement learning to develop reasoning capabilities. Becomes #1 on iOS App Store in the US.
- **Paradigm:** Harness Engineering
- **Why it matters:** Proves reasoning models aren't an OpenAI monoculture. Open-source + open-weights means anyone can build harnesses around reasoning models. Also demonstrates that reasoning can emerge from RL without massive RLHF datasets, suggesting a more scalable path.

### 📅 February 2025 — Anthropic Claude Code
- **What:** Anthropic releases Claude Code, an agentic coding tool that runs in the terminal. It can read, write, and execute code, manage git, and navigate codebases autonomously.
- **Paradigm:** Harness Engineering
- **Why it matters:** The terminal itself becomes a harness. Claude Code's behavior is configured via markdown files (CLAUDE.md, AGENTS.md) — this is pure harness engineering. You don't prompt the model; you configure the environment it operates in.

### 📅 April 9, 2025 — Google A2A Protocol (Agent-to-Agent)
- **What:** Google Cloud announces the Agent2Agent (A2A) protocol — an open standard for AI agents from different vendors to discover and communicate with each other. Contributed to Linux Foundation in June 2025.
- **Paradigm:** Harness Engineering
- **Why it matters:** If MCP is "how agents talk to tools," A2A is "how agents talk to each other." Together they define the communication layer of the harness. Multi-vendor, multi-agent orchestration becomes standardized rather than bespoke.

### 📅 May 2025 — OpenAI Codex (Cloud Agent)
- **What:** OpenAI launches Codex as a cloud-based coding agent integrated with ChatGPT. It executes software engineering tasks (writing features, fixing bugs, reviewing code) autonomously in sandboxed environments before returning results.
- **Paradigm:** Harness Engineering
- **Why it matters:** Codex operates in a full cloud environment — the harness *is* the sandbox. OpenAI's own engineering team later uses Codex to build over 1M lines of code, proving that harness engineering works at scale.

### 📅 February 5, 2026 — Mitchell Hashimoto Coins "Harness Engineering"
- **What:** Mitchell Hashimoto (co-founder of HashiCorp, creator of Terraform/Vagrant) publishes "My AI Adoption Journey" on mitchellh.com. He describes his practice of engineering solutions so agents never repeat mistakes, and names it "harness engineering."
- **Paradigm:** Harness Engineering (named)
- **Why it matters:** The crystallizing moment. Hashimoto gave a name to what practitioners had been doing — building the environment, guardrails, and feedback loops that make agents reliable. Coming from the creator of Terraform (infrastructure-as-code), this framing resonated deeply: harness engineering is "infrastructure-as-code for AI agents."

### 📅 February 11, 2026 — OpenAI Formally Adopts "Harness Engineering"
- **What:** OpenAI publishes "Harness Engineering: Leveraging Codex in an Agent-First World" on their blog, detailing how their engineering team uses harness engineering methodology with Codex agents for production development.
- **Paradigm:** Harness Engineering (industry adoption)
- **Why it matters:** When OpenAI — the company that popularized "prompt engineering" — officially adopts the "harness engineering" term, it signals a paradigm shift. The blog details concrete practices: declarative agent configurations, feedback loops, and structured environments.

### 📅 March 24, 2026 — Anthropic Publishes Harness Design Philosophy
- **What:** Anthropic publishes their harness design philosophy on their engineering blog, detailing their evolution from multi-agent to single-agent architectures and their approach to building reliable agent harnesses.
- **Paradigm:** Harness Engineering (industry adoption)
- **Why it matters:** Both major frontier labs now use "harness engineering" as their framing. The term has achieved industry consensus in under 2 months — reflecting how ready the field was for this conceptual shift.

---

## Summary: The Three Paradigm Shifts

| Era | Period | Core Question | Human's Job | Key Insight |
|---|---|---|---|---|
| **Prompt Engineering** | 2019–2022 | "How do I ask the model?" | Craft the perfect prompt | Better prompts → better outputs |
| **Context Engineering** | 2023–2024 | "What do I give the model?" | Orchestrate context (RAG, tools, agents) | Better context → better outputs |
| **Harness Engineering** | 2025–2026 | "What world does the model operate in?" | Build the environment, guardrails, feedback loops | Better harness → reliable autonomous agents |

---

## Narrative Arc for Presentation

1. **The Prompt Era** was about *talking to* AI — crafting the right words.
2. **The Context Era** was about *feeding* AI — giving it the right information and tools.
3. **The Harness Era** is about *deploying* AI — building the world it operates in autonomously.

Each transition was driven by model capability:
- GPT-3 was smart enough to follow prompts → Prompt Engineering
- GPT-4 was smart enough to use tools → Context Engineering  
- o1/R1 are smart enough to reason autonomously → Harness Engineering

The key message: **The human role didn't diminish — it elevated.** From wordsmithing prompts, to architecting data pipelines, to engineering entire agent operating environments. Each step requires *more* engineering sophistication, not less.
