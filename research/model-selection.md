# Task 1.1: Model Selection for CoT Demo

## Research Summary (2026-03-27)

### The Problem
GPT-2 (124M) is too small for CoT prompting to improve answer accuracy. Wei et al. 2022 found CoT is an "emergent ability" that appears at scale (100B+). Smaller models may actually perform WORSE with CoT.

### Key Finding
**HuggingFace Open CoT Leaderboard** reports that **Phi-2 (2.7B) benefits MORE from CoT than larger models like Mixtral**. This makes Phi-2 the ideal candidate:
- Small enough to run on MacBook (2.7B params, ~5.4GB in FP16, ~1.4GB in 4-bit)
- Shows clear CoT improvement on reasoning tasks
- Supported by TransformerLens (needs verification)

### Recommendation: Dual-Model Approach

| Aspect | Model | Why |
|---|---|---|
| **Hidden state visualization** | GPT-2 Small (124M) | Fastest, best tool support (TransformerLens native), great for showing vector space trajectories |
| **CoT accuracy improvement** | Phi-2 (2.7B) or API call to GPT-4o-mini | Shows real accuracy gain with/without CoT |

### Presentation Narrative
1. Start with GPT-2 — show the "memory stack" (context window) and vector space
2. Show CoT prompt in GPT-2 — hidden states take longer path, but accuracy doesn't improve (model too small)
3. Switch to Phi-2 — same CoT prompt, accuracy jumps significantly
4. Punchline: "CoT works by giving the model more computational steps — but only if the model has enough capacity to use them"
5. Bridge: "This is why the industry moved from Prompt Engineering to Context Engineering — we needed to manage much more than just the prompt"

### TransformerLens Compatibility
- GPT-2: ✅ Native support (primary model)
- Phi-2: ⚠️ Need to verify — TransformerLens supports some models via HuggingFace integration
- Alternative: Use nnsight library if TransformerLens doesn't support Phi-2

### Next Steps
- [ ] Verify Phi-2 works with TransformerLens or nnsight
- [ ] Run benchmark: multi-step arithmetic (e.g., "What is 27 × 13 + 45?") with/without CoT on both models
- [ ] Confirm GPT-2 hidden state extraction works on MacBook
