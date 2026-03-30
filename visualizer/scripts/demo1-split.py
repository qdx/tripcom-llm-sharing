"""Demo 1 Split Screen: Context Window + PCA Vector Space.

Injects GPT-2 prompt messages into the context window visualizer
while the PCA panel shows the corresponding hidden state trajectories.

Usage:
  python demo1-split.py              # Full auto demo
  python demo1-split.py --step       # Step-by-step (press Enter)
  python demo1-split.py --fast       # Fast mode (shorter delays)
"""

import httpx
import sys
import time

BASE = "http://localhost:8080"


def wait(seconds: float, step_mode: bool = False):
    if step_mode:
        input("  [Press Enter to continue...]")
    else:
        time.sleep(seconds)


def main():
    step_mode = "--step" in sys.argv
    fast = "--fast" in sys.argv
    delay = 0.5 if fast else 1.5

    print("═" * 60)
    print("Demo 1: Split Screen — Context Window + Vector Space")
    print("═" * 60)
    print()

    # Reset and set mode to simple (triggers split screen)
    httpx.post(f"{BASE}/reset")
    time.sleep(0.3)
    httpx.post(f"{BASE}/mode", json={"mode": "simple"})
    time.sleep(0.3)

    print("📋 The visualizer should now show the split-screen layout:")
    print("   Left:  Context window (messages)")
    print("   Right: PCA trajectory visualization")
    print()

    # --- Scene 1: System prompt ---
    print("┌─ Scene 1: System Prompt")
    print("│  Establishing the model's role...")
    httpx.post(f"{BASE}/inject", json={
        "role": "system",
        "content": (
            "You are GPT-2 Small, a 124M parameter language model with "
            "12 transformer layers and 768-dimensional hidden states. "
            "Your context window is 1024 tokens. "
            "Answer questions about the world."
        ),
    })
    print("│  ✅ System message injected → visible in left panel")
    print("│  💡 Point: This is the first entry in the context window")
    print("└─")
    wait(delay * 1.5, step_mode)

    # --- Scene 2: User query (matches PCA prompt) ---
    print()
    print("┌─ Scene 2: User Query")
    print("│  The user asks about France — same prompt as PCA data...")
    httpx.post(f"{BASE}/inject", json={
        "role": "user",
        "content": "The capital of France is",
    })
    print("│  ✅ User message injected")
    print("│  👉 Right panel: Notice how each token traces a different")
    print("│     trajectory through the 12 layers of hidden states")
    print("│  💡 'France' token diverges most — it carries semantic load")
    print("└─")
    wait(delay * 2, step_mode)

    # --- Scene 3: Model response ---
    print()
    print("┌─ Scene 3: Model Response")
    print("│  The model completes the sentence...")
    httpx.post(f"{BASE}/inject", json={
        "role": "assistant",
        "content": (
            "Paris. Paris has been the capital of France since the late "
            "10th century. It is the largest city in France and serves as "
            "the country's political, economic, and cultural center."
        ),
    })
    print("│  ✅ Assistant response injected")
    print("│  💡 Context window now contains 3 messages")
    print("│  💡 Token counter shows total context usage")
    print("└─")
    wait(delay * 2, step_mode)

    # --- Scene 4: Direct vs CoT comparison ---
    print()
    print("┌─ Scene 4: Direct vs Chain-of-Thought")
    print("│  Switch to 'Direct vs CoT' view in the right panel...")
    print("│  👉 Click 'Direct vs CoT' button in the PCA panel")
    print("│")
    print("│  Key insight:")
    print("│  • Direct prompt: short path through representation space")
    print("│  • CoT prompt: longer path = more 'thinking' in hidden states")
    print("│  • The context window IS the model's working memory")
    print("│  • More tokens in prompt = more space for reasoning")
    print("└─")
    wait(delay, step_mode)

    # --- Scene 5: Show CoT in context window ---
    print()
    print("┌─ Scene 5: CoT Prompt in Context Window")
    print("│  Now let's see what CoT looks like in the context window...")

    # Reset and inject CoT example
    httpx.post(f"{BASE}/reset")
    time.sleep(0.3)
    httpx.post(f"{BASE}/mode", json={"mode": "simple"})
    time.sleep(0.3)

    httpx.post(f"{BASE}/inject", json={
        "role": "system",
        "content": "You are a helpful math assistant. Think step by step.",
    })
    wait(delay * 0.5, step_mode)

    httpx.post(f"{BASE}/inject", json={
        "role": "user",
        "content": "Q: What is 23 + 45?",
    })
    wait(delay, step_mode)

    httpx.post(f"{BASE}/inject", json={
        "role": "assistant",
        "content": (
            "A: Let me think step by step.\n\n"
            "23 + 45\n"
            "= 20 + 40 + 3 + 5\n"
            "= 60 + 8\n"
            "= 68\n\n"
            "The answer is 68."
        ),
    })
    print("│  ✅ CoT conversation injected")
    print("│  💡 The assistant message is MUCH longer — it contains")
    print("│     the step-by-step reasoning. This uses more context")
    print("│     window space but produces more accurate results.")
    print("│")
    print("│  🔑 KEY POINT: Chain-of-Thought works because it gives")
    print("│     the model more tokens to 'think through' the problem.")
    print("│     The context window is literally the model's scratchpad.")
    print("└─")

    print()
    print("═" * 60)
    print("✅ Demo 1 complete!")
    print()
    print("Summary:")
    print("  • Context window = model's entire working memory")
    print("  • Each token traverses 12 transformer layers")
    print("  • Hidden states diverge based on semantic content")
    print("  • CoT = more tokens = longer path = better reasoning")
    print("  • This is why prompt engineering matters!")
    print("═" * 60)


if __name__ == "__main__":
    main()
