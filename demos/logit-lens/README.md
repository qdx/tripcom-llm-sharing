# Logit Lens: Layer-by-Layer Belief Visualization

Part of **Demo 1** — "What happens inside a transformer?" for the Trip.com LLM Knowledge Sharing session.

## What Is This?

The **Logit Lens** (nostalgebraist, 2020) is an interpretability technique: at each transformer layer, project the residual stream through the unembedding matrix to see what the model "believes" the next token should be.

This visualization compares **Direct** vs **Chain-of-Thought (CoT)** prompting to show:
- **CoT**: gradual, smooth convergence toward the target token across layers
- **Direct**: uncertain early layers followed by a sharp jump in the final layers

## Quick Start

### Pre-computed (no GPU needed)
```bash
# Just open the HTML — it uses Plotly CDN and pre-computed data
open logit_lens_visualization.html
```

### Live with GPT-2 (requires PyTorch)
```bash
pip install -r requirements.txt
python logit_lens.py --output-dir .
# Or interactive mode:
python logit_lens.py --interactive
```

## Visualizations

1. **Convergence Overlay** — Target token probability across all 12 layers for 3 examples (arithmetic, factual, reasoning). CoT lines (solid) vs Direct (dashed).
2. **Per-Example Detail** — Probability + entropy curves for each prompt pair.
3. **Heatmap** (live mode only) — Top-5 tokens at each layer showing how predictions shift.

## Key Talking Points

- **Early layers (L0–L3)**: Both prompting styles are uncertain — near-zero probability for the target.
- **Middle layers (L4–L8)**: CoT shows earlier convergence — reasoning tokens provide stepping stones.
- **Final layers (L9–L11)**: Direct prompts show a sharp jump — the model "scrambles" without intermediate support.
- **Entropy**: CoT consistently lower → model is more confident with reasoning context.
- **Takeaway**: CoT fundamentally changes the internal computation trajectory, not just the output.

## Files

| File | Description |
|---|---|
| `logit_lens.py` | Main script — supports both live (TransformerLens) and pre-computed modes |
| `logit_lens_visualization.html` | Standalone interactive HTML (pre-computed, no deps needed) |
| `requirements.txt` | Python dependencies for live mode |
