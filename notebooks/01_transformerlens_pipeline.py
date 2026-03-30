"""
TransformerLens + Model Inference Pipeline
==========================================
Task 1.2: Set up TransformerLens, load GPT-2, extract hidden states per layer per token.
Verify forward pass works on MacBook.

This script can be run as-is or converted to a Jupyter notebook via:
  jupyter notebook  (then open this as .py with jupytext, or use the .ipynb version)

For the knowledge sharing demo: "From Prompt Engineering to Harness Engineering"
"""

# %% [markdown]
# # TransformerLens Pipeline: Extracting LLM Hidden States
# 
# This notebook demonstrates how to:
# 1. Load GPT-2 with TransformerLens
# 2. Run a forward pass and extract hidden states at every layer
# 3. Inspect attention patterns
# 4. Compare Direct vs CoT prompting in representation space
# 
# **Target audience:** Trip.com software engineers learning LLM internals

# %% Setup and imports
import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %% [markdown]
# ## 1. Load GPT-2 Small with TransformerLens
# 
# GPT-2 Small (124M params, 12 layers, 768-dim hidden states)
# - Perfect for visualization: small enough to inspect every layer
# - TransformerLens wraps it with hooks at every component

# %% Load model
print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
print(f"Model loaded: {model.cfg.model_name}")
print(f"  Layers: {model.cfg.n_layers}")
print(f"  Hidden dim: {model.cfg.d_model}")
print(f"  Attention heads: {model.cfg.n_heads}")
print(f"  Head dim: {model.cfg.d_head}")
print(f"  Vocab size: {model.cfg.d_vocab}")
print(f"  Context window: {model.cfg.n_ctx}")

# %% [markdown]
# ## 2. Basic Forward Pass with Hidden State Extraction
# 
# TransformerLens provides `run_with_cache()` which captures
# activations at every layer. This is the key tool for mechanistic interpretability.

# %% Run forward pass with cache
prompt = "The capital of France is"
print(f"Prompt: '{prompt}'")

# Tokenize
tokens = model.to_tokens(prompt)
token_strs = model.to_str_tokens(prompt)
print(f"Tokens ({len(token_strs)}): {token_strs}")

# Run with cache - this captures ALL intermediate activations
logits, cache = model.run_with_cache(prompt)
print(f"\nLogits shape: {logits.shape}")  # [batch, seq_len, vocab_size]

# Show top predictions for last token
last_token_logits = logits[0, -1, :]
top_k = 10
top_values, top_indices = torch.topk(last_token_logits, top_k)
print(f"\nTop {top_k} predictions after '{prompt}':")
for i, (val, idx) in enumerate(zip(top_values, top_indices)):
    token = model.tokenizer.decode(idx.item())
    print(f"  {i+1}. '{token}' (logit: {val.item():.2f})")

# %% [markdown]
# ## 3. Extract Hidden States Per Layer Per Token
# 
# The cache contains residual stream activations at every layer.
# Shape: [batch, seq_len, d_model] for each layer.

# %% Extract hidden states
n_layers = model.cfg.n_layers
n_tokens = len(token_strs)
d_model = model.cfg.d_model

# Collect residual stream at each layer (post-attention + post-MLP = "resid_post")
hidden_states = torch.zeros(n_layers + 1, n_tokens, d_model)

# Layer 0 = embedding (before any transformer layer)
hidden_states[0] = cache["embed"][0]  # [seq_len, d_model]

# Layers 1..n_layers = output of each transformer block
for layer in range(n_layers):
    act_name = get_act_name("resid_post", layer)
    hidden_states[layer + 1] = cache[act_name][0]  # [seq_len, d_model]

print(f"Hidden states tensor shape: {hidden_states.shape}")
print(f"  = [{n_layers + 1} layers] × [{n_tokens} tokens] × [{d_model} dims]")

# Show norms per layer for last token (demonstrates information accumulation)
print(f"\nResidual stream norm per layer (last token '{token_strs[-1]}'):")
for layer in range(n_layers + 1):
    norm = hidden_states[layer, -1].norm().item()
    bar = "█" * int(norm / 2)
    label = "embed" if layer == 0 else f"layer {layer:2d}"
    print(f"  {label}: {norm:7.2f} {bar}")

# %% [markdown]
# ## 4. PCA Visualization: Token Trajectories Through Layers
# 
# Each token traces a path through 768-dimensional space as it passes through layers.
# PCA projects this to 2D so we can visualize the trajectory.

# %% PCA trajectory visualization
# Reshape: all (layer, token) pairs into flat matrix for PCA
all_hidden = hidden_states.detach().numpy().reshape(-1, d_model)  # [(n_layers+1)*n_tokens, d_model]

pca = PCA(n_components=2)
projected = pca.fit_transform(all_hidden)  # [(n_layers+1)*n_tokens, 2]
projected = projected.reshape(n_layers + 1, n_tokens, 2)

print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}")

# Build trajectory plot
fig = go.Figure()
colors = px.colors.qualitative.Set2

for t_idx in range(n_tokens):
    xs = projected[:, t_idx, 0]
    ys = projected[:, t_idx, 1]
    token_label = token_strs[t_idx].replace(" ", "·")
    
    # Trajectory line
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines+markers',
        name=f"'{token_label}'",
        line=dict(color=colors[t_idx % len(colors)], width=2),
        marker=dict(size=6),
        hovertemplate=f"Token: {token_label}<br>Layer: %{{customdata}}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}",
        customdata=list(range(n_layers + 1))
    ))
    
    # Start marker
    fig.add_trace(go.Scatter(
        x=[xs[0]], y=[ys[0]], mode='markers',
        marker=dict(size=12, symbol='circle', color=colors[t_idx % len(colors)],
                    line=dict(width=2, color='black')),
        showlegend=False, hoverinfo='skip'
    ))
    
    # End marker
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]], mode='markers',
        marker=dict(size=12, symbol='star', color=colors[t_idx % len(colors)],
                    line=dict(width=2, color='black')),
        showlegend=False, hoverinfo='skip'
    ))

fig.update_layout(
    title="Token Trajectories Through GPT-2 Layers (PCA)",
    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
    template="plotly_dark",
    width=900, height=600,
    legend=dict(x=1.02, y=1)
)
fig.write_html("../outputs/token_trajectories_pca.html")
print("Saved: outputs/token_trajectories_pca.html")

# %% [markdown]
# ## 5. Attention Pattern Extraction
# 
# Visualize which tokens attend to which at each layer/head.

# %% Attention patterns
print("Attention patterns shape per layer:")
layer_idx = 5  # Middle layer
attn_pattern = cache["pattern", layer_idx][0]  # [n_heads, seq_len, seq_len]
print(f"  Layer {layer_idx}: {attn_pattern.shape} = [{model.cfg.n_heads} heads] × [{n_tokens} src] × [{n_tokens} dst]")

# Visualize one attention head
head_idx = 0
attn = attn_pattern[head_idx].detach().numpy()

fig_attn = go.Figure(data=go.Heatmap(
    z=attn,
    x=token_strs,
    y=token_strs,
    colorscale='Viridis',
    text=[[f"{v:.2f}" for v in row] for row in attn],
    texttemplate="%{text}",
    textfont={"size": 10},
))

fig_attn.update_layout(
    title=f"Attention Pattern: Layer {layer_idx}, Head {head_idx}",
    xaxis_title="Key (attending to)",
    yaxis_title="Query (attending from)",
    template="plotly_dark",
    width=600, height=500,
)
fig_attn.write_html("../outputs/attention_pattern.html")
print(f"Saved: outputs/attention_pattern.html")

# %% [markdown]
# ## 6. Direct vs CoT Comparison (Hidden State Trajectories)
# 
# This is the core demo: show how Chain-of-Thought changes the model's
# internal trajectory through representation space.

# %% Direct vs CoT comparison
direct_prompt = "Q: What is 23 + 45?\nA: The answer is"
cot_prompt = "Q: What is 23 + 45?\nA: Let me think step by step. 23 + 45 = 20 + 40 + 3 + 5 = 60 + 8 = 68. The answer is"

print(f"Direct prompt ({len(model.to_str_tokens(direct_prompt))} tokens):")
print(f"  {direct_prompt}")
print(f"\nCoT prompt ({len(model.to_str_tokens(cot_prompt))} tokens):")
print(f"  {cot_prompt}")

# Run both
_, cache_direct = model.run_with_cache(direct_prompt)
_, cache_cot = model.run_with_cache(cot_prompt)

# Extract last-token hidden states at each layer (the "answer position")
def extract_last_token_trajectory(cache, n_layers, d_model):
    """Extract hidden state of last token at each layer."""
    trajectory = torch.zeros(n_layers + 1, d_model)
    trajectory[0] = cache["embed"][0, -1]  # last token embedding
    for layer in range(n_layers):
        act_name = get_act_name("resid_post", layer)
        trajectory[layer + 1] = cache[act_name][0, -1]
    return trajectory

traj_direct = extract_last_token_trajectory(cache_direct, n_layers, d_model)
traj_cot = extract_last_token_trajectory(cache_cot, n_layers, d_model)

# PCA on combined trajectories
combined = torch.cat([traj_direct, traj_cot], dim=0).detach().numpy()
pca2 = PCA(n_components=2)
proj = pca2.fit_transform(combined)
proj_direct = proj[:n_layers + 1]
proj_cot = proj[n_layers + 1:]

# Compute path lengths
def path_length(trajectory_2d):
    diffs = np.diff(trajectory_2d, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

len_direct = path_length(proj_direct)
len_cot = path_length(proj_cot)
print(f"\nPath length in PCA space:")
print(f"  Direct: {len_direct:.2f}")
print(f"  CoT:    {len_cot:.2f}")
print(f"  Ratio:  {len_cot/len_direct:.2f}x")

# Plot comparison
fig_comp = go.Figure()

# Direct trajectory
fig_comp.add_trace(go.Scatter(
    x=proj_direct[:, 0], y=proj_direct[:, 1],
    mode='lines+markers', name=f'Direct (path={len_direct:.1f})',
    line=dict(color='#ff6b6b', width=3),
    marker=dict(size=8),
    customdata=list(range(n_layers + 1)),
    hovertemplate="Direct<br>Layer %{customdata}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}"
))

# CoT trajectory  
fig_comp.add_trace(go.Scatter(
    x=proj_cot[:, 0], y=proj_cot[:, 1],
    mode='lines+markers', name=f'CoT (path={len_cot:.1f})',
    line=dict(color='#4ecdc4', width=3),
    marker=dict(size=8),
    customdata=list(range(n_layers + 1)),
    hovertemplate="CoT<br>Layer %{customdata}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}"
))

# Start/end markers
for proj_data, color, label in [(proj_direct, '#ff6b6b', 'Direct'), (proj_cot, '#4ecdc4', 'CoT')]:
    fig_comp.add_trace(go.Scatter(
        x=[proj_data[0, 0]], y=[proj_data[0, 1]], mode='markers',
        marker=dict(size=15, symbol='circle', color=color, line=dict(width=2, color='white')),
        showlegend=False, name=f'{label} start'
    ))
    fig_comp.add_trace(go.Scatter(
        x=[proj_data[-1, 0]], y=[proj_data[-1, 1]], mode='markers',
        marker=dict(size=15, symbol='star', color=color, line=dict(width=2, color='white')),
        showlegend=False, name=f'{label} end'
    ))

fig_comp.update_layout(
    title="Direct vs CoT: Last Token Trajectory Through GPT-2 Layers",
    xaxis_title=f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} var)",
    yaxis_title=f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} var)",
    template="plotly_dark",
    width=900, height=600,
    annotations=[dict(
        text=f"⭕ = Layer 0 (embed), ⭐ = Layer {n_layers} (final)<br>"
             f"CoT path is {len_cot/len_direct:.1f}× longer in representation space",
        xref="paper", yref="paper", x=0.5, y=-0.12,
        showarrow=False, font=dict(size=12)
    )]
)
fig_comp.write_html("../outputs/direct_vs_cot_trajectory.html")
print("Saved: outputs/direct_vs_cot_trajectory.html")

# %% [markdown]
# ## 7. Layer-by-Layer Cosine Similarity (Direct vs CoT)
# 
# How similar are the last-token representations between Direct and CoT at each layer?

# %% Cosine similarity analysis
cos_sims = []
for layer in range(n_layers + 1):
    cos_sim = torch.nn.functional.cosine_similarity(
        traj_direct[layer].unsqueeze(0), 
        traj_cot[layer].unsqueeze(0)
    ).item()
    cos_sims.append(cos_sim)

fig_cos = go.Figure(go.Bar(
    x=[f"{'embed' if i == 0 else f'L{i}'}" for i in range(n_layers + 1)],
    y=cos_sims,
    marker_color=['#666'] + [f'rgb({int(255*(1-s))}, {int(255*s)}, 100)' for s in cos_sims[1:]],
    text=[f"{s:.3f}" for s in cos_sims],
    textposition='outside'
))

fig_cos.update_layout(
    title="Cosine Similarity: Direct vs CoT Last-Token Hidden States Per Layer",
    xaxis_title="Layer",
    yaxis_title="Cosine Similarity",
    yaxis_range=[min(cos_sims) - 0.1, 1.05],
    template="plotly_dark",
    width=900, height=400,
)
fig_cos.write_html("../outputs/cosine_similarity_layers.html")
print("Saved: outputs/cosine_similarity_layers.html")

# %% [markdown]
# ## 8. Summary & Verification
# 
# ✅ TransformerLens loaded GPT-2 Small successfully  
# ✅ Forward pass extracts hidden states at all 12 layers  
# ✅ Attention patterns captured  
# ✅ PCA trajectory visualization works  
# ✅ Direct vs CoT comparison shows different paths through representation space

# %% Print summary
print("=" * 60)
print("PIPELINE VERIFICATION SUMMARY")
print("=" * 60)
print(f"Model:          GPT-2 Small (124M params)")
print(f"Layers:         {n_layers}")
print(f"Hidden dim:     {d_model}")
print(f"Device:         CPU (MacBook compatible)")
print(f"Cache entries:  {len(cache)} activation tensors captured")
print(f"Hidden states:  {hidden_states.shape} extracted successfully")
print(f"Attention:      {attn_pattern.shape} patterns captured")
print(f"PCA viz:        ✅ Token trajectories rendered")
print(f"Direct vs CoT:  ✅ Path length ratio = {len_cot/len_direct:.2f}x")
print(f"Outputs:        outputs/*.html (interactive Plotly)")
print("=" * 60)
print("\nReady for Task 1.3 (Hidden state trajectory visualization)")
