#!/usr/bin/env python3
"""
Task 1.3: Hidden State Trajectory Visualization — PCA/t-SNE Direct vs CoT Comparison

Extracts hidden states from GPT-2 for Direct and Chain-of-Thought prompts,
reduces to 2D/3D via PCA and t-SNE, and plots trajectories showing how CoT
takes a longer path through representation space.

Generates interactive Plotly HTML files for use in the Trip.com LLM Knowledge Sharing presentation.
"""

import os
import json
import numpy as np
import torch
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "hidden_states")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Load Model
# ============================================================

print("Loading GPT-2 Small via TransformerLens...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model}d hidden states")


# ============================================================
# 2. Define Prompts — Direct vs CoT
# ============================================================

PROMPTS = {
    "Direct": "Q: What is 23 + 47?\nA: The answer is",
    "CoT": "Q: What is 23 + 47?\nA: Let me think step by step. First, 23 + 47. I add the ones: 3 + 7 = 10, carry 1. Then the tens: 2 + 4 + 1 = 7. So the answer is",
}

# Additional prompt pairs for robustness
EXTRA_PROMPTS = {
    "Direct_multi": "Q: A store has 15 apples. 8 are sold. How many remain?\nA: The answer is",
    "CoT_multi": "Q: A store has 15 apples. 8 are sold. How many remain?\nA: Let me work through this. The store starts with 15 apples. Then 8 are sold, so I subtract: 15 - 8 = 7. The answer is",
    "Direct_logic": "Q: If all cats are animals, and Whiskers is a cat, is Whiskers an animal?\nA:",
    "CoT_logic": "Q: If all cats are animals, and Whiskers is a cat, is Whiskers an animal?\nA: Let me reason through this. Premise 1: All cats are animals. Premise 2: Whiskers is a cat. Since Whiskers is a cat, and all cats are animals, then Whiskers must be an animal. So the answer is:",
}


# ============================================================
# 3. Extract Hidden States
# ============================================================

def extract_hidden_states(model, prompt: str) -> dict:
    """Run a forward pass and extract hidden states at every layer for every token."""
    tokens = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)

    # Run with cache to get all intermediate activations
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    # Extract residual stream at each layer (post-attention + MLP)
    # Shape per layer: (seq_len, d_model)
    hidden_states = {}
    for layer in range(model.cfg.n_layers):
        # Use the residual stream post each layer
        key = f"blocks.{layer}.hook_resid_post"
        hidden_states[layer] = cache[key].detach().numpy()  # (seq_len, d_model)

    # Also get the embedding (layer 0 input)
    embed = cache["hook_embed"].detach().numpy()  # (seq_len, d_model)

    return {
        "tokens": token_strs,
        "n_tokens": len(token_strs),
        "embed": embed,
        "hidden_states": hidden_states,  # layer -> (seq_len, d_model)
        "n_layers": model.cfg.n_layers,
    }


def get_last_token_trajectory(result: dict) -> np.ndarray:
    """Get the hidden state of the LAST token across all layers.
    This traces the 'thought trajectory' — how the model's representation
    of the final prediction evolves through layers.
    Shape: (n_layers + 1, d_model) — embedding + each layer.
    """
    last_idx = result["n_tokens"] - 1
    trajectory = [result["embed"][last_idx]]  # Start with embedding
    for layer in range(result["n_layers"]):
        trajectory.append(result["hidden_states"][layer][last_idx])
    return np.array(trajectory)


def get_all_tokens_at_layer(result: dict, layer: int) -> np.ndarray:
    """Get hidden states of ALL tokens at a specific layer.
    Shape: (n_tokens, d_model)
    """
    if layer == -1:
        return result["embed"]
    return result["hidden_states"][layer]


# ============================================================
# 4. Run Extraction
# ============================================================

print("\nExtracting hidden states...")
results = {}
for name, prompt in {**PROMPTS, **EXTRA_PROMPTS}.items():
    print(f"  Processing: {name} ({len(prompt)} chars)")
    results[name] = extract_hidden_states(model, prompt)
    print(f"    → {results[name]['n_tokens']} tokens, {results[name]['n_layers']} layers")


# ============================================================
# 5. Visualization 1: Last-Token Trajectory Through Layers (PCA 2D)
# ============================================================

def plot_trajectory_2d(prompt_pairs, title_suffix="", filename="trajectory_2d.html"):
    """Plot the trajectory of the last token through layers for Direct vs CoT."""
    # Collect all trajectories for joint PCA
    all_trajectories = []
    labels = []
    for name in prompt_pairs:
        traj = get_last_token_trajectory(results[name])
        all_trajectories.append(traj)
        labels.append(name)

    # Stack all trajectories for joint PCA fitting
    all_points = np.vstack(all_trajectories)
    pca = PCA(n_components=2)
    all_reduced = pca.fit_transform(all_points)

    # Split back
    fig = go.Figure()
    offset = 0
    colors = {
        "Direct": "#FF6B6B", "CoT": "#4ECDC4",
        "Direct_multi": "#FF8E8E", "CoT_multi": "#6BE0D6",
        "Direct_logic": "#FFB3B3", "CoT_logic": "#8EEDEA",
    }

    for name, traj in zip(labels, all_trajectories):
        n = len(traj)
        reduced = all_reduced[offset:offset + n]
        offset += n

        is_cot = "CoT" in name
        color = colors.get(name, "#888888")

        # Line trace
        fig.add_trace(go.Scatter(
            x=reduced[:, 0], y=reduced[:, 1],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=3 if is_cot else 2),
            marker=dict(size=8, color=color, 
                       line=dict(width=1, color="white")),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Layer: %{customdata}<br>"
                "PC1: %{x:.3f}<br>"
                "PC2: %{y:.3f}<extra></extra>"
            ),
            customdata=[f"embed" if i == 0 else f"{i}" for i in range(n)],
        ))

        # Mark start and end
        fig.add_trace(go.Scatter(
            x=[reduced[0, 0]], y=[reduced[0, 1]],
            mode="markers",
            marker=dict(size=14, color=color, symbol="star", 
                       line=dict(width=2, color="white")),
            name=f"{name} (start)",
            showlegend=False,
            hovertemplate=f"<b>{name} START</b> (embedding)<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[reduced[-1, 0]], y=[reduced[-1, 1]],
            mode="markers",
            marker=dict(size=14, color=color, symbol="diamond",
                       line=dict(width=2, color="white")),
            name=f"{name} (end)",
            showlegend=False,
            hovertemplate=f"<b>{name} END</b> (layer {n-1})<extra></extra>",
        ))

    # Compute path lengths
    annotations = []
    offset = 0
    for name, traj in zip(labels, all_trajectories):
        n = len(traj)
        reduced = all_reduced[offset:offset + n]
        offset += n
        path_len = sum(np.linalg.norm(reduced[i+1] - reduced[i]) for i in range(n-1))
        annotations.append(f"{name}: path length = {path_len:.2f}")

    variance_explained = pca.explained_variance_ratio_
    fig.update_layout(
        title=dict(
            text=f"Hidden State Trajectory: Direct vs CoT {title_suffix}<br>"
                 f"<sub>Last token representation through GPT-2 layers (PCA 2D) | "
                 f"Variance explained: PC1={variance_explained[0]:.1%}, PC2={variance_explained[1]:.1%}</sub>",
            font=dict(size=16),
        ),
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_dark",
        width=1000,
        height=700,
        legend=dict(x=0.02, y=0.98),
        annotations=[dict(
            text="<br>".join(annotations),
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            showarrow=False,
            font=dict(size=11, color="white"),
            align="right",
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=6,
        )],
    )

    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(outpath, include_plotlyjs="cdn")
    print(f"  Saved: {outpath}")
    return fig


# ============================================================
# 6. Visualization 2: Last-Token Trajectory Through Layers (PCA 3D)
# ============================================================

def plot_trajectory_3d(prompt_pairs, title_suffix="", filename="trajectory_3d.html"):
    """3D PCA trajectory of last token through layers."""
    all_trajectories = []
    labels = []
    for name in prompt_pairs:
        traj = get_last_token_trajectory(results[name])
        all_trajectories.append(traj)
        labels.append(name)

    all_points = np.vstack(all_trajectories)
    pca = PCA(n_components=3)
    all_reduced = pca.fit_transform(all_points)

    fig = go.Figure()
    offset = 0
    colors = {
        "Direct": "#FF6B6B", "CoT": "#4ECDC4",
        "Direct_multi": "#FF8E8E", "CoT_multi": "#6BE0D6",
        "Direct_logic": "#FFB3B3", "CoT_logic": "#8EEDEA",
    }

    path_lengths = {}
    for name, traj in zip(labels, all_trajectories):
        n = len(traj)
        reduced = all_reduced[offset:offset + n]
        offset += n
        color = colors.get(name, "#888888")
        is_cot = "CoT" in name

        path_len = sum(np.linalg.norm(reduced[i+1] - reduced[i]) for i in range(n-1))
        path_lengths[name] = path_len

        fig.add_trace(go.Scatter3d(
            x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2],
            mode="lines+markers",
            name=f"{name} (path={path_len:.1f})",
            line=dict(color=color, width=5 if is_cot else 3),
            marker=dict(size=5, color=color,
                       line=dict(width=1, color="white")),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Layer: %{customdata}<br>"
                "PC1: %{x:.3f}<br>"
                "PC2: %{y:.3f}<br>"
                "PC3: %{z:.3f}<extra></extra>"
            ),
            customdata=[f"embed" if i == 0 else f"{i}" for i in range(n)],
        ))

        # Start marker
        fig.add_trace(go.Scatter3d(
            x=[reduced[0, 0]], y=[reduced[0, 1]], z=[reduced[0, 2]],
            mode="markers",
            marker=dict(size=10, color=color, symbol="diamond"),
            name=f"{name} start", showlegend=False,
        ))
        # End marker
        fig.add_trace(go.Scatter3d(
            x=[reduced[-1, 0]], y=[reduced[-1, 1]], z=[reduced[-1, 2]],
            mode="markers",
            marker=dict(size=10, color="gold" if is_cot else "red", symbol="x"),
            name=f"{name} end", showlegend=False,
        ))

    variance_explained = pca.explained_variance_ratio_
    fig.update_layout(
        title=dict(
            text=f"3D Hidden State Trajectory: Direct vs CoT {title_suffix}<br>"
                 f"<sub>Rotate to explore! | "
                 f"Variance: PC1={variance_explained[0]:.1%}, PC2={variance_explained[1]:.1%}, PC3={variance_explained[2]:.1%}</sub>",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        template="plotly_dark",
        width=1000,
        height=800,
    )

    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(outpath, include_plotlyjs="cdn")
    print(f"  Saved: {outpath}")
    return fig


# ============================================================
# 7. Visualization 3: t-SNE of All Token Hidden States at Final Layer
# ============================================================

def plot_tsne_all_tokens(prompt_pairs, layer=-1, filename="tsne_tokens.html"):
    """t-SNE visualization of all token hidden states at a given layer.
    Shows how Direct and CoT tokens occupy different regions of representation space.
    layer=-1 means final layer.
    """
    actual_layer = list(results.values())[0]["n_layers"] - 1 if layer == -1 else layer

    all_states = []
    all_labels = []
    all_tokens = []
    all_prompt_types = []

    for name in prompt_pairs:
        r = results[name]
        states = get_all_tokens_at_layer(r, actual_layer)
        tokens = r["tokens"]
        prompt_type = "CoT" if "CoT" in name else "Direct"

        for i, (state, tok) in enumerate(zip(states, tokens)):
            all_states.append(state)
            all_labels.append(name)
            all_tokens.append(tok)
            all_prompt_types.append(prompt_type)

    all_states = np.array(all_states)

    # t-SNE reduction
    tsne = TSNE(n_components=2, perplexity=min(15, len(all_states) - 1),
                random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(all_states)

    fig = go.Figure()
    colors = {"Direct": "#FF6B6B", "CoT": "#4ECDC4"}
    markers = {"Direct": "circle", "CoT": "diamond"}

    for prompt_type in ["Direct", "CoT"]:
        mask = [pt == prompt_type for pt in all_prompt_types]
        indices = [i for i, m in enumerate(mask) if m]

        fig.add_trace(go.Scatter(
            x=reduced[indices, 0],
            y=reduced[indices, 1],
            mode="markers+text",
            name=prompt_type,
            marker=dict(
                size=12,
                color=colors[prompt_type],
                symbol=markers[prompt_type],
                line=dict(width=1, color="white"),
            ),
            text=[all_tokens[i] for i in indices],
            textposition="top center",
            textfont=dict(size=8, color="white"),
            hovertemplate=(
                f"<b>{prompt_type}</b><br>"
                "Token: %{text}<br>"
                "Source: %{customdata}<extra></extra>"
            ),
            customdata=[all_labels[i] for i in indices],
        ))

    fig.update_layout(
        title=dict(
            text=f"t-SNE: Token Representations at Layer {actual_layer}<br>"
                 f"<sub>All tokens from Direct vs CoT prompts — shows how reasoning tokens create distinct clusters</sub>",
            font=dict(size=16),
        ),
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        template="plotly_dark",
        width=1100,
        height=800,
    )

    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(outpath, include_plotlyjs="cdn")
    print(f"  Saved: {outpath}")
    return fig


# ============================================================
# 8. Visualization 4: Layer-by-Layer Distance Between Direct and CoT
# ============================================================

def plot_layer_distance(direct_name, cot_name, filename="layer_distance.html"):
    """Plot the Euclidean distance between Direct and CoT last-token hidden states
    at each layer. Shows where in the network the representations diverge."""
    r_direct = results[direct_name]
    r_cot = results[cot_name]

    distances = []
    layers = []

    # Embedding layer
    d_embed = np.linalg.norm(r_direct["embed"][-1] - r_cot["embed"][-1])
    distances.append(d_embed)
    layers.append("embed")

    for layer in range(r_direct["n_layers"]):
        d = np.linalg.norm(
            r_direct["hidden_states"][layer][-1] - r_cot["hidden_states"][layer][-1]
        )
        distances.append(d)
        layers.append(str(layer))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=layers, y=distances,
        marker=dict(
            color=distances,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Distance"),
        ),
        hovertemplate="Layer: %{x}<br>Distance: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"Representation Divergence: {direct_name} vs {cot_name}<br>"
                 f"<sub>Euclidean distance between last-token hidden states at each layer</sub>",
            font=dict(size=16),
        ),
        xaxis_title="Layer",
        yaxis_title="Euclidean Distance",
        template="plotly_dark",
        width=900,
        height=500,
    )

    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(outpath, include_plotlyjs="cdn")
    print(f"  Saved: {outpath}")
    return fig


# ============================================================
# 9. Visualization 5: Animated trajectory (step-by-step layer progression)
# ============================================================

def plot_animated_trajectory(prompt_pairs, filename="trajectory_animated.html"):
    """Animated 2D PCA trajectory showing layer-by-layer progression."""
    all_trajectories = []
    labels = []
    for name in prompt_pairs:
        traj = get_last_token_trajectory(results[name])
        all_trajectories.append(traj)
        labels.append(name)

    all_points = np.vstack(all_trajectories)
    pca = PCA(n_components=2)
    all_reduced = pca.fit_transform(all_points)

    # Split back into trajectories
    reduced_trajs = {}
    offset = 0
    for name, traj in zip(labels, all_trajectories):
        n = len(traj)
        reduced_trajs[name] = all_reduced[offset:offset + n]
        offset += n

    n_layers = all_trajectories[0].shape[0]
    colors = {"Direct": "#FF6B6B", "CoT": "#4ECDC4"}

    # Create frames for animation
    frames = []
    for step in range(1, n_layers + 1):
        frame_data = []
        for name in labels:
            reduced = reduced_trajs[name]
            color = colors.get(name.split("_")[0], "#888888")
            frame_data.append(go.Scatter(
                x=reduced[:step, 0], y=reduced[:step, 1],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=3),
                marker=dict(size=8, color=color),
            ))
        layer_label = "embed" if step == 1 else f"Layer {step - 1}"
        frames.append(go.Frame(data=frame_data, name=layer_label))

    # Initial frame (just the start points)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[reduced_trajs[name][0, 0]], y=[reduced_trajs[name][0, 1]],
                mode="markers",
                name=name,
                line=dict(color=colors.get(name.split("_")[0], "#888888"), width=3),
                marker=dict(size=10, color=colors.get(name.split("_")[0], "#888888")),
            )
            for name in labels
        ],
        frames=frames,
    )

    # Animation controls
    fig.update_layout(
        title=dict(
            text="Animated: Hidden State Evolution Through Layers<br>"
                 "<sub>Press Play to watch representations evolve layer by layer</sub>",
            font=dict(size=16),
        ),
        xaxis_title="PC1",
        yaxis_title="PC2",
        xaxis=dict(range=[all_reduced[:, 0].min() - 1, all_reduced[:, 0].max() + 1]),
        yaxis=dict(range=[all_reduced[:, 1].min() - 1, all_reduced[:, 1].max() + 1]),
        template="plotly_dark",
        width=1000,
        height=700,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.12, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 500, "redraw": True},
                                  "fromcurrent": True}]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[
                dict(args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                       "mode": "immediate"}],
                     label=f.name, method="animate")
                for f in frames
            ],
            x=0.05, len=0.9,
            currentvalue=dict(prefix="Layer: ", font=dict(size=14)),
        )],
    )

    outpath = os.path.join(OUTPUT_DIR, filename)
    fig.write_html(outpath, include_plotlyjs="cdn")
    print(f"  Saved: {outpath}")
    return fig


# ============================================================
# 10. Generate Summary Statistics
# ============================================================

def compute_path_stats(prompt_pairs):
    """Compute and print path length statistics for presentation."""
    all_trajectories = []
    labels = []
    for name in prompt_pairs:
        traj = get_last_token_trajectory(results[name])
        all_trajectories.append(traj)
        labels.append(name)

    all_points = np.vstack(all_trajectories)
    pca = PCA(n_components=2)
    all_reduced = pca.fit_transform(all_points)

    stats = {}
    offset = 0
    for name, traj in zip(labels, all_trajectories):
        n = len(traj)
        reduced = all_reduced[offset:offset + n]
        offset += n

        path_len = sum(np.linalg.norm(reduced[i+1] - reduced[i]) for i in range(n-1))
        start_end_dist = np.linalg.norm(reduced[-1] - reduced[0])
        stats[name] = {
            "path_length_pca2d": round(float(path_len), 3),
            "start_end_distance": round(float(start_end_dist), 3),
            "tortuosity": round(float(path_len / max(start_end_dist, 1e-6)), 3),
            "n_tokens": results[name]["n_tokens"],
        }

    return stats


# ============================================================
# Main: Generate All Visualizations
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    # Primary comparison: arithmetic
    primary = ["Direct", "CoT"]
    print("\n--- Primary (Arithmetic) ---")
    plot_trajectory_2d(primary, "(Arithmetic)", "trajectory_2d_arithmetic.html")
    plot_trajectory_3d(primary, "(Arithmetic)", "trajectory_3d_arithmetic.html")
    plot_animated_trajectory(primary, "trajectory_animated_arithmetic.html")
    plot_layer_distance("Direct", "CoT", "layer_distance_arithmetic.html")

    # Multi-step comparison
    multi = ["Direct_multi", "CoT_multi"]
    print("\n--- Multi-step (Subtraction) ---")
    plot_trajectory_2d(multi, "(Subtraction)", "trajectory_2d_subtraction.html")
    plot_trajectory_3d(multi, "(Subtraction)", "trajectory_3d_subtraction.html")

    # Logic comparison
    logic = ["Direct_logic", "CoT_logic"]
    print("\n--- Logic (Syllogism) ---")
    plot_trajectory_2d(logic, "(Logic)", "trajectory_2d_logic.html")

    # All prompts together
    all_names = list(PROMPTS.keys()) + list(EXTRA_PROMPTS.keys())
    print("\n--- All Prompts Combined ---")
    plot_trajectory_2d(all_names, "(All Prompts)", "trajectory_2d_all.html")
    plot_trajectory_3d(all_names, "(All Prompts)", "trajectory_3d_all.html")

    # t-SNE token visualization
    print("\n--- t-SNE Token Clusters ---")
    plot_tsne_all_tokens(primary, layer=-1, filename="tsne_tokens_final_layer.html")
    plot_tsne_all_tokens(primary, layer=0, filename="tsne_tokens_layer0.html")
    plot_tsne_all_tokens(primary, layer=5, filename="tsne_tokens_layer5.html")

    # Statistics
    print("\n--- Path Length Statistics ---")
    for pairs, label in [(primary, "Arithmetic"), (multi, "Subtraction"), (logic, "Logic")]:
        stats = compute_path_stats(pairs)
        print(f"\n{label}:")
        for name, s in stats.items():
            print(f"  {name}: path={s['path_length_pca2d']:.2f}, "
                  f"straight={s['start_end_distance']:.2f}, "
                  f"tortuosity={s['tortuosity']:.2f}, "
                  f"tokens={s['n_tokens']}")

    # Save stats as JSON
    all_stats = {}
    for pairs, label in [(primary, "Arithmetic"), (multi, "Subtraction"), (logic, "Logic")]:
        all_stats[label] = compute_path_stats(pairs)

    stats_path = os.path.join(OUTPUT_DIR, "path_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved: {stats_path}")

    print("\n" + "=" * 60)
    print("All visualizations generated in:", OUTPUT_DIR)
    print("=" * 60)
