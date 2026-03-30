"""
Hidden State Trajectory Visualization: Direct vs Chain-of-Thought

Extracts hidden states from GPT-2 small for direct and CoT prompts,
reduces dimensionality via PCA and t-SNE, and plots trajectories
showing how CoT takes a longer/different path through representation space.

Part of Trip.com LLM Knowledge Sharing project.
"""

import os
import numpy as np
import torch
import transformer_lens
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "gpt2"  # GPT-2 small — fits on MacBook

# Prompt pairs: (direct, chain-of-thought)
PROMPT_PAIRS = [
    (
        "What is 23 * 47?",
        "What is 23 * 47? Let's think step by step."
    ),
    (
        "What is the capital of France?",
        "What is the capital of France? Let me reason through this carefully."
    ),
    (
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "If a train travels 60 mph for 2.5 hours, how far does it go? Let's break this down step by step."
    ),
]

# ─── Load Model ───────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME)
model.eval()
n_layers = model.cfg.n_layers
print(f"Model loaded: {n_layers} layers, {model.cfg.d_model}d hidden states")


def get_hidden_states(prompt: str) -> np.ndarray:
    """
    Run prompt through model and extract hidden states at each layer.
    Returns array of shape (n_layers+1, n_tokens, d_model).
    Layer 0 = embedding output, layers 1..n = transformer block outputs.
    """
    tokens = model.to_tokens(prompt)
    n_tokens = tokens.shape[1]
    
    # Run with cache to get all residual stream states
    _, cache = model.run_with_cache(tokens)
    
    # Collect residual stream at each layer
    # hook_embed gives us the initial embedding
    states = []
    states.append(cache["hook_embed"][0].detach().cpu().numpy())  # (n_tokens, d_model)
    
    for layer in range(n_layers):
        # resid_post is the residual stream after each layer
        key = f"blocks.{layer}.hook_resid_post"
        states.append(cache[key][0].detach().cpu().numpy())  # (n_tokens, d_model)
    
    return np.array(states)  # (n_layers+1, n_tokens, d_model)


def compute_layer_trajectory(states: np.ndarray, method: str = "last") -> np.ndarray:
    """
    Compute a single trajectory through representation space.
    
    Methods:
      - "last": Use the last token's hidden state at each layer (what the model
        uses for next-token prediction — most semantically meaningful).
      - "mean": Average across all tokens at each layer.
    
    Returns array of shape (n_layers+1, d_model).
    """
    if method == "last":
        return states[:, -1, :]  # last token at each layer
    else:
        return states.mean(axis=1)


def compute_trajectory_length(trajectory_2d: np.ndarray) -> float:
    """Compute total Euclidean path length of a 2D/3D trajectory."""
    diffs = np.diff(trajectory_2d, axis=0)
    return np.sum(np.sqrt(np.sum(diffs**2, axis=1)))


def plot_trajectories_pca(all_trajectories: dict, title: str, filename: str):
    """
    Apply PCA to all trajectories jointly, then plot 2D trajectories.
    all_trajectories: {label: (n_layers+1, d_model) array}
    """
    # Stack all trajectories for joint PCA
    labels = list(all_trajectories.keys())
    stacked = np.vstack([all_trajectories[k] for k in labels])
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(stacked)
    
    # Split back
    idx = 0
    reduced_dict = {}
    for label in labels:
        n = all_trajectories[label].shape[0]
        reduced_dict[label] = reduced[idx:idx+n]
        idx += n
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors_direct = cm.Blues(np.linspace(0.3, 1.0, len([l for l in labels if "Direct" in l])))
    colors_cot = cm.Reds(np.linspace(0.3, 1.0, len([l for l in labels if "CoT" in l])))
    
    di, ci = 0, 0
    for label in labels:
        traj = reduced_dict[label]
        path_len = compute_trajectory_length(traj)
        
        if "Direct" in label:
            color = colors_direct[di]
            di += 1
            linestyle = '-'
        else:
            color = colors_cot[ci]
            ci += 1
            linestyle = '--'
        
        ax.plot(traj[:, 0], traj[:, 1], linestyle, color=color, linewidth=2,
                label=f"{label} (len={path_len:.1f})", alpha=0.8)
        
        # Mark start (layer 0) and end
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker='o', s=100, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker='*', s=200, zorder=5)
        
        # Add layer numbers
        for i in range(0, len(traj), 2):
            ax.annotate(f'L{i}', (traj[i, 0], traj[i, 1]), fontsize=7,
                       color=color, alpha=0.7)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_trajectories_tsne(all_trajectories: dict, title: str, filename: str):
    """
    Apply t-SNE to all trajectories jointly, then plot.
    """
    labels = list(all_trajectories.keys())
    stacked = np.vstack([all_trajectories[k] for k in labels])
    
    perplexity = min(30, max(5, stacked.shape[0] - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(stacked)
    
    idx = 0
    reduced_dict = {}
    for label in labels:
        n = all_trajectories[label].shape[0]
        reduced_dict[label] = reduced[idx:idx+n]
        idx += n
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors_direct = cm.Blues(np.linspace(0.3, 1.0, len([l for l in labels if "Direct" in l])))
    colors_cot = cm.Reds(np.linspace(0.3, 1.0, len([l for l in labels if "CoT" in l])))
    
    di, ci = 0, 0
    for label in labels:
        traj = reduced_dict[label]
        path_len = compute_trajectory_length(traj)
        
        if "Direct" in label:
            color = colors_direct[di]
            di += 1
            linestyle = '-'
        else:
            color = colors_cot[ci]
            ci += 1
            linestyle = '--'
        
        ax.plot(traj[:, 0], traj[:, 1], linestyle, color=color, linewidth=2,
                label=f"{label} (len={path_len:.1f})", alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker='o', s=100, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker='*', s=200, zorder=5)
        
        for i in range(0, len(traj), 2):
            ax.annotate(f'L{i}', (traj[i, 0], traj[i, 1]), fontsize=7,
                       color=color, alpha=0.7)
    
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_3d_pca(all_trajectories: dict, title: str, filename: str):
    """3D PCA trajectory plot."""
    labels = list(all_trajectories.keys())
    stacked = np.vstack([all_trajectories[k] for k in labels])
    
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(stacked)
    
    idx = 0
    reduced_dict = {}
    for label in labels:
        n = all_trajectories[label].shape[0]
        reduced_dict[label] = reduced[idx:idx+n]
        idx += n
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors_direct = cm.Blues(np.linspace(0.3, 1.0, len([l for l in labels if "Direct" in l])))
    colors_cot = cm.Reds(np.linspace(0.3, 1.0, len([l for l in labels if "CoT" in l])))
    
    di, ci = 0, 0
    for label in labels:
        traj = reduced_dict[label]
        path_len = compute_trajectory_length(traj)
        
        if "Direct" in label:
            color = colors_direct[di]
            di += 1
            linestyle = '-'
        else:
            color = colors_cot[ci]
            ci += 1
            linestyle = '--'
        
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linestyle, color=color,
                linewidth=2, label=f"{label} (len={path_len:.1f})", alpha=0.8)
        ax.scatter(*traj[0], color=color, marker='o', s=100, zorder=5)
        ax.scatter(*traj[-1], color=color, marker='*', s=200, zorder=5)
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_path_length_comparison(path_lengths: dict, filename: str):
    """Bar chart comparing trajectory path lengths for direct vs CoT."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    prompt_names = []
    direct_lengths = []
    cot_lengths = []
    
    for name, (d_len, c_len) in path_lengths.items():
        prompt_names.append(name)
        direct_lengths.append(d_len)
        cot_lengths.append(c_len)
    
    x = np.arange(len(prompt_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, direct_lengths, width, label='Direct', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, cot_lengths, width, label='CoT', color='indianred', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Prompt')
    ax.set_ylabel('Trajectory Path Length (PCA 2D)')
    ax.set_title('Hidden State Trajectory Length: Direct vs Chain-of-Thought')
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_per_token_heatmap(states: np.ndarray, prompt: str, label: str, filename: str):
    """
    Heatmap showing hidden state norms per (layer, token) — reveals
    where the model 'thinks hardest'.
    """
    # states shape: (n_layers+1, n_tokens, d_model)
    norms = np.linalg.norm(states, axis=2)  # (n_layers+1, n_tokens)
    
    tokens = model.to_tokens(prompt)
    token_strs = [model.to_string(tokens[0, i:i+1]).strip() for i in range(tokens.shape[1])]
    # Truncate long tokens for display
    token_strs = [t[:12] for t in token_strs]
    
    fig, ax = plt.subplots(figsize=(max(8, len(token_strs) * 0.8), 8))
    im = ax.imshow(norms, aspect='auto', cmap='YlOrRd')
    
    ax.set_xlabel('Token')
    ax.set_ylabel('Layer')
    ax.set_title(f'Hidden State Norms: {label}')
    ax.set_xticks(range(len(token_strs)))
    ax.set_xticklabels(token_strs, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(norms.shape[0]))
    ax.set_yticklabels([f'L{i}' for i in range(norms.shape[0])])
    
    plt.colorbar(im, ax=ax, label='L2 Norm')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_trajectories = {}
    path_lengths_for_bar = {}
    
    for i, (direct_prompt, cot_prompt) in enumerate(PROMPT_PAIRS):
        prompt_label = f"Prompt {i+1}"
        print(f"\n{'='*60}")
        print(f"Processing {prompt_label}")
        print(f"  Direct: {direct_prompt}")
        print(f"  CoT:    {cot_prompt}")
        
        # Extract hidden states
        print("  Extracting hidden states (direct)...")
        direct_states = get_hidden_states(direct_prompt)
        print(f"    Shape: {direct_states.shape}")
        
        print("  Extracting hidden states (CoT)...")
        cot_states = get_hidden_states(cot_prompt)
        print(f"    Shape: {cot_states.shape}")
        
        # Compute layer trajectories using last token (most meaningful for generation)
        direct_traj = compute_layer_trajectory(direct_states, method="last")
        cot_traj = compute_layer_trajectory(cot_states, method="last")
        
        all_trajectories[f"{prompt_label} Direct"] = direct_traj
        all_trajectories[f"{prompt_label} CoT"] = cot_traj
        
        # Per-token heatmaps
        plot_per_token_heatmap(direct_states, direct_prompt,
                              f"{prompt_label} - Direct", f"heatmap_p{i+1}_direct.png")
        plot_per_token_heatmap(cot_states, cot_prompt,
                              f"{prompt_label} - CoT", f"heatmap_p{i+1}_cot.png")
    
    # ─── Joint visualizations ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating joint trajectory visualizations...")
    
    # PCA 2D
    plot_trajectories_pca(
        all_trajectories,
        "Hidden State Trajectories (PCA 2D): Direct vs CoT\n○ = Layer 0 (embedding), ★ = Final layer",
        "trajectory_pca_2d.png"
    )
    
    # PCA 3D
    plot_3d_pca(
        all_trajectories,
        "Hidden State Trajectories (PCA 3D): Direct vs CoT",
        "trajectory_pca_3d.png"
    )
    
    # t-SNE 2D
    plot_trajectories_tsne(
        all_trajectories,
        "Hidden State Trajectories (t-SNE 2D): Direct vs CoT\n○ = Layer 0, ★ = Final layer",
        "trajectory_tsne_2d.png"
    )
    
    # Path length comparison (using PCA 2D)
    stacked_all = np.vstack(list(all_trajectories.values()))
    pca_for_lengths = PCA(n_components=2)
    reduced_all = pca_for_lengths.fit_transform(stacked_all)
    
    idx = 0
    for i in range(len(PROMPT_PAIRS)):
        d_n = all_trajectories[f"Prompt {i+1} Direct"].shape[0]
        c_n = all_trajectories[f"Prompt {i+1} CoT"].shape[0]
        d_traj = reduced_all[idx:idx+d_n]
        idx += d_n
        c_traj = reduced_all[idx:idx+c_n]
        idx += c_n
        d_len = compute_trajectory_length(d_traj)
        c_len = compute_trajectory_length(c_traj)
        path_lengths_for_bar[f"Prompt {i+1}"] = (d_len, c_len)
        print(f"  Prompt {i+1}: Direct path={d_len:.2f}, CoT path={c_len:.2f}, ratio={c_len/d_len:.2f}x")
    
    plot_path_length_comparison(path_lengths_for_bar, "path_length_comparison.png")
    
    print(f"\n✅ All visualizations saved to {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")
