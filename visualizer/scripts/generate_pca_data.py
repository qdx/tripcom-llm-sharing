"""Generate PCA trajectory data for Demo 1 split-screen visualization.

Pre-computes hidden state trajectories through GPT-2 layers and saves as JSON
for the frontend visualizer to render without needing Python runtime.
"""

import json
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from sklearn.decomposition import PCA

OUTPUT_PATH = "../outputs/pca_trajectories.json"

def extract_trajectories(model, prompt: str, n_layers: int, d_model: int):
    """Run prompt through model and extract hidden states at each layer for each token."""
    tokens = model.to_str_tokens(prompt)
    _, cache = model.run_with_cache(prompt)
    
    hidden_states = torch.zeros(n_layers + 1, len(tokens), d_model)
    hidden_states[0] = cache["embed"][0]
    for layer in range(n_layers):
        act_name = get_act_name("resid_post", layer)
        hidden_states[layer + 1] = cache[act_name][0]
    
    return tokens, hidden_states


def compute_pca_data(model):
    """Compute PCA trajectories for the demo prompt."""
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    prompt = "The capital of France is"
    print(f"Processing prompt: '{prompt}'")
    
    tokens, hidden_states = extract_trajectories(model, prompt, n_layers, d_model)
    
    # PCA on all (layer, token) hidden states
    all_hidden = hidden_states.detach().numpy().reshape(-1, d_model)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_hidden)
    projected = projected.reshape(n_layers + 1, len(tokens), 2)
    
    # Build per-token trajectory data
    trajectories = []
    for t_idx, token_str in enumerate(tokens):
        points = []
        for layer in range(n_layers + 1):
            points.append({
                "layer": layer,
                "x": round(float(projected[layer, t_idx, 0]), 4),
                "y": round(float(projected[layer, t_idx, 1]), 4),
            })
        trajectories.append({
            "token": token_str.replace(" ", "·") if token_str.startswith(" ") else token_str,
            "tokenRaw": token_str,
            "index": t_idx,
            "points": points,
        })
    
    # Also compute Direct vs CoT comparison
    direct_prompt = "Q: What is 23 + 45?\nA: The answer is"
    cot_prompt = "Q: What is 23 + 45?\nA: Let me think step by step. 23 + 45 = 20 + 40 + 3 + 5 = 60 + 8 = 68. The answer is"
    
    print(f"Processing Direct prompt ({len(model.to_str_tokens(direct_prompt))} tokens)")
    print(f"Processing CoT prompt ({len(model.to_str_tokens(cot_prompt))} tokens)")
    
    _, cache_direct = model.run_with_cache(direct_prompt)
    _, cache_cot = model.run_with_cache(cot_prompt)
    
    def extract_last_token_trajectory(cache):
        traj = torch.zeros(n_layers + 1, d_model)
        traj[0] = cache["embed"][0, -1]
        for layer in range(n_layers):
            act_name = get_act_name("resid_post", layer)
            traj[layer + 1] = cache[act_name][0, -1]
        return traj
    
    traj_direct = extract_last_token_trajectory(cache_direct)
    traj_cot = extract_last_token_trajectory(cache_cot)
    
    # PCA on combined
    combined = torch.cat([traj_direct, traj_cot], dim=0).detach().numpy()
    pca2 = PCA(n_components=2)
    proj = pca2.fit_transform(combined)
    proj_direct = proj[:n_layers + 1]
    proj_cot = proj[n_layers + 1:]
    
    # Path lengths
    def path_length(traj_2d):
        diffs = np.diff(traj_2d, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    len_direct = path_length(proj_direct)
    len_cot = path_length(proj_cot)
    
    comparison = {
        "direct": {
            "prompt": direct_prompt,
            "tokenCount": len(model.to_str_tokens(direct_prompt)),
            "pathLength": round(len_direct, 2),
            "points": [{"layer": i, "x": round(float(proj_direct[i, 0]), 4), "y": round(float(proj_direct[i, 1]), 4)} for i in range(n_layers + 1)],
        },
        "cot": {
            "prompt": cot_prompt,
            "tokenCount": len(model.to_str_tokens(cot_prompt)),
            "pathLength": round(len_cot, 2),
            "points": [{"layer": i, "x": round(float(proj_cot[i, 0]), 4), "y": round(float(proj_cot[i, 1]), 4)} for i in range(n_layers + 1)],
        },
        "pathRatio": round(len_cot / len_direct, 2),
        "pcaVariance": [round(float(pca2.explained_variance_ratio_[0]), 4), round(float(pca2.explained_variance_ratio_[1]), 4)],
    }
    
    result = {
        "model": "gpt2-small",
        "nLayers": n_layers,
        "dModel": d_model,
        "tokenTrajectories": {
            "prompt": prompt,
            "tokens": [t["token"] for t in trajectories],
            "trajectories": trajectories,
            "pcaVariance": [round(float(pca.explained_variance_ratio_[0]), 4), round(float(pca.explained_variance_ratio_[1]), 4)],
        },
        "comparison": comparison,
    }
    
    return result


def main():
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model}-dim")
    
    data = compute_pca_data(model)
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved PCA data to {OUTPUT_PATH}")
    print(f"  Token trajectories: {len(data['tokenTrajectories']['trajectories'])} tokens × {data['nLayers'] + 1} layers")
    print(f"  Comparison: Direct path={data['comparison']['direct']['pathLength']}, CoT path={data['comparison']['cot']['pathLength']}")
    print(f"  Path ratio: {data['comparison']['pathRatio']}×")


if __name__ == "__main__":
    main()
