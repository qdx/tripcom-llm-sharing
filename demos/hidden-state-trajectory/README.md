# Hidden State Trajectory Visualization

Compares how Direct prompts and Chain-of-Thought (CoT) prompts traverse the representation space of a transformer (GPT-2 small).

## Key Insight

CoT prompts take a **longer path** through the model's hidden state space compared to direct prompts. This suggests CoT induces more computational work across layers — the model's internal representations change more dramatically when reasoning step-by-step.

## What It Does

1. Runs direct prompts (e.g., "What is 23*47?") and CoT prompts (e.g., "What is 23*47? Let's think step by step.") through GPT-2
2. Extracts hidden states at every layer using TransformerLens
3. Averages hidden states across tokens at each layer → one point per layer
4. Applies PCA / t-SNE to project these high-dimensional trajectories to 2D/3D
5. Plots trajectories + compares path lengths

## Output Visualizations

- `trajectory_pca_2d.png` — Joint PCA 2D trajectory plot for all prompts
- `trajectory_pca_3d.png` — 3D PCA view
- `trajectory_tsne_2d.png` — t-SNE 2D view
- `path_length_comparison.png` — Bar chart comparing trajectory lengths
- `heatmap_p*_*.png` — Per-token hidden state norm heatmaps

## Usage

```bash
cd demos/hidden-state-trajectory
source ../../venv/bin/activate
python hidden_state_trajectory.py
```

Output images are saved to `output/`.

## Dependencies

See `requirements.txt`. All included in the project venv.
