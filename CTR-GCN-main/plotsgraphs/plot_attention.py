import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


temporal = np.load(
    "./results/attention/temporal_attention.npy"
)

joint = np.load(
    "./results/attention/joint_attention.npy"
)

print("Temporal shape:", temporal.shape)
print("Joint shape:", joint.shape)


# Average across batch dimension
temporal = temporal.mean(axis=0)

joint = joint.mean(axis=0)

# Flatten safely
temporal = temporal.squeeze()

joint = joint.squeeze()

# If still multidimensional
if temporal.ndim > 1:
    temporal = temporal.mean(axis=0)

if joint.ndim > 1:
    joint = joint.mean(axis=0)


plt.figure(figsize=(12,4))

sns.heatmap(
    temporal[np.newaxis, :],
    cmap="viridis"
)

plt.title("Temporal Attention Map")
plt.xlabel("Frame Index")

plt.tight_layout()

os.makedirs(
    "./paper_figures",
    exist_ok=True
)

plt.savefig(
    "./paper_figures/temporal_attention.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(10,4))

sns.heatmap(
    joint[np.newaxis, :],
    cmap="magma"
)

plt.title("Joint Attention Map")
plt.xlabel("Joint Index")

plt.tight_layout()

plt.savefig(
    "./paper_figures/joint_attention.png",
    dpi=300
)

plt.show()

print("\nAttention maps saved!")