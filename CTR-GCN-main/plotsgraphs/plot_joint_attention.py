import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# LOAD ATTENTION MAP
# ==========================================

attention = np.load(
    r"C:\Users\IIITA\Desktop\anchal\3d\Dataset_FINAL-20260429T093408Z-3-001\Dataset_FINAL\3D skeleton\CTR-GCN-main\results\attention\attention_maps.npy"
)

print(
    "\nOriginal shape:",
    attention.shape
)

# ==========================================
# REDUCE DIMENSIONS
# ==========================================

# Expected:
# (N, C, T, V, M)

attention = attention.mean(axis=0)

print(
    "After batch mean:",
    attention.shape
)

# Mean over channels
attention = attention.mean(axis=0)

print(
    "After channel mean:",
    attention.shape
)

# Mean over time
attention = attention.mean(axis=0)

print(
    "After temporal mean:",
    attention.shape
)

# Remove person dimension
attention = attention.squeeze()

print(
    "Final shape:",
    attention.shape
)

# ==========================================
# JOINT LABELS
# ==========================================

joint_names = [

    "Pelvis",
    "L-Hip",
    "R-Hip",
    "Spine1",
    "L-Knee",
    "R-Knee",
    "Spine2",
    "L-Ankle",
    "R-Ankle",
    "Spine3",
    "L-Foot",
    "R-Foot",
    "Neck",
    "L-Collar",
    "R-Collar",
    "Head",
    "L-Shoulder",
    "R-Shoulder",
    "L-Elbow",
    "R-Elbow",
    "L-Wrist",
    "R-Wrist",
    "L-Hand",
    "R-Hand"
]

# ==========================================
# RESHAPE FOR HEATMAP
# ==========================================

attention = attention.reshape(1, -1)

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(16,3))

sns.heatmap(

    attention,

    cmap="viridis",

    xticklabels=joint_names,

    yticklabels=[],

    cbar=True
)

plt.xticks(

    rotation=45,
    ha='right'
)

plt.title(

    "Joint Attention Importance",
    fontsize=14
)

plt.tight_layout()

# ==========================================
# SAVE
# ==========================================

os.makedirs(
    "../paper_figures",
    exist_ok=True
)

plt.savefig(

    "../paper_figures/joint_attention.png",

    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(
    "\n✅ Joint attention visualization saved!"
)