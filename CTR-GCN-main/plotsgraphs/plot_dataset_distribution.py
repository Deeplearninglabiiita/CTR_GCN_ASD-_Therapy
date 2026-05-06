import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# LOAD LABELS
# ==========================================

labels = np.load(
   r"C:\Users\IIITA\Desktop\anchal\3d\Dataset_FINAL-20260429T093408Z-3-001\Dataset_FINAL\3D skeleton\processed_data\splits\y_train.npy"
)


class_names = [
    "Arm Swing",
    "Body Swing",
    "Chest Expansion",
    "Drumming",
    "Frog Pose",
    "Maracas Forward",
    "Maracas Shaking",
    "Sing & Clap",
    "Squat",
    "Tree Pose",
    "Twist Pose"
]
counts = []

for i in range(len(class_names)):

    count = np.sum(labels == i)

    counts.append(count)

    print(
        f"{class_names[i]}: {count}"
    )


plt.figure(figsize=(12,6))

bars = plt.bar(
    class_names,
    counts
)

# Add values above bars
for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 2,
        str(int(height)),
        ha='center'
    )

plt.xticks(
    rotation=45,
    ha='right'
)

plt.ylabel(
    "Number of Samples",
    fontsize=12
)

plt.xlabel(
    "Activity Classes",
    fontsize=12
)

plt.title(
    "MMASD Dataset Class Distribution",
    fontsize=14
)

plt.tight_layout()

os.makedirs(
    "../paper_figures",
    exist_ok=True
)

plt.savefig(
    "../paper_figures/class_distribution.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(
    "\n✅ Dataset distribution graph saved!"
)