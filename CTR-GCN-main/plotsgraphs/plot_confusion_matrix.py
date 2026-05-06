import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


true_labels = np.load(
    "./results/confusion_matrix/true_labels.npy"
)

pred_labels = np.load(
    "./results/confusion_matrix/pred_labels.npy"
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


cm = confusion_matrix(
    true_labels,
    pred_labels
)

# Normalize row-wise
cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5
)

plt.xlabel(
    "Predicted Label",
    fontsize=14
)

plt.ylabel(
    "True Label",
    fontsize=14
)

plt.title(
    "Normalized Confusion Matrix",
    fontsize=16
)
plt.xticks(
    rotation=45,
    ha='right'
)
plt.yticks(
    rotation=0
)
plt.tight_layout()

os.makedirs(
    "./paper_figures",
    exist_ok=True
)

plt.savefig(
    "./paper_figures/confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\n✅ Confusion matrix saved!")