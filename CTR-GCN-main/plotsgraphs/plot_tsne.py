import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# ===============================
# LOAD EMBEDDINGS
# ===============================

embeddings = np.load(
    "./results/embeddings/test_embeddings.npy"
)

labels = np.load(
    "./results/embeddings/test_labels.npy"
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
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

emb_2d = tsne.fit_transform(embeddings)
plt.figure(figsize=(12,10))

for i in range(len(class_names)):

    idx = labels == i

    plt.scatter(
        emb_2d[idx, 0],
        emb_2d[idx, 1],
        label=class_names[i],
        alpha=0.7
    )

plt.legend()
plt.title("t-SNE Embedding Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.tight_layout()

os.makedirs(
    "./paper_figures",
    exist_ok=True
)

plt.savefig(
    "./paper_figures/tsne_embeddings.png",
    dpi=300
)

plt.show()

print("\n✅ t-SNE saved!")