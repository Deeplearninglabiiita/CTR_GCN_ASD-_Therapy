import sys
import os

sys.path.append(
    os.path.abspath("..")
)

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from model.ctrgcn import Model

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = (
    "../work_dir/mmasd/ctrgcn_motion/"
    "runs-30-4440.pt"
)

DATA_PATH = (
    "../../processed_data/splits/X_test.npy"
)

LABEL_PATH = (
    "../../processed_data/splits/y_test.npy"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================
# CLASS NAMES
# ==========================================

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

# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading model...")

model = Model(

    num_class=11,
    num_point=24,
    num_person=1,

    graph='graph.mmasd.Graph',

    graph_args={
        'labeling_mode': 'spatial'
    }
)

weights = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

weights = {

    k.replace("module.", ""): v
    for k, v in weights.items()
}

model.load_state_dict(weights)

model = model.to(DEVICE)

model.eval()

print("✅ Model loaded!")

# ==========================================
# LOAD DATA
# ==========================================

X = np.load(DATA_PATH)

y = np.load(LABEL_PATH)

print(
    "\nData shape:",
    X.shape
)

# ==========================================
# SMALL SUBSET FOR VISUALIZATION
# ==========================================

X = X[:200]

y = y[:200]

# ==========================================
# EXTRACT EMBEDDINGS
# ==========================================

embeddings = []

print("\nExtracting embeddings...")

with torch.no_grad():

    for i in range(len(X)):

        sample = X[i:i+1]

        data = torch.tensor(
            sample,
            dtype=torch.float32
        ).to(DEVICE)

        model(data)

        emb = (
            model.embedding
            .cpu()
            .numpy()
            .squeeze()
        )

        embeddings.append(emb)

embeddings = np.array(
    embeddings
)

print(
    "Embedding shape:",
    embeddings.shape
)

# ==========================================
# TSNE
# ==========================================

print("\nRunning t-SNE...")

tsne = TSNE(

    n_components=2,

    perplexity=30,

    random_state=42
)

emb_2d = tsne.fit_transform(
    embeddings
)

print(
    "t-SNE shape:",
    emb_2d.shape
)

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(10,8))

for cls in np.unique(y):

    idx = y == cls

    plt.scatter(

        emb_2d[idx, 0],
        emb_2d[idx, 1],

        label=class_names[cls],

        alpha=0.7
    )

plt.legend(
    bbox_to_anchor=(1.05,1),
    loc='upper left'
)

plt.title(
    "t-SNE Visualization of Learned Embeddings",
    fontsize=14
)

plt.xlabel("t-SNE Dimension 1")

plt.ylabel("t-SNE Dimension 2")

plt.tight_layout()

# ==========================================
# SAVE
# ==========================================

os.makedirs(
    "../paper_figures",
    exist_ok=True
)

plt.savefig(

    "../paper_figures/tsne_embeddings_test.png",

    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(
    "\n✅ t-SNE visualization saved!"
)