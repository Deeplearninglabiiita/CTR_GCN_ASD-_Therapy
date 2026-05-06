import os
import sys

import numpy as np
import torch

sys.path.append(
    os.path.abspath(".")
)

from model.ctrgcn import Model

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = (
    "./work_dir/mmasd/ctrgcn_motion/"
    "runs-30-4440.pt"
)

DATA_PATH = (
    "../processed_data/splits/X_test.npy"
)

SAVE_DIR = "./results/attention"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

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

print("\nLoading test data...")

X = np.load(DATA_PATH)

print(
    "Data shape:",
    X.shape
)

# ==========================================
# TAKE SMALL BATCH
# ==========================================

X = X[:16]

data = torch.tensor(
    X,
    dtype=torch.float32
).to(DEVICE)

# ==========================================
# FORWARD PASS
# ==========================================

print("\nExtracting attention maps...")

with torch.no_grad():

    _, _, _, attention = model(
        data,
        return_attention=True
    )

attention = (
    attention
    .cpu()
    .numpy()
)

print(
    "Attention shape:",
    attention.shape
)

# ==========================================
# SAVE
# ==========================================

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

save_path = os.path.join(
    SAVE_DIR,
    "attention_maps.npy"
)

np.save(
    save_path,
    attention
)

print(
    f"\n✅ Saved to:\n{save_path}"
)