import sys
import os

sys.path.append(
    os.path.abspath("..")
)

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from collections import Counter

from model.ctrgcn import Model

# ==========================================
# CONFIG
# ==========================================

MODEL_PATH = (
    "../work_dir/mmasd/ctrgcn_motion/"
    "runs-30-4440.pt"
)

SKELETON_FOLDER =r'C:\Users\IIITA\Desktop\anchal\3d\Dataset_FINAL-20260429T093408Z-3-001\Dataset_FINAL\3D skeleton\CTR-GCN-main\Test\mediapipe_output'

WINDOW_SIZE = 30

STRIDE = 15

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
    graph_args={'labeling_mode': 'spatial'}
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
# LOAD SKELETONS
# ==========================================

print("\nLoading skeletons...")

files = sorted(
    os.listdir(SKELETON_FOLDER)
)

frames = []

for file in files:

    if file.endswith(".npz"):

        path = os.path.join(
            SKELETON_FOLDER,
            file
        )

        data = np.load(path)

        coords = data["coordinates"]

        coords = coords[0]

        frames.append(coords)

sequence = np.array(frames)

print(
    "\nSequence shape:",
    sequence.shape
)

# ==========================================
# ANALYSIS
# ==========================================

all_predictions = []

clap_windows = 0

total_frames = sequence.shape[0]

for start in range(
    0,
    total_frames - WINDOW_SIZE,
    STRIDE
):

    end = start + WINDOW_SIZE

    chunk = sequence[start:end]

    # ======================================
    # PREPROCESSING
    # ======================================

    chunk = chunk - chunk[:, 0:1, :]

    scale = np.max(
        np.linalg.norm(chunk, axis=2)
    )

    chunk = chunk / (scale + 1e-6)

    chunk[:, :, 2] *= 0.5

    # ======================================
    # FORMAT
    # ======================================

    chunk = chunk.transpose(
        2,
        0,
        1
    )

    chunk = np.expand_dims(
        chunk,
        axis=0
    )

    chunk = np.expand_dims(
        chunk,
        axis=-1
    )

    data = torch.tensor(
        chunk,
        dtype=torch.float32
    ).to(DEVICE)

    # ======================================
    # MODEL INFERENCE
    # ======================================

    with torch.no_grad():

        action_out, _, _ = model(data)

        action_prob = F.softmax(
            action_out,
            dim=1
        )

        pred_class = torch.argmax(
            action_prob,
            dim=1
        ).item()

    all_predictions.append(
        pred_class
    )

    # ======================================
    # WRIST ANALYSIS
    # ======================================

    LEFT_WRIST = 5
    RIGHT_WRIST = 6

    wrist_distances = []

    for frame in sequence[start:end]:

        left = frame[LEFT_WRIST]

        right = frame[RIGHT_WRIST]

        dist = np.linalg.norm(
            left - right
        )

        wrist_distances.append(dist)

    wrist_distances = np.array(
        wrist_distances
    )

    clap_frames = np.sum(
        wrist_distances < 0.15
    )

    if clap_frames > 5:

        clap_windows += 1

# ==========================================
# TOP ACTIVITIES
# ==========================================

counter = Counter(
    all_predictions
)

top3 = counter.most_common(3)

top_activities = []

for cls, count in top3:

    top_activities.append(
        class_names[cls]
    )

# ==========================================
# MOTION FEATURES
# ==========================================

motion_features = []

if clap_windows > 0:

    motion_features.append(
        "Clapping"
    )

if clap_windows > 3:

    motion_features.append(
        "Repetitive hand motion"
    )

drumming_count = counter.get(3, 0)

if drumming_count > 2:

    motion_features.append(
        "Drumming-like motion"
    )

if len(motion_features) == 0:

    motion_features.append(
        "No strong motion features"
    )

# ==========================================
# CREATE TABLE
# ==========================================

summary = {

    "Top Activities": top_activities,

    "Detected Motion Features": [

        ", ".join(motion_features)

    ] * len(top_activities)
}

df = pd.DataFrame(summary)

# ==========================================
# PRINT
# ==========================================

print("\n===================================")
print("BEHAVIOR SUMMARY TABLE")
print("===================================")

print(df)

# ==========================================
# SAVE
# ==========================================

os.makedirs(
    "../paper_figures",
    exist_ok=True
)

df.to_csv(
    "../paper_figures/behavior_summary.csv",
    index=False
)

print(
    "\n✅ Behavior summary table saved!"
)