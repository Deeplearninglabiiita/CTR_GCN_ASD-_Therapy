import sys
import os

sys.path.append(
    os.path.abspath("..")
)

import numpy as np
import torch
import torch.nn.functional as F

from collections import Counter
from model.ctrgcn import Model

MODEL_PATH = (
    "../work_dir/mmasd/ctrgcn_motion/"
    "runs-30-4440.pt"
)

SKELETON_FOLDER = "mediapipe_output"

WINDOW_SIZE = 30

STRIDE = 15

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class_names = [
    "Arm Swing",
    "Body Swing",
    "Chest Expansion",
    "Drumming",
    "Frog Pose",
    "Maracas Forward",
    "Maracas Shaking",
    "Sing and Clap",
    "Squat",
    "Tree Pose",
    "Twist Pose"
]

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

print("Model loaded!")

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

print("\nAnalyzing video...")
total_frames = sequence.shape[0]
all_predictions = []
clap_windows = 0
for start in range(
    0,
    total_frames - WINDOW_SIZE,
    STRIDE
):

    end = start + WINDOW_SIZE
    chunk = sequence[start:end]
    chunk = chunk - chunk[:, 0:1, :]

    scale = np.max(
        np.linalg.norm(chunk, axis=2)
    )
    chunk = chunk / (scale + 1e-6)
    chunk[:, :, 2] *= 0.5
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

    with torch.no_grad():

        action_out, _, _ = model(data)

        action_prob = F.softmax(
            action_out,
            dim=1
        )

        topk_probs, topk_classes = torch.topk(
            action_prob,
            k=3
        )
    for i in range(3):

        cls = (
            topk_classes[0][i]
            .item()
        )

        all_predictions.append(cls)

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


counter = Counter(
    all_predictions
)

top3 = counter.most_common(3)

print("\n===================================")
print("VIDEO BEHAVIOR SUMMARY")
print("===================================")

print("\nTop Activities:")

for i, (cls, count) in enumerate(top3):

    print(
        f"{i+1}. "
        f"{class_names[cls]}"
    )

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
if drumming_count > 5:
    motion_features.append(
        "Drumming-like motion"
    )
print("\nDetected Motion Features:")
if len(motion_features) == 0:
    print(
        "No strong motion features detected"
    )
else:
    for feature in motion_features:
        print(
            f"+ {feature}"
        )
print("\n✅ Analysis complete!")

# ==========================================
# WRIST DISTANCE VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt

all_wrist_distances = []

LEFT_WRIST = 5
RIGHT_WRIST = 6

for frame in sequence:

    left = frame[LEFT_WRIST]

    right = frame[RIGHT_WRIST]

    dist = np.linalg.norm(
        left - right
    )

    all_wrist_distances.append(dist)

all_wrist_distances = np.array(
    all_wrist_distances
)

plt.figure(figsize=(12,5))

plt.plot(
    all_wrist_distances,
    linewidth=2
)

# Threshold line
plt.axhline(
    y=0.15,
    linestyle='--'
)

plt.xlabel(
    "Frame Number",
    fontsize=12
)

plt.ylabel(
    "Wrist Distance",
    fontsize=12
)

plt.title(
    "Inter-Wrist Distance Over Time",
    fontsize=14
)

plt.grid(True)
os.makedirs(
    "../paper_figures",
    exist_ok=True
)

plt.savefig(
    "../paper_figures/wrist_distance_plot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(
    "\n✅ Wrist distance graph saved!"
)