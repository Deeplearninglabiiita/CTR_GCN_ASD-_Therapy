import os
import numpy as np
from tqdm import tqdm

# ===============================
# 🔧 CONFIG
# ===============================
DATASET_PATH = "ROMP_3D_Coordinates"
SAVE_PATH = "processed_data"

WINDOW = 100   # number of frames

os.makedirs(SAVE_PATH, exist_ok=True)

# ===============================
# ADOS / CLINICAL LABELS
# ===============================

severity_map = {

    20274: 3,
    20313: 3,
    20383: 3,
    20453: 3,
    20564: 3,
    20583: 3,
    20594: 3,
    20613: 3,
    20626: 3,

    40013: 3,
    40074: 2,
    40104: 3,
    40143: 3,
    40173: 3,

    40323: 3,
    40475: 3,
    40484: 3,
    40493: 3,

    40504: 3,
    40513: 3,
    40533: 2,
    40543: 3,
    40555: 3,
    40573: 2,

    40614: 3,
    40665: 3,
    40683: 3,

    40704: 3,
    40714: 3,
    40723: 2,
    40743: 3,
    40753: 3,
    40794: 3,

    40803: 2,
    40023: 2
}


rrb_map = {

    20274: 3,
    20313: 5,
    20383: 3,
    20453: 3,
    20564: 4,
    20583: 3,
    20594: 3,
    20613: 4,
    20626: 3,

    40013: 14,
    40074: 3,
    40104: 6,
    40143: 8,
    40173: 4,

    40323: 25,
    40475: 7,
    40484: 8,
    40493: 7,

    40504: 8,
    40513: 6,
    40533: 2,
    40543: 5,
    40555: 8,
    40573: 1,

    40614: 6,
    40665: 6,
    40683: 2,

    40704: 14,
    40714: 5,
    40723: 4,
    40743: 3,
    40753: 2,
    40794: 23,

    40803: 1,
    40023: 3
}

# ===============================
# LABEL MAP
# ===============================
classes = sorted(os.listdir(DATASET_PATH))
label_map = {cls: i for i, cls in enumerate(classes)}

print("Classes:", label_map)

X = []
y = []

severity_labels = []
rrb_labels = []

sample_names = []
# ===============================
# LOAD DATASET
# ===============================
for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)

    for subject in tqdm(os.listdir(cls_path), desc=f"Processing {cls}"):

        subject_path = os.path.join(cls_path, subject)
        # Extract subject ID
        subject_id = int(subject.split('_')[1])
        # Skip subjects without ADOS labels

        if not os.path.isdir(subject_path):
            continue

        frames = []

        files = sorted(os.listdir(subject_path))

        for file in files:
            if file.endswith(".npz"):

                file_path = os.path.join(subject_path, file)

                try:
                    data = np.load(file_path)

                    # ✅ CORRECT KEY
                    coords = data["coordinates"]  # (3, 24, 3)

                    # ✅ TAKE BEST SKELETON
                    coords = coords[0]  # (24, 3)

                    frames.append(coords)

                except Exception as e:
                    print("Error:", file_path, e)

        if len(frames) == 0:
            continue

        sequence = np.array(frames)  # (T, 24, 3)

        # =========================
        # NORMALIZATION
        # =========================
        sequence = sequence - sequence[:, 0:1, :]  # root-relative

        scale = np.max(np.linalg.norm(sequence, axis=2))
        sequence = sequence / (scale + 1e-6)

        sequence[:, :, 2] *= 0.5  # reduce z-noise

        # =========================
        # FIX LENGTH
        # =========================
        if sequence.shape[0] < WINDOW:
            pad = np.zeros((WINDOW - sequence.shape[0], 24, 3))
            sequence = np.concatenate([sequence, pad], axis=0)
        else:
            sequence = sequence[:WINDOW]

        X.append(sequence)

        y.append(label_map[cls])

        # Severity label
        severity = severity_map.get(
            subject_id,
            -1
        )

        severity_labels.append(severity)
        # RRB grouping
        rrb_score = rrb_map.get(
            subject_id,
            -1
        )

        if rrb_score == -1:

            rrb_class = -1

        elif rrb_score <= 3:

            rrb_class = 0

        elif rrb_score <= 7:

            rrb_class = 1

        else:

            rrb_class = 2

        rrb_labels.append(rrb_class)

        # Save sample name
        sample_names.append(subject)

# ===============================
# CONVERT TO NUMPY
# ===============================
X = np.array(X)
y = np.array(y)
severity_labels = np.array(severity_labels)
rrb_labels = np.array(rrb_labels)
sample_names = np.array(sample_names)

print("\n===== FINAL DATA =====")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ===============================
# CONVERT TO CTR-GCN FORMAT
# ===============================
# (N, T, V, C) → (N, C, T, V, M)

X = X.transpose(0, 3, 1, 2)   # (N, 3, T, 24)
X = np.expand_dims(X, -1)     # (N, 3, T, 24, 1)

print("\n===== CTR-GCN INPUT =====")
print("X shape:", X.shape)

# ===============================
# SAVE
# ===============================
np.save(os.path.join(SAVE_PATH, "X.npy"), X)

np.save(os.path.join(SAVE_PATH, "y.npy"), y)

np.save(
    os.path.join(SAVE_PATH, "severity.npy"),
    severity_labels
)

np.save(
    os.path.join(SAVE_PATH, "rrb.npy"),
    rrb_labels
)

np.save(
    os.path.join(SAVE_PATH, "sample_names.npy"),
    sample_names
)
print("\n✅ Dataset saved to:", SAVE_PATH)

# ===============================
# CLASS WEIGHTS
# ===============================

class_counts = np.bincount(
    y,
    minlength=len(classes)
)

class_weights = 1.0 / class_counts

class_weights = (
    class_weights /
    class_weights.sum()
)

np.save(
    os.path.join(
        SAVE_PATH,
        "class_weights.npy"
    ),
    class_weights
)

print("\nClass counts:")
print(class_counts)

print("\nClass weights:")
print(class_weights)