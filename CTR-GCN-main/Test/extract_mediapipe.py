import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================

VIDEO_PATH = "v2.mp4"

OUTPUT_DIR = "./mediapipe_output"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# MEDIAPIPE
# ==========================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("\nTotal frames:", frame_count)

frame_id = 0

# ==========================================
# JOINT SELECTION
# ==========================================

selected_joints = [
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
]

# ==========================================
# PROCESS VIDEO
# ==========================================

for _ in tqdm(range(frame_count)):

    ret, frame = cap.read()

    if not ret:
        break

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(image)

    if results.pose_landmarks:

        coords = []

        for idx in selected_joints:

            lm = results.pose_landmarks.landmark[idx]

            coords.append([
                lm.x,
                lm.y,
                lm.z
            ])

        coords = np.array(coords)

        # ==================================
        # PAD TO 24 JOINTS
        # ==================================

        if coords.shape[0] < 24:

            pad = np.zeros(
                (
                    24 - coords.shape[0],
                    3
                )
            )

            coords = np.concatenate(
                [coords, pad],
                axis=0
            )

        coords = np.expand_dims(
            coords,
            axis=0
        )

        save_path = os.path.join(
            OUTPUT_DIR,
            f"{frame_id:06d}.npz"
        )

        np.savez(
            save_path,
            coordinates=coords
        )

        frame_id += 1

cap.release()

print("\n================================")
print("MEDIAPIPE EXTRACTION COMPLETE")
print("================================")

print("\nSaved to:")
print(OUTPUT_DIR)