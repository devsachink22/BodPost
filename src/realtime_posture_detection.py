import os
import json
import cv2
import joblib
import numpy as np
import mediapipe as mp

# Detect whether running inside GitHub Codespaces
if "CODESPACE_NAME" in os.environ:
    # GitHub Codespaces Unix-style paths
    PROJECT_ROOT = "/workspaces/BodPost"
else:
    # Local Windows machine
    PROJECT_ROOT = "E:/MSCS/1st Semester/Intro to AI (52560)/Project Work"

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "posture_model.pkl")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "posture_feature_columns.json")

# Load trained model and feature column order
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(FEATURE_COLS_PATH):
    raise FileNotFoundError(f"Feature columns JSON not found: {FEATURE_COLS_PATH}")

print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Loading feature column names from: {FEATURE_COLS_PATH}")
with open(FEATURE_COLS_PATH, "r") as f:
    feature_cols = json.load(f)


# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Mapping from joint name (prefix in your CSV columns) to MediaPipe index
NAME_TO_IDX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

def extract_features_from_landmarks(landmarks, feature_cols):
    """
    Convert MediaPipe landmarks to a feature vector that matches
    the training CSV column order stored in feature_cols.
    feature_cols looks like:
    ['nose_x', 'nose_y', 'nose_v', 'left_eye_inner_x', ..., 'right_foot_index_v']
    """
    features = []
    for col in feature_cols:
        joint, coord = col.rsplit("_", 1)  # e.g. "nose_x" -> ("nose", "x")
        idx = NAME_TO_IDX.get(joint, None)

        if idx is None:
            # If something weird sneaks into the feature list, just append 0
            features.append(0.0)
            continue

        lm = landmarks[idx]

        if coord == "x":
            value = lm.x
        elif coord == "y":
            value = lm.y
        elif coord == "v":
            value = lm.visibility
        else:
            value = 0.0

        features.append(value)

    return np.array(features, dtype=np.float32).reshape(1, -1)

# Real-time webcam loop
def main():
    print("Starting webcam for real-time posture detection...")
    print("Press 'q' to quit.")

    cap = cv2.VideoCapture(0)  # 0 = default camera

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera or device settings.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        status_text = "No person detected"
        color = (0, 255, 255)  # Yellow

        if results.pose_landmarks:
            # Draw pose skeleton & bounding box
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Build feature vector
            landmarks = results.pose_landmarks.landmark
            features = extract_features_from_landmarks(landmarks, feature_cols)

            # Predict using your trained model
            pred = model.predict(features)[0]

            # IMPORTANT: Your dataset uses:
            # label = 0 -> CORRECT posture
            # label = 1 -> INCORRECT posture
            if pred == 0:
                status_text = "Posture: CORRECT"
                color = (0, 255, 0)  # Green
            else:
                status_text = "Posture: INCORRECT"
                color = (0, 0, 255)  # Red

        # Show status text on the frame
        cv2.putText(
            frame,
            status_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA
        )

        # Display the result
        cv2.imshow("Real-time Posture Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed. Goodbye!")

if __name__ == "__main__":
    main()
