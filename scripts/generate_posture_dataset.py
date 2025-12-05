import os
import cv2
import mediapipe as mp
import pandas as pd

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.6
)

# Landmark names for better CSV column names
LANDMARKS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
    'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
    'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
    'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
    'right_heel', 'left_foot_index', 'right_foot_index'
]

def extract_keypoints(image):
    # Extract 33 pose keypoints using MediaPipe
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.visibility])
    return keypoints

def process_images(INPUT_FOLDER, OUTPUT_CSV):
    rows = []
    print("Processing images...")
    for label_name in os.listdir(INPUT_FOLDER):
        label_path = os.path.join(INPUT_FOLDER, label_name)
        if not os.path.isdir(label_path):
            continue

        # Map folder name to numeric label
        if label_name.lower() == "correct":
            target = 0
        elif label_name.lower() == "incorrect":
            target = 1
        else:
            print(f"Skipping unknown folder: {label_name}")
            continue

        print(f"Processing label folder: {label_name} (label={target})")

        for file in os.listdir(label_path):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(label_path, file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[SKIPPED] Cannot read image {file}")
                continue

            keypoints = extract_keypoints(image)
            if keypoints is None:
                print(f"[SKIPPED] No person detected in {file}")
                continue

            # Add image path + keypoints + label
            rows.append([img_path] + keypoints + [target])

    if not rows:
        print("No data collected. Dataset is empty.")
        return

    # Create column names
    columns = ["image_path"]
    for lm in LANDMARKS:
        columns += [f"{lm}_x", f"{lm}_y", f"{lm}_v"]
    columns.append("label")

    # Save to CSV
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Dataset generated successfully:", OUTPUT_CSV)

if __name__ == "__main__":
    PROJECT_ROOT = "E:/MSCS/1st Semester/Intro to AI (52560)/Project Work"
    INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data")
    OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/posture_dataset_mediapipe.csv")

    process_images(INPUT_FOLDER, OUTPUT_CSV)
    print("Completed.")