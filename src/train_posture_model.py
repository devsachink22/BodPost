import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Paths (adjust if your folder structure is different)
if "CODESPACE_NAME" in os.environ:
    # GitHub Codespaces Unix-style paths
    PROJECT_ROOT = "/workspaces/BodPost"
else:
    # Local Windows machine
    PROJECT_ROOT = "E:/MSCS/1st Semester/Intro to AI (52560)/Project Work"

DATA_CSV = os.path.join(PROJECT_ROOT, "data", "posture_dataset_mediapipe.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "posture_model.pkl")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "posture_feature_columns.json")

# 1. Load dataset
print(f"Loading data from: {DATA_CSV}")
df = pd.read_csv(DATA_CSV)

if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

# 2. Feature columns: all except image_path and label
feature_cols = [c for c in df.columns if c not in ["image_path", "label"]]
X = df[feature_cols].values
y = df["label"].values  # 0/1 (you decide which is correct/incorrect)

print(f"Dataset shape: {df.shape}")
print(f"Number of features: {len(feature_cols)}")

# 3. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])

# 4. Preprocessing and Classification
#    - StandardScaler (optional but safe)
#    - RandomForestClassifier (tree-based, robust)
pipeline = Pipeline([
    ("scaler", StandardScaler()),          # will scale all numeric features
    ("clf", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

# 5. Hyperparameter optimization (GridSearchCV)
#    Keep grid moderate so it's not too slow.
param_grid = {
    "clf__n_estimators": [150, 300, 500],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=2
)

print("Starting grid search...")
grid_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# 6. Evaluate on the test set
y_pred = best_model.predict(X_test)

print("\nTest set performance:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["class_0", "class_1"]  # correct-incorrect
))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save model + feature column order
print(f"\nSaving trained model to: {MODEL_PATH}")
joblib.dump(best_model, MODEL_PATH)

print(f"Saving feature column names to: {FEATURE_COLS_PATH}")
with open(FEATURE_COLS_PATH, "w") as f:
    json.dump(feature_cols, f)

print("Done! Your posture classification model is ready.")
