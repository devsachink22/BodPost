# BodPost: Real-Time Body Posture Detection System
BodPost is a machine-learning–powered system that detects correct vs incorrect human posture using MediaPipe keypoints and an optimized ML classifier. It supports:

• Dataset generation from images

• MediaPipe feature extraction

• Model training & evaluation

• Real-time webcam posture detection

• Modular, extensible project architecture

# BodPost: Project Structure
BodPost/

│── data/                     # Raw posture images grouped in class folders

│──   ├── correct/

│──   ├── ├── img1.jpg

│──   ├── ├── img2.jpg

│──   ├── incorrect/

│──   ├── ├── img1.jpg

│──   ├── ├── img2.jpg

│──   ├── labelled_data/            # Generated posture feature CSV

│── models/                   # Trained ML model + feature columns JSON

│──   ├── posture_feature_columns.json

│──   ├── posture_model.pkl

│── scripts/                  # Offline scripts (dataset prep, feature generation)

│──   ├── generate_posture_dataset.py

│── src/                      # Core source code

│──   ├── train_posture_model.py

│──   ├── realtime_posture_detection.py

│── documents/                # Proposal, notes, ppts, other docs

│── README.md                 # Project documentation

# Project Objective
To automatically classify correct vs incorrect posture by extracting 33 MediaPipe pose keypoints, computing engineered features, and training a machine-learning classifier optimized for accuracy and real-time performance.

# Technology Used
• Python 3.12.0

• OpenCV 4.12.0.88

• MediaPipe 0.10.21

• Pandas 2.3.3

• NumPy 2.2.6

• Scikit-Learn 1.7.2

# Future Enhancements
• Convert this binary classificcation model into multi classification model

• Add ergonomic risk scoring system

• Add cloud dashboard for posture logging

• Deploy model as a web app (Flask/FastAPI)

• Add pose-correction suggestions using vector geometry

• Integrate with wearable devices

# Author
Sachin Kumar

Machine Learning & Computer Vision Developer

GitHub: https://github.com/devsachink22