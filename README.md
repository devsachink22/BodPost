# BodPost: Real-Time Body Posture Detection System
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
