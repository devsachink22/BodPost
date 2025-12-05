# Project Structure
BodPost/
│── data/                     # Raw posture images grouped in class folders
│──   ├── correct/
│──   ├── ├── img1.jpg
│──   ├── ├── img2.jpg
│──   ├── incorrect/
│──   ├── ├── img1.jpg
│──   ├── ├── img2.jpg
│── labelled_data/            # Generated posture feature CSV
│── models/                   # Trained ML model + feature columns JSON
│── scripts/                  # Offline scripts (dataset prep, feature generation)
│── src/                      # Core source code
│──   ├── feature_extraction.py
│──   ├── train_classifier.py
│──   ├── realtime_posture_detection.py
│──   └── utils.py
│── documents/                # Proposal, notes, ppts, other docs
│── README.md                 # Project documentation
