# Computer-vision
An end-to-end computer vision system implementing image analysis, object detection, and AI-driven visual intelligence using Python and deep learning frameworks. Designed for experimentation, learning, and real-world deployment.


---

**Folder:** [`Project/`](./Project)

An end-to-end livestock pose estimation system that detects and tracks individual
cows across video frames using **YOLOv12s-Pose** and multiple tracking algorithms.
Built for **CSE 445: Computer Vision** on Kaggle (GPU environment).

---

### 🎯 Objective
Estimate skeletal keypoints of cows in real-time video, assign persistent IDs
across frames, and compare the performance of two state-of-the-art trackers.

---

### 🗃️ Dataset
| Property | Detail |
|---|---|
| Source | [Roboflow Universe — Cow Pose Estimation](https://universe.roboflow.com/mikaelapisani/cow-pose-estimation-fxosp) |
| Format | COCO Keypoints (bounding boxes + skeletal keypoints) |
| Splits | Train / Validation / Test |

---

### 🏗️ Model Architecture: YOLOv12s-Pose
| Component | Detail |
|---|---|
| Base Model | YOLOv12s (detection backbone) |
| Strategy | Transfer Learning — backbone pretrained, pose head trained from scratch |
| Attention | Area Attention + Flash Attention |
| Blocks | R-ELAN (Residual Efficient Layer Aggregation) |

---

### 🔧 Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| LR Schedule | Cosine Annealing |
| Epochs | 100 (Early stopping: patience=20) |
| Input Size | 640×640 |

### 🎨 Augmentation Pipeline
| Augmentation | Value | Purpose |
|---|---|---|
| Mosaic | 1.0 | Multi-image spatial diversity |
| MixUp | 0.15 | Robustness via image blending |
| Copy-Paste | 0.1 | Simulates occlusion |
| Rotation | ±20° | Cow orientation variation |
| HSV Shifts | H=0.015, S=0.7, V=0.4 | Lighting/weather robustness |
| Label Smoothing | 0.01 | Regularization |

---


### 📊 Results

| Metric | Score | Interpretation |
|---|---|---|
| **mAP50 (pose)** | **0.8241** | Strong keypoint detection accuracy |
| **mAP50-95 (pose)** | **0.2132** | Challenging at stricter IoU thresholds |
| **mAP50 (box)** | **0.9950** | Near-perfect cow detection |
| **mAP50-95 (box)** | **0.9397** | Excellent detection across all IoU thresholds |
| **Precision** | **1.0000** | Zero false positives |
| **Recall** | **0.9988** | Almost no cows missed |
---
> **Note on mAP50-95 (pose):** The lower value (0.21) is expected for pose estimation —
> at strict IoU thresholds (up to 0.95), keypoints must be placed near pixel-perfectly.
> The mAP50 score of **0.82** is the more meaningful indicator of real-world performance.

### 🎬 Multi-Object Tracking Comparison

**ByteTrack** — Kalman Filter + Hungarian algorithm IoU matching.
Uses both high and low-confidence detections (the "byte" idea) to recover
occluded tracks.

**BoT-SORT** — Extends ByteTrack with camera motion compensation
and Re-ID feature matching for more stable long-term tracking.

## 🎬 Tracking Demo Videos

| Tracker | Demo |
| :--- | :--- |
| ByteTrack | [▶ Watch Video](https://drive.google.com/file/d/1dzw7R4RTzWOm3mdx_1wdWVhgJKN3boFV/view?usp=sharing) |
| BoT-SORT | [▶ Watch Video](https://drive.google.com/file/d/1UQlcxMQkFMJKABH07qunUS-McYo7kqCE/view?usp=sharing) |

| Feature | ByteTrack | BoT-SORT |
|---|---|---|
| Kalman Filter | ✅ | ✅ |
| Low-score detection recovery | ✅ | ✅ |
| Camera Motion Compensation | ❌ | ✅ |
| Re-ID Matching | ❌ | ✅ |

---

### 🛠 Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv12-blue)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-purple)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

### 📁 Project Structure
```
Project/
└── Cow-Pose-Estimation.ipynb    # Full pipeline: data → train → eval → track
```

## 🛠 Dependencies & Installation

To run these notebooks, install the following dependencies:

```bash
pip install torch torchvision timm opencv-python matplotlib numpy tqdm
timm: Required for loading state-of-the-art models (ConvNeXt, Swin, ViT).

torchvision: Used for datasets and image transforms.

opencv-python: Used for advanced image manipulation in Lab 1.

🚀 Usage
1.Clone this repository.

2.Install requirements.

3.Launch Jupyter Lab or Notebook:

jupyter notebook (Bash)

4.Open the specific lab file from the index above.
