# Computer-vision
An end-to-end computer vision system implementing image analysis, object detection, and AI-driven visual intelligence using Python and deep learning frameworks. Designed for experimentation, learning, and real-world deployment.

# CSE 445: Interpretable Computer Vision Labs

This repository contains the laboratory notebooks for **CSE 445**, focusing on image representation, transfer learning, interpretability, and robustness analysis.

**Current Focus:** Image Representation, CNNs (ConvNeXt), and Vision Transformers (ViT, Swin).

---

## 📚 Lab Index

| Lab | Notebook File | Key Topics |
| :--- | :--- | :--- |
| **Lab 1** | `cse-445-lab-1-image-representation.ipynb` | Tensor manipulation, Color spaces (RGB/HSV/LAB), Augmentation, Data Bias. |
| **Lab 2A** | `cse445-lab2-a-interpretable-cnn-models.ipynb` | CNNs (ConvNeXt), Transfer Learning, Grad-CAM, Robustness Testing. |
| **Lab 2B** | `cse445-lab2-b-interpretable-transformer-models.ipynb` | Vision Transformers (ViT, Swin), Attention Rollout, Failure Analysis. |


---

## 📘 Lab Details

### 1. Lab 1: Image Representation & Data Exploration
**File:** `cse-445-lab-1-image-representation.ipynb`

A foundational lab to understand how machines "see" images. It covers manual data inspection and preprocessing before training models.
* **Core Concepts:** `torchvision` datasets, tensor statistics (mean/std), and channel analysis.
* **Visualizations:** Comparing RGB vs. HSV/LAB color spaces to find distinctive features.
* **Augmentation:** Visualizing `RandomHorizontalFlip`, `Rotation`, and `ColorJitter`.

### 2. Lab 2A: Interpretable CNN Models
**File:** `cse445-lab2-a-interpretable-cnn-models.ipynb`

Focuses on **Convolutional Neural Networks (ConvNeXt)**. We move beyond accuracy to understand *why* the model makes decisions.
* **Feature Viz:** Visualizing what early layers (edges) vs. deep layers (semantics) detect.
* **Explainability:** Using **Grad-CAM** to generate heatmaps on leaf images.
* **Robustness:** Testing model accuracy against synthetic noise, blur, and occlusion.

### 3. Lab 2B: Interpretable Transformer Models
**File:** `cse445-lab2-b-interpretable-transformer-models.ipynb`

Explores **Vision Transformers (ViT, ConViT, Swin)**. This lab contrasts CNNs with attention-based architectures.
* **Architecture Comparison:** Benchmarking MobileViT vs. Swin vs. ConViT.
* **Attention Mechanisms:** Visualizing Attention Rollout (CLS token $\to$ Pixel focus).
* **Stress Testing:** Evaluating how Transformers handle image corruption compared to CNNs.

---

## 🍃 Dataset

The labs utilize the **Pumpkin Leaf Dataset**, classified into 5 categories:
1.  **Bacterial Leaf Spot**
2.  **Downy Mildew**
3.  **Healthy Leaf**
4.  **Mosaic Disease**
5.  **Powdery Mildew**

*Note: Ensure your dataset path is configured correctly in the notebook `Config` cell (e.g., `/kaggle/input/...` or local path).*

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
