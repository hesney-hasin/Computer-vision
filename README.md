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
