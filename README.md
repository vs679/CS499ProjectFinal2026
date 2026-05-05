# CS499ProjectFinal2026
Final Project for CS499/599 Spring 2026

# Diabetic Retinopathy Detection using CNN

## Overview

This project builds a Convolutional Neural Network (CNN) to detect and classify diabetic retinopathy from retinal fundus images. The goal is to automate early diagnosis and assist medical professionals in identifying disease severity.

---

## Problem Statement

Diabetic retinopathy is a leading cause of blindness worldwide. Manual diagnosis is time-consuming and requires expert knowledge. This project aims to develop a deep learning model that can classify retinal images into severity levels, improving efficiency and accessibility.

---

## Dataset

* **Source:** APTOS 2019 Blindness Detection (Kaggle)
* **Classes:**

  * 0 → No DR
  * 1 → Mild
  * 2 → Moderate
  * 3 → Severe
  * 4 → Proliferative DR

**Note:**
The dataset is not included in this repository. Download it from Kaggle and place it in the appropriate directory.

---

## Project Pipeline

1. Data Collection
2. Image Preprocessing
3. Data Augmentation
4. Model Training (CNN)
5. Evaluation & Visualization

---

## Preprocessing

* Image cropping and RGB conversion
* CLAHE to improve retinal detail
* Image resizing
* Normalization
* Data augmentation (Horizontal flips, random rotations, and brightness/contrast changes)

---

## Model Architecture

* Baseline Convolutional Neural Network (CNN)
* Layers:

  * Convolution + ReLU
  * Max Pooling
  * Fully Connected Layers
* Loss Function: Sparse Categorical Crossentropy
* Optimizer: Adam
* ResNet50
* EfficientnetB0

---

## Results

* Accuracy: 
* Observations:

  * Model performs well on majority classes
  * Struggles with class imbalance
  * Some overfitting observed

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/diabetic-retinopathy-cnn.git
cd diabetic-retinopathy-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Jupyter Notebook

```bash
jupyter notebook notebooks/CS499_Diabetic_Retinopathy_Final_Project.ipynb
```

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Scikit-learn

---

## Challenges

* Class imbalance in dataset
* Limited data for higher severity levels
* Overfitting during training

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

* Kaggle APTOS 2019 dataset
* Course: CS 499/599 Computer Vision for Healthcare

---

## Authors

* Vikram Singh
* Ian Nieto
