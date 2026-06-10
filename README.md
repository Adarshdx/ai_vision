# ai_vision
CIFAR-10 image classifier using ResNet18 with transfer learning. Features data augmentation, training/validation split, learning rate scheduling. Achieves 90-92% accuracy on test set. Includes training script, inference, and visualization.
# 🖼️ CIFAR-10 Image Classifier using ResNet18

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-90%25%2B-brightgreen.svg)]()

A high-performance image classification model for CIFAR-10 dataset using **ResNet18** with **Transfer Learning** from ImageNet. Achieves 90-92% test accuracy with advanced data augmentation techniques.

## 🎯 Features

- ✅ **Transfer Learning** - Pretrained ResNet18 backbone
- ✅ **Data Augmentation** - Rotation, flip, color jitter, affine transforms
- ✅ **Training/Validation Split** - 90/10 split for proper evaluation
- ✅ **Learning Rate Scheduling** - Cosine annealing scheduler
- ✅ **Comprehensive Metrics** - Accuracy, precision, recall, F1-score
- ✅ **Visualizations** - Training curves, confusion matrix, sample predictions
- ✅ **Easy Inference** - Support for single image and batch prediction

## 📊 Dataset

**CIFAR-10** contains 60,000 32x32 color images in 10 classes:

| Class | Class | Class | Class | Class |
|-------|-------|-------|-------|-------|
| ✈️ Airplane | 🚗 Automobile | 🐦 Bird | 🐱 Cat | 🦌 Deer |
| 🐕 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚚 Truck |


**Total Parameters:** 11.7M | **Trainable:** 11.7M

## 📈 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 90-92% |
| **Precision (avg)** | 0.90 |
| **Recall (avg)** | 0.90 |
| **F1-Score (avg)** | 0.90 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Airplane | 0.91 | 0.90 | 0.90 |
| Automobile | 0.93 | 0.94 | 0.93 |
| Bird | 0.88 | 0.86 | 0.87 |
| Cat | 0.85 | 0.84 | 0.84 |
| Deer | 0.89 | 0.90 | 0.89 |
| Dog | 0.86 | 0.85 | 0.85 |
| Frog | 0.92 | 0.93 | 0.92 |
| Horse | 0.91 | 0.92 | 0.91 |
| Ship | 0.93 | 0.92 | 0.92 |
| Truck | 0.92 | 0.93 | 0.92 |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/cifar10-classifier.git
cd cifar10-classifier

- **Training:** 45,000 images
- **Validation:** 5,000 images  
- **Test:** 10,000 images

## 🏗️ Model Architecture
cifar10-classifier/
├── train.py              # Training script
├── inference.py          # Inference script
├── train_notebook.ipynb  # Jupyter notebook
├── model.py              # Model architecture
├── dataset.py            # Data loading & augmentation
├── utils.py              # Utility functions
├── config.py             # Configuration
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── data/                 # Dataset (auto-downloaded)
├── models/               # Saved models
│   ├── best_model.pth    # Best checkpoint
│   └── final_model.pth   # Final model
└── outputs/              # Generated plots
    ├── training_curves.png
    ├── confusion_matrix.png
    └── sample_predictions.png
