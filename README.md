# Skin Disease Classification using ResNet18

## Overview

This project implements a deep learning pipeline for multi-class skin disease classification using transfer learning with ResNet18. The model is trained on a Kaggle skin disease dataset and leverages image augmentation, PyTorch DataLoaders, and a fine-tuned pretrained convolutional neural network to achieve strong classification performance.

## Features

* Transfer Learning using ResNet18
* Custom PyTorch Dataset and DataLoader
* Data Augmentation using Torchvision
* GPU Training with Google Colab
* Multi-class Skin Disease Classification
* Validation and Test Evaluation Pipeline
* Confusion Matrix and Classification Metrics

## Dataset

Dataset Source:

Kaggle Skin Disease Dataset

The dataset contains approximately 1,494 images distributed across 5 skin disease classes.

## Data Processing

* Train / Validation / Test Split
* Image Resizing (224 × 224)
* RGB Conversion
* Data Augmentation:

  * Random Horizontal Flip
  * Random Rotation
  * Color Jitter
* ImageNet Normalization

## Model Architecture

* Backbone: ResNet18 (ImageNet Pretrained)
* Custom Classification Head
* CrossEntropy Loss
* Adam Optimizer

## Results

| Metric              | Score |
| ------------------- | ----- |
| Training Accuracy   | 99.5% |
| Validation Accuracy | 91.0% |
| Test Accuracy       | 92.0% |

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-Learn
* Google Colab

## Repository Structure

```text
├── notebooks/
├── dataset/
├── models/
├── outputs/
├── README.md
├── requirements.txt
```

## Future Improvements

* EfficientNet and ConvNeXt comparison
* Grad-CAM Explainability
* Streamlit Web Application
* Model Deployment
