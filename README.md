# 🧠 Brain Tumor Detection Using Deep Learning

This project presents an **end-to-end deep learning system** for brain tumor classification from MRI scans.  
The goal is to explore how a CNN-based approach can **assist medical image analysis** by learning tumor-related patterns, while being evaluated using **medical-relevant metrics rather than accuracy alone**.

The work was developed as a group project and later formalized into a **published research paper**.

---

## 🎯 Problem Overview

Brain tumor diagnosis using MRI scans is time-consuming and requires expert interpretation.  
With large volumes of scans and limited radiology resources in many regions, automated **decision-support systems** can help improve efficiency and consistency.

This project focuses on:
- Multi-class brain tumor classification from MRI images  
- Reliable evaluation beyond accuracy  
- A complete pipeline from training to deployment  

> ⚠️ This system is a **research and decision-support prototype**, not a clinical diagnostic tool.

---

## 📂 Dataset

- **Source:** Brain Tumor MRI Images (17 classes) – Kaggle  
- **Data:** Real, anonymized MRI scans labeled by radiologists  
- **Modalities:** T1, T1C+, T2  
- **Classes:** Multiple tumor types + normal scans  

---

## 🏗️ Methodology

### Data Preparation
- Clean **train / validation / test split** to avoid data leakage  
- **Training-only data augmentation** (flips, rotations, intensity changes)  
- Validation and test sets kept untouched  

### Model
- **VGG16-inspired CNN**, trained from scratch  
- Batch Normalization and Dropout for stability and generalization  
- Designed to learn **MRI-specific features**

### Training
- Loss: CrossEntropyLoss  
- Optimizer: Adam with weight decay  
- GPU-accelerated training (Google Colab)  
- **Early stopping and model selection based on validation F1-score**

---

## 📊 Evaluation

Instead of relying only on accuracy, the model is evaluated using:
- **Macro & per-class F1-score**
- **Specificity**
- **Confusion matrix**
- **Calibration curve**

These metrics provide a clearer picture of **class-wise reliability and confidence**, which is critical in medical applications.

---

## 🚀 Deployment

- Best-performing model saved with:
  - Model weights
  - Class label mapping
  - Preprocessing configuration  
- Deployed using **Streamlit** for inference-only usage  
- Ensures preprocessing consistency between training and deployment  

---

## 📄 Research Publication

The experimental results, methodology, and evaluation were formalized into a research paper and published at **INCOFT 2025**.  
The contribution focuses on **applied deep learning**, medical-aware evaluation, and reproducibility rather than architectural novelty.

---

## 🔍 Limitations & Future Work

- Class imbalance affects some rare tumor categories  
- Single-source dataset without external validation  
- No explainability methods (e.g., Grad-CAM) yet  

Future improvements include multi-center validation, explainability, and better imbalance handling.

---

## 👤 Contribution

My primary contribution focused on:
- Data preparation and augmentation strategy  
- CNN architecture design for improved generalization  
- Selection and analysis of healthcare-relevant evaluation metrics  
- Deployment of the trained model for inference

---

This project demonstrates a **practical and responsible application of deep learning in medical imaging**, with emphasis on evaluation, transparency, and real-world constraints.
