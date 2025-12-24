# 𝗕𝗿𝗮𝗶𝗻 𝗧𝘂𝗺𝗼𝗿 Detection 🔍

---

👩‍⚕️ Doctors spend hours analyzing MRI scans, but this Deep Learning model inspired by **VGG16 CNN Architecture** spots tumors **faster and smarter**

---

## 🔎 Key Features

- **🎯 CNN-Powered Precision:** Accurate multi-class tumor detection  
- **🤖 Automated Analysis:** No manual magnifying glass analysis required  
- **⚡ Optimized Accuracy:** Near expert-level results  
- **⏱️ Efficiency:** Faster than manual diagnosis  

---

## 📂 Dataset & Preprocessing

- Dataset from [Kaggle Brain Tumor MRI Images (17 classes)](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes)  
- Anonymized MRI scans, labeled by radiologists  
- Minimal preprocessing; data augmentation (rotations, flips, etc.) for robustness  

---

## 🏗️ Model Architecture & Training

- Based on **VGG16** with added **Batch Normalization** for stability and faster training  
- **Loss function:** CrossEntropyLoss (for multi-class classification)  
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)  
- Fine-tuning and extended epochs to boost accuracy  

---

## 📊 Evaluation Metrics

| Metric             | Purpose                                        |
|--------------------|------------------------------------------------|
| **Accuracy**       | Overall prediction correctness                 |
| **Specificity**    | Correctly identifying non-tumor cases          |
| **F1 Score**       | Balance between precision and recall            |
| **Confusion Matrix**| Visualizing true vs false positives/negatives |
| **Calibration Curve**| Model confidence calibration                   |
| **SSIM**           | Preserving structural similarity in images     |

---

## 💡 Strengths & Weaknesses

**Strengths:**  
- 🎯 Training accuracy: **95.06%** — excellent learning  
- 💪 Strong F1 scores (>0.90) for most classes  
- 🌟 Good generalization across tumor types  

**Weaknesses:**  
- ⚠️ Test accuracy dips to **86.46%** — some overfitting  
- 🚧 Lower performance on underrepresented classes  
- 🔍 Challenges in generalizing certain tumor classes  

---

## 🚀 Dive Deeper

Explore the full model details, training process, and results in my [Colab Notebook](https://colab.research.google.com/drive/1_RybqqdYU0vu34HJC6fGXTkbgAK9soUh?usp=sharing)  

---

*Thanks for stopping by!* 😊
