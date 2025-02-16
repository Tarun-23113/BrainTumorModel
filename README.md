# 🧠 Brain Tumor Classification : AI That Sees What You Can’t 🔍  
---
Because staring at MRI scans for hours is so last century, my Brain Tumor Detection Model does the job faster, smarter, and without needing ☕ coffee breaks.  
Powered by Deep Learning 🤖,  
this AI doesn’t just analyze scans—it judges them, spotting tumors with an attitude that says,  
"Oh, you didn’t see that? Rookie move." 😏

---
🔬 Features:  
✅ CNN-Powered Precision – Because guessing isn’t an option.  
✅ Automated Tumor Detection – No magnifying glass needed. 🔎  
✅ Optimized Accuracy – Almost like a second opinion, but from AI. ⚡ 
✅ Faster Than Your Radiologist – Sorry, humans.  

---
So, while doctors debate their findings, my AI is already sitting there like, "Yeah, that’s a tumor. Next?" 💥🚀  

Now we will discuss about the tech-stack used and how the model is trained,  
Let us start it from collection of  
**Dataset -**   
Dataset is originally obtained from the Kaggle where Images of real exams, without any data from the patient's medical record, thus preserving their identity.  
Exams interpreted by radiologists and provided for study purposes.  
Below is the link provided fir the dataset - https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes  
This consist of 17 diffrent classes of Tumours.  

**Data Augmentations and preprocessing -**   
Dataset is available in structured manner so it doesn't need any preprocessing, and the only work here will be for making dataset classes.  
Now taking about the Data Augmentation we used diffrent transformations to make data more complex for model to study in order to achieve a reliable and highly accurate model.  

**Model Architecture -**  
Model Architechture is inspired from VGG-16 Architecture,  
one thing that is additional in this model is that it has BatchNorm layer in it.  
Batch Normalization is used for Faster Training & Stability.  
  
**Loss and Optimizer Function -**  
For loss calculating, I used CrossEntropyLoss() which is mainly for multiclass classification.  
We know that, CrossEntropyLoss() is mainly used when we have unbalanced training dataset.  
  
For optimizer, I used Adam optimizer which is better in this case.  
Parameters used were lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999).  
  
**Training Model -**  
Now comes the most time consuming, cpu consuming work which is training the model until we get accuracy.  
Here for better accuracy, I used techniques like fine tuning, also increased no. of loops.  
  
**Model Evaluation -**  
Model is evaluated on 5 metrices -  
  
1.Accuracy – Measures the proportion of correctly classified cases, indicating the overall model performance.  

2.Specificity – Assesses the model's ability to correctly identify non-tumor cases, reducing false positives.  

3.F1 Score – A balanced metric combining precision and recall to evaluate classification effectiveness.  

4.Confusion Matrix – Visualizes true/false positives and negatives, helping to analyze misclassification patterns.  

5.Calibration Curve – Compares predicted probabilities with actual outcomes to assess the model's confidence calibration.  

6.Structural Similarity Index (SSIM) – Measures the similarity between original and predicted images, ensuring better feature preservation.  

After Evaluation these are the Strengths and weaknesses of model -  
Strengths:  
1.	High preparation accuracy (95.06%): The display learned to effectively understand the designs within the preparation facts,  
    showing that the design is properly perfect for the task and data set. This high accuracy reflects excellent study skills.  
2.	Strong class performance: F1 scores for several classes are greater than 0.90 (e.g. class zero: 0.9040, class 1: zero.9107, course four: 0.9508),  
    indicating the adjusted accuracy and ranking for these classes. Publish behaves quite correctly when looking ahead to these classes.  
3.	Balanced F1 scores in most classes: Classes that include nine (0.9444) and 11 (0.9310) appear to be reliable and stable performers,  
    illustrating the version's potential to generalize well across categories.  
Weaknesses:  
1.	Testing Accuracy (86.46%): The difference between preparation and testing accuracy (approximately 9%) suggests reassembly.  
    The show probably learned to draft stats by rote, causing it to underperform on subtle control information.  
2.	Characteristic Biased Class: Lower F1 scores in instructions that contain 16 (zero.7234) and 5 (0.7642) may be the end result of  
    lesson imbalance or insufficient statistical tests for these categories, causing the display to struggle with accurate predictions.  
3.	Higher trial calamity (0.404): The comparison of prepared calamity (-0.153) and reported calamity seems to be trying to generalize.  
    A high calamity harbinger regularly focuses on issues with shouting or hazing in connection with an impending calamity.  

For more detailed study of model you can see the colab notebook link - 
[Brain Tumor Detection Notebook](https://colab.research.google.com/drive/1_RybqqdYU0vu34HJC6fGXTkbgAK9soUh?usp=sharing)
