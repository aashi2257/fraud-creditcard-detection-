# 💳 Credit Card Fraud Detection System
> **An Artificial Intelligence solution to identify and prevent fraudulent transactions.**

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

---

## 📌 Project Overview
This project addresses the **Class Imbalance** problem in financial data. In a typical dataset, fraudulent transactions are extremely rare (less than 0.2%). Our model uses **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the data and a **Random Forest Classifier** to detect suspicious patterns.

### 🎯 Objectives
* **Identify** fraud in real-time.
* **Minimize** False Positives (don't block real customers).
* **Handle** high-dimensional PCA data.

---

## ⚙️ How It Works (The Pipeline)

| Stage | Process | Description |
| :--- | :--- | :--- |
| **1** | **Data Cleaning** | Normalizing `Amount` and dropping `Time`. |
| **2** | **Oversampling** | Using **SMOTE** to create synthetic fraud cases. |
| **3** | **Training** | Building an ensemble of **100 Decision Trees**. |
| **4** | **Inference** | A live terminal loop for instant fraud checking. |



---

## 📊 Performance Metrics
The model is evaluated based on its ability to catch fraud (**Recall**) without being too aggressive (**Precision**).

* **Accuracy:** `99.94%` 🚀
* **Precision (Class 1):** `0.89`
* **Recall (Class 1):** `0.78`
* **F1-Score:** `0.83`



---

## 🚀 Step-by-Step Guide

### 1️⃣ Installation
Ensure you have the required libraries installed:
```bash
pip install pandas scikit-learn imbalanced-learn numpy

### 2️⃣ Training the Model
Run the main script. It will load the data, apply SMOTE, train the Random Forest, and show the performance report.

Bash
python main.py

### 3️⃣ Testing with Real Data
Use the Extractor Script to get a formatted string of 29 values:

Bash
python get_data.py
Copy the Fraud line.

Paste it into the main.py terminal prompt.

Observe the result! 🚨


------


🧠 #  Key Logic Explained
Why SMOTE? > Without SMOTE, the AI would be "lazy" and learn to say "No Fraud" 100% of the time. SMOTE creates "fake" fraud examples to challenge the AI and make it smarter.

Why Random Forest?
It is an Ensemble Learning method. By combining the results of many trees, it reduces the chance of making a wrong guess based on a single weird transaction.