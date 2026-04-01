# Predicting 30-Day Hospital Readmission & AI Fairness Audit

> **A dual-focus ML project: predicting 30-day hospital readmissions for diabetic patients and conducting a rigorous statistical AI fairness audit for racial bias.**

## Project Overview
This project tackles a critical intersection of data science and healthcare AI safety. The 30-day readmission rate is a standard global metric for assessing the quality of patient care. A false negative (predicting a highly vulnerable patient is safe to send home) can lead to severe complications. 

If risk-prediction models inherit historical biases, they may inadvertently deprive minority groups of access to life-saving post-discharge care. This repository contains an end-to-end pipeline that not only builds a predictive model for readmission risk but also rigorously audits that model to ensure it does not systematically disadvantage specific demographics.

## Tech Stack & Dataset
* **Language & Libraries:** Python, Pandas, NumPy, Scikit-learn, Matplotlib
* **Dataset:** [UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) 
* **Key Methods:** K-Means Clustering, Random Forest Classifier, Permutation Testing

## Key Findings

### 1. Patient Group Clustering (EDA)
Using K-Means clustering, we identified distinct patient profiles. The analysis revealed that readmission risk is heavily influenced by prior healthcare utilization (specifically prior inpatient and emergency visits) and the number of medications prescribed, rather than demographic factors alone.

### 2. Predictive Modeling & Feature Importance
Our Random Forest model successfully identified key indicators for readmission. The most critical features driving the model's predictions were:
1. `number_inpatient`
2. `discharge_disposition_id_11`
3. `discharge_disposition_id_22`
4. `number_emergency`

<img width="317" height="167" alt="image" src="https://github.com/user-attachments/assets/dfda960b-255f-4167-ad4a-97c569236f6c" />


### 3. AI Fairness Audit
We conducted a Fairness Audit focusing on the **False Negative Rate (FNR)**. A higher FNR for a specific demographic means the model systematically misses high-risk patients in that group.

We performed a Permutation Test to check for statistical significance in the FNR disparity between African American and Caucasian patients:
* **Observed FNR Difference:** -0.0090
* **P-value:** 0.7466

<img width="341" height="168" alt="image" src="https://github.com/user-attachments/assets/139aa4a7-f4f8-46fa-91c3-f3e616fe9da8" />


**Conclusion:** We cannot conclude that the model treats African American patients differently from Caucasian patients in terms of missed diagnoses. The small FNR gap observed is statistically indistinguishable from random chance, indicating our model remains fair across these groups.

## Repository Structure
To ensure clear logic and readability, the analysis pipeline is split into two main notebooks:

* `/notebooks/01_EDA_and_Clustering.ipynb`: Data preprocessing, exploratory data analysis, and patient profiling using K-Means.
* `/notebooks/02_Modeling_and_Fairness_Audit.ipynb`: Training the Random Forest classifier and conducting the statistical fairness audit via permutation testing.
* `/docs`: Includes the comprehensive final report and presentation slides detailing the clinical context and methodology.
