# SmartPharma: AI-Based Medication Error Detection

## Project Overview
This project applies Data Analytics and Machine Learning to reduce medication errors in Malaysian inpatient wards. Using a synthetic dataset of 20,000 prescriptions, we developed an AI pipeline to verify dosage safety and predict operational delays.

## Methodology
We implemented an end-to-end pipeline including:
1. **Data Cleaning & Imputation:** Handling missing values and outliers.
2. **Exploratory Data Analysis (EDA):** Visualizing risk factors (Age, Dosage, Ward).
3. **Machine Learning Models:**
   * **Neural Network (MLP):** Predicts medication errors with **97.40% Accuracy**.
   * **K-Means Clustering:** Identifies high-risk patient profiles unsupervised.
   * **Linear Regression:** Analyzes operational bottlenecks.

## Files in this Repository
* `coding_assignment.py`: The full Python script (Preprocessing -> Training -> Visualization).
* `smartpharma_dataset_20k.csv`: The dataset used for training.
* `viz_*.png`: Generated figures showing model performance and clusters.

## Results
* The optimized Neural Network successfully identifies **80% of errors** (Recall) with high precision.
* Clustering revealed that "High-Risk Elderly" patients form a distinct mathematical group.
