import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score, confusion_matrix
from sklearn.decomposition import PCA

# ==========================================
# 1. DATA LOADING
# ==========================================
print("--- 1. DATA LOADING ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, 'smartpharma_dataset_20k.csv')

if not os.path.exists(filename):
    print(f"File not found at {filename}. Please ensure the file is in the folder.")
    exit()

df = pd.read_csv(filename)
print(f"Data Loaded. Shape: {df.shape}")

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
print("\n--- 2. PREPROCESSING PIPELINE ---")

# A. Cleaning & Deduplication
df = df.drop_duplicates(subset=['prescription_id'])
df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True, errors='coerce')
df['administration_date'] = pd.to_datetime(df['administration_date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['order_date', 'administration_date'])
df = df[df['administration_date'] >= df['order_date']]
df = df[(df['patient_age'] >= 0) & (df['patient_age'] <= 120)]

# B. Feature Engineering
df['admin_delay_hours'] = (df['administration_date'] - df['order_date']).dt.total_seconds() / 3600

# C. Imputation
numeric_features = ['dose_mg', 'frequency_per_day', 'prescriber_experience_years', 'patient_age', 'renal_impairment', 'allergy_flag', 'admin_delay_hours']
categorical_features = ['gender', 'ward_type', 'drug_class', 'hospital_location']

imputer_num = SimpleImputer(strategy='median')
df[numeric_features] = imputer_num.fit_transform(df[numeric_features])

imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

# ==========================================
# 2.5 EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\n--- 2.5 RUNNING EDA (Generating 4 Key Figures) ---")
sns.set(style="whitegrid")

# Figure 1: Target Balance
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='medication_error', data=df, palette='viridis')
plt.title('Figure 1: Distribution of Medication Errors', fontsize=14)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + 0.35, p.get_height() + 100))
plt.savefig('eda_fig1_target_balance.png')
print("Saved Figure 1")

# Figure 2: Dosage vs Error
plt.figure(figsize=(10, 6))
sns.boxplot(x='medication_error', y='dose_mg', data=df, palette='coolwarm')
plt.title('Figure 2: Dosage Distribution by Error Status', fontsize=14)
plt.savefig('eda_fig2_dose_vs_error.png')
print("Saved Figure 2")

# Figure 3: Ward Errors
plt.figure(figsize=(12, 6))
if df['medication_error'].nunique() > 1:
    sns.histplot(data=df, x='ward_type', hue='medication_error', multiple='fill', palette='viridis', shrink=0.8)
    plt.legend(title='Error Status', labels=['Error', 'Safe'])
else:
    sns.countplot(x='ward_type', data=df, palette='viridis')
plt.title('Figure 3: Proportion of Errors by Ward Type', fontsize=14)
plt.tight_layout()
plt.savefig('eda_fig3_ward_errors.png')
print("Saved Figure 3")

# Figure 4: Correlation
plt.figure(figsize=(14, 10))
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['prescription_id', 'patient_id'], errors='ignore')
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='RdBu', center=0, square=True, cbar_kws={"shrink": 0.7})
plt.title('Figure 4: Feature Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_fig4_correlation.png', dpi=300)
print("Saved Figure 4")

# ==========================================
# RESUME PREPROCESSING
# ==========================================
# D. Scaling
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# E. Encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Prepare Data
drop_cols = ['prescription_id', 'patient_id', 'order_date', 'administration_date', 'medication_error']
X = df_encoded.drop(columns=drop_cols)
y = df_encoded['medication_error']

if X.isnull().sum().sum() > 0:
    X = X.fillna(0)

print(f"Preprocessing Complete. Features ready: {X.shape}")

# ==========================================
# 3. METHOD 1: BASE NEURAL NETWORK
# ==========================================
print("\n--- 3A. TRAINING BASE MODEL ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=200, random_state=42, verbose=True)

print("Training Base Model...")
nn_model.fit(X_train, y_train) 
y_pred = nn_model.predict(X_test)
base_acc = accuracy_score(y_test, y_pred) 

print(f"Base Model Accuracy: {base_acc:.2%}")
print(classification_report(y_test, y_pred))

# ==========================================
# RE-TRAINING: OPTIMIZED NEURAL NETWORK
# ==========================================
print("\n--- 3B. RE-TRAINING WITH OPTIMIZED PARAMETERS ---")

optimized_nn = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', learning_rate='adaptive', max_iter=500, random_state=42, verbose=True)

print("Training Optimized Model...")
optimized_nn.fit(X_train, y_train)
y_pred_opt = optimized_nn.predict(X_test)
new_acc = accuracy_score(y_test, y_pred_opt)

print(f"\nPrevious Accuracy: {base_acc:.2%}")
print(f"New Accuracy:      {new_acc:.2%}")

if new_acc > base_acc:
    print("SUCCESS: The optimized model performed better!")
else:
    print("NOTE: The previous model was already very efficient.")

print("\nClassification Report (Optimized):")
print(classification_report(y_test, y_pred_opt))

# ==========================================
# 4. METHOD 2: CLUSTERING (K-Means)
# ==========================================
print("\n--- 4. METHOD 2: CLUSTERING (K-Means) ---")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

analysis_df = X.copy()
analysis_df['Cluster_Label'] = clusters
analysis_df['Actual_Error'] = y.values

print("\nError Rate by Cluster:")
print(analysis_df.groupby('Cluster_Label')['Actual_Error'].mean())

# ==========================================
# 5. METHOD 3: REGRESSION
# ==========================================
print("\n--- 5. METHOD 3: REGRESSION (Predicting Operational Delays) ---")
# Define Target (y_reg) and Features (X_reg)
y_reg = df_encoded['admin_delay_hours']
reg_drop_cols = ['prescription_id', 'patient_id', 'order_date', 'administration_date', 'medication_error', 'admin_delay_hours']
X_reg = df_encoded.drop(columns=reg_drop_cols)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print(f"Regression MAE: {mae:.2f} hours")
print(f"Regression R2 Score: {r2:.4f}")

# ==========================================
# 6. VISUALIZATION (Main Analysis Figures)
# ==========================================
print("\n--- 6. VISUALIZATION (Generating 3 Figures) ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# FIGURE 5A: PATIENT CLUSTERS (Unsupervised)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('Figure 5A: Patient Risk Clusters (K-Means)', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Group')
plt.grid(True, alpha=0.3)
plt.savefig('viz_cluster_groups.png')
print("Saved 'viz_cluster_groups.png'")

# FIGURE 5B: ACTUAL ERRORS (Ground Truth)
plt.figure(figsize=(10, 6))
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6, edgecolors='k', s=30)
plt.title('Figure 5B: Actual Medication Errors (Ground Truth)', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
cbar = plt.colorbar(scatter2, ticks=[0, 1])
cbar.ax.set_yticklabels(['Safe (0)', 'Error (1)'])
plt.grid(True, alpha=0.3)
plt.savefig('viz_error_distribution.png')
print("Saved 'viz_error_distribution.png'")

# FIGURE 6: REGRESSION (Actual vs Predicted Style)
plt.figure(figsize=(8, 8))
plt.scatter(y_reg_test, y_reg_pred, color='#6A0DAD', alpha=0.4, label='Predictions')

min_val = min(y_reg_test.min(), y_reg_pred.min())
max_val = max(y_reg_test.max(), y_reg_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (Ideal)')

plt.title('Figure 6: Regression Performance (Actual vs Predicted)', fontsize=14)
plt.xlabel('Actual Processing Delay (Hours)', fontsize=12)
plt.ylabel('Predicted Processing Delay (Hours)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig('viz_regression_analysis.png')
print("Saved 'viz_regression_analysis.png' (Shows Method 3)")

# ==========================================
# 7. IMPROVED VISUALIZATION (Professional Grade)
# ==========================================
print("\n--- 7. VISUALIZATION (Generating 3 Professional Figures) ---")

# --- FIGURE 7: CONFUSION MATRIX ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Predicted Safe', 'Predicted Error'],
            yticklabels=['Actual Safe', 'Actual Error'])
plt.title('Figure 7: Confusion Matrix (Neural Network)', fontsize=14)
plt.ylabel('Ground Truth')
plt.xlabel('AI Prediction')
plt.savefig('viz_confusion_matrix.png')
print("Saved 'viz_confusion_matrix.png'")

# --- FIGURE 8: TRAINING LOSS CURVE ---
plt.figure(figsize=(10, 5))
plt.plot(optimized_nn.loss_curve_, color='#FF5733', linewidth=2)
plt.title('Figure 8: Neural Network Learning Curve (Loss over Epochs)', fontsize=14)
plt.xlabel('Iterations (Epochs)')
plt.ylabel('Loss (Error Rate)')
plt.grid(True, alpha=0.3)
plt.savefig('viz_learning_curve.png')
print("Saved 'viz_learning_curve.png'")

# --- FIGURE 9: CLUSTER PROFILE HEATMAP ---
df_clusters = X.copy()
df_clusters['Cluster'] = clusters
cluster_means = df_clusters.groupby('Cluster')[['dose_mg', 'patient_age', 'prescriber_experience_years', 'admin_delay_hours']].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means.T, cmap='YlGnBu', annot=True, fmt=".2f", linewidths=.5)
plt.title('Figure 9: Cluster Characteristics (Who is in each group?)', fontsize=14)
plt.xlabel('Cluster ID')
plt.ylabel('Feature Average (Normalized)')
plt.savefig('viz_cluster_heatmap.png')
print("Saved 'viz_cluster_heatmap.png'")

print("Pipeline Finished Successfully. All 9 Figures Generated.")
plt.show()