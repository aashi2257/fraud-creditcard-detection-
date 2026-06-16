import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. LOAD DATASET
# Download 'creditcard.csv' from Kaggle and place it in the same folder
try:
    df = pd.read_csv('project/creditcard.csv.zip')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Please download creditcard.csv from Kaggle first.")
    exit()

# 2. PRE-PROCESSING
# 'Amount' needs scaling because its range is different from V1-V28 features
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Dropping 'Time' (not very useful) and original 'Amount'
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# 3. DEFINE FEATURES AND TARGET
X = df.drop('Class', axis=1)
y = df['Class']

# 4. SPLIT DATA (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 5. ADDRESS CLASS IMBALANCE (SMOTE)
# This creates synthetic examples of the minority class (fraud)
print("Applying SMOTE to balance the dataset...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 6. INITIALIZE AND TRAIN MODEL
print("Training the Random Forest Model (this may take a minute)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)

# 7. PREDICTIONS AND EVALUATION
y_pred = model.predict(X_test)

print("\n--- Model Performance Report ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. VISUALIZE CONFUSION MATRIX
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Fraud Detection Confusion Matrix')
plt.show()
# Add this to the end of your script
import numpy as np

importances = model.feature_importances_
indices = np.argsort(importances)[-10:] # Top 10 features

plt.title('Top 10 Important Features for Fraud Detection')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()