import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. LOAD DATASET

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

import numpy as np

def live_input():
    print("\n" + "="*50)
    print("      READY FOR USER INPUT      ")
    print("="*50)
    
    while True:
        user_input = input("\nPaste the 29 values (or type 'exit'): ").strip()
        
        if user_input.lower() == 'exit':
            break

        try:
            # Cleaning the input (removes spaces, newlines, and double commas)
            cleaned_list = [float(val.strip()) for val in user_input.split(',') if val.strip()]

            if len(cleaned_list) != 29:
                print(f"❌ Error: Model needs 29 values, but you gave {len(cleaned_list)}.")
                continue

            # Convert to array and predict
            # Note: Ensure 'model' is the name of your trained variable
            test_data = np.array(cleaned_list).reshape(1, -1)
            prediction = model.predict(test_data)
            prob = model.predict_proba(test_data)

            print("\n" + "-"*30)
            if prediction[0] == 1:
                print(f"🚨 RESULT: FRAUDULENT (Confidence: {prob[0][1]*100:.2f}%)")
            else:
                print(f"✅ RESULT: LEGITIMATE (Confidence: {prob[0][0]*100:.2f}%)")
            print("-" * 30)

        except ValueError:
            print("❌ Error: Please enter only numbers separated by commas.")

# Start the loop
live_input()