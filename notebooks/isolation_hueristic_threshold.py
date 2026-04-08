# ---------------------------------------------------------
# THE ULTIMATE LEAK-FREE SCORE MAXIMIZER (Clean State)
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report

print("1. Loading fresh data to clear Jupyter memory...")
df_clean = pd.read_csv('../data/processed/road_processed.csv')

X_clean = df_clean.drop('Accident_severity', axis=1)
y_clean = df_clean['Accident_severity']

# Encode target mathematically (Fatal=0, Serious=1, Slight=2 usually)
from sklearn.preprocessing import LabelEncoder
le_clean = LabelEncoder()
y_clean_encoded = le_clean.fit_transform(y_clean)

# One-hot encode features and fix column names for XGBoost
X_clean_encoded = pd.get_dummies(X_clean, drop_first=True)
X_clean_encoded.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X_clean_encoded.columns]

print("2. Splitting into strictly pure Train and Test sets...")
# SPLIT FIRST! NO SMOTE!
X_train_pure, X_test_pure, y_train_pure, y_test_pure = train_test_split(
    X_clean_encoded, y_clean_encoded, test_size=0.2, random_state=42, stratify=y_clean_encoded
)

print("3. Calculating Penalty Weights for XGBoost...")
weights = compute_sample_weight(class_weight='balanced', y=y_train_pure)

print("4. Training XGBoost with strict penalties...")
xgb_final = XGBClassifier(
    objective='multi:softprob', 
    num_class=3, 
    random_state=42,
    n_estimators=300,      
    learning_rate=0.05,    
    max_depth=8,           
    n_jobs=-1              
)

# Train on pure training data using weights
xgb_final.fit(X_train_pure, y_train_pure, sample_weight=weights)

print("5. Applying Threshold Shift to boost Minority Detection...")
y_pred_proba = xgb_final.predict_proba(X_test_pure)

# We boost the model's confidence in Fatal and Serious crashes
multiplier = np.array([2.5, 1.5, 1.0]) 
y_pred_shifted = y_pred_proba * multiplier
y_pred_final = np.argmax(y_pred_shifted, axis=1)

print("\n" + "="*40)
print("🏆 THE ACTUAL TRUE METRICS 🏆")
print("="*40)
macro_f1 = f1_score(y_test_pure, y_pred_final, average='macro')
micro_f1 = f1_score(y_test_pure, y_pred_final, average='micro')

print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Micro F1-Score: {micro_f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test_pure, y_pred_final, target_names=le_clean.classes_))