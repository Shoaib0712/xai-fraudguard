import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)
import joblib

def create_directories():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

def main():
    create_directories()
    filepath = "data/creditcard.csv"
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please place creditcard.csv inside data/.")
    
    print("[INFO] Loading Kaggle Credit Card Fraud dataset...")
    df = pd.read_csv(filepath)
    
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Removed {initial_rows - len(df)} duplicate rows. Clean dataset shape: {df.shape}")
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    print("[INFO] Performing stratified train-test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    cols_to_scale = ['Time', 'Amount']
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    print("[INFO] Applying SMOTE to training partition to resolve extreme class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"[INFO] Resampled training shape: {X_train_resampled.shape}")
    
    joblib.dump(scaler, "models/scaler.pkl")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    best_score = 0
    best_model_obj = None
    
    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        pr_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        print(f"-> {name} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | MCC: {mcc:.4f}")
        joblib.dump(model, f"models/{name.lower().replace(' ', '_')}.pkl")
        
        if pr_auc > best_score:
            best_score = pr_auc
            best_model_obj = model
            
    joblib.dump(best_model_obj, "models/best_model.pkl")
    print(f"\n[SUCCESS] Training complete! Best model saved successfully with PR-AUC: {best_score:.4f}")

if __name__ == "__main__":
    main()