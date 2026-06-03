# model_trainer.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

def engineer_lag_features(df, feature_cols, lags=[1, 2, 3, 5, 10]):
    """
    Creates lagged versions of our features so XGBoost can 'see' the past.
    Added a 10-day lag to capture slightly longer macro momentum.
    """
    df_engineered = df.copy()
    
    for col in feature_cols:
        for lag in lags:
            df_engineered[f'{col}_lag{lag}'] = df_engineered[col].shift(lag)
            
    # Drop rows with NaNs introduced by lagging
    df_engineered = df_engineered.dropna()
    return df_engineered

def train_and_evaluate_xgboost(df, target_col='Target_Smooth_10d', test_size=0.2):
    """
    Splits data chronologically, applies sample weights for imbalanced classes,
    trains an XGBoost classifier, and evaluates the results.
    """
    print(f"Preparing data. Target variable is: {target_col}")
    
    # 1. Strict Data Leakage Prevention
    # We must drop ANY column that looks into the future or is a direct proxy for the target.
    # We explicitly drop 'Target_Regime' because the smoothed target is derived directly from it.
    leakage_cols = ['Future_Sq_Return', 'Rolling_Thresh_Low', 'Rolling_Thresh_High', 'Target_Regime', target_col]
    
    # Isolate valid features
    features = [c for c in df.columns if c not in leakage_cols]
    
    print("\n--- Validated Features (X) ---")
    print(", ".join(features))
    
    X = df[features]
    y = df[target_col]
    
    # 2. Chronological Train/Test Split
    split_idx = int(len(df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTraining Samples: {len(X_train)} | Testing Samples: {len(X_test)}")
    print("\nClass Distribution in Training Set:")
    print(y_train.value_counts(normalize=True).sort_index() * 100)
    
    # 3. Handle Class Imbalance
    # Because your smoothing reduces the number of Class 1 and Class 2 days, 
    # sample weights are more critical than ever.
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # 4. Initialize and Train XGBoost
    print("\nTraining XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        max_depth=4,           # Keep depth shallow
        learning_rate=0.05,
        n_estimators=250,      # Slightly increased to give it more time to find complex macro patterns
        subsample=0.8,
        colsample_bytree=0.8,  # Add column sampling to prevent single features (like VIX) from dominating
        random_state=42
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 5. Evaluate the Model
    print("\n" + "="*50)
    print("Model Evaluation on Unseen Test Data (Chronological)")
    print("="*50)
    
    y_pred = model.predict(X_test)
    
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=['Class 0 (Calm)', 'Class 1 (Elevated)', 'Class 2 (Shock)'],
        zero_division=0
    ))
    
    return model, features, X_test, y_test

def plot_feature_importance(model, feature_names):
    """
    Extracts and displays what the model actually learned.
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    
    # If there are many lag features, we only plot the top 25 to keep the chart readable
    top_n = 25
    if len(sorted_idx) > top_n:
        sorted_idx = sorted_idx[-top_n:]
        
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center', color='#2ca02c')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title(f'XGBoost Feature Importance (Top {top_n} Features)')
    plt.xlabel('Relative Importance (Information Gain)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
