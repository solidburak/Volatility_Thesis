# src/models.py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

def train_xgboost_baseline(df, target_col='Target_Smooth_10d', test_size=0.2):
    """
    Trains the XGBoost baseline model with strict chronological splitting 
    and leakage prevention.
    """
    print(f"Preparing data. Target variable: {target_col}")
    
    # 1. Strict Data Leakage Prevention
    leakage_cols = ['Future_Sq_Return', 'Rolling_Thresh_Low', 'Rolling_Thresh_High', 'Target_Regime', target_col]
    features = [c for c in df.columns if c not in leakage_cols]
    
    X = df[features]
    y = df[target_col]
    
    # 2. Chronological Split
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")
    
    # 3. Handle Imbalance
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # 4. Train Model
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=250,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 5. Evaluate
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

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Visualizes the top predictive features of the XGBoost model.
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    
    if len(sorted_idx) > top_n:
        sorted_idx = sorted_idx[-top_n:]
        
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center', color='#2ca02c')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title(f'XGBoost Feature Importance (Top {top_n})')
    plt.xlabel('Relative Importance (Information Gain)')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_xgb_walk_forward(df, target_col='Target_Smooth_10d', n_splits=4):
    """
    Performs an expanding-window walk-forward evaluation.
    Example with n_splits=4:
    - Fold 1: Train on first 20%, Test on next 20%
    - Fold 2: Train on first 40%, Test on next 20%
    - Fold 3: Train on first 60%, Test on next 20%
    - Fold 4: Train on first 80%, Test on final 20%
    """
    print(f"Starting expanding-window Walk-Forward Validation ({n_splits} splits)...")
    
    # 1. Leakage Prevention & Feature Isolation
    leakage_cols = ['Future_Sq_Return', 'Rolling_Thresh_Low', 'Rolling_Thresh_High', 'Target_Regime', target_col]
    features = [c for c in df.columns if c not in leakage_cols]
    
    X = df[features]
    y = df[target_col]
    
    total_samples = len(df)
    # Each block/step size is calculated based on splits + 1 (e.g., 5 blocks of 20%)
    block_size = int(total_samples / (n_splits + 1))
    
    all_walk_forward_preds = []
    all_walk_forward_targets = []
    
    # 2. Walk Forward Loop
    for i in range(1, n_splits + 1):
        # Calculate dynamic training boundaries
        train_end_idx = i * block_size
        test_end_idx = train_end_idx + block_size if i < n_splits else total_samples
        
        X_train, X_test = X.iloc[:train_end_idx], X.iloc[train_end_idx:test_end_idx]
        y_train, y_test = y.iloc[:train_end_idx], y.iloc[train_end_idx:test_end_idx]
        
        print(f"\n--- Fold {i} ---")
        print(f"  Training on index 0 to {train_end_idx} (Samples: {len(X_train)})")
        print(f"  Testing on index {train_end_idx} to {test_end_idx} (Samples: {len(X_test)})")
        
        # Calculate balanced weights specifically for this fold's training history
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        # Initialize standard baseline model parameters
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            max_depth=4,
            learning_rate=0.05,
            n_estimators=250,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Train on the accumulated past data
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Predict on the incoming unseen fold
        preds = model.predict(X_test)

        print(classification_report(
            y_test, 
            preds, 
            labels=[0, 1, 2],
            target_names=['Class 0 (Calm)', 'Class 1 (Elevated)', 'Class 2 (Shock)'],
            zero_division=0
        ))
        
        # Store predictions and targets for global evaluation
        all_walk_forward_preds.extend(preds)
        all_walk_forward_targets.extend(y_test)
        
    # 3. Final Overall Evaluation
    # 3. Final Overall Evaluation
    print("\n" + "="*60)
    print("AGGREGATED WALK-FORWARD PERFORMANCE REPORT")
    print("="*60)
    
    print(classification_report(
        all_walk_forward_targets, 
        all_walk_forward_preds, 
        labels=[0, 1, 2],
        target_names=['Class 0 (Calm)', 'Class 1 (Elevated)', 'Class 2 (Shock)'],
        zero_division=0
    ))    
    # Return the final trained model from the last fold for feature importance analysis
    return model, features
