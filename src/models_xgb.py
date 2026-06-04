import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import mlflow
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# --- Setup Module Logger ---
logger = logging.getLogger(__name__)

def train_xgb_walk_forward(
    df: pd.DataFrame, 
    target_col: str = 'Target_HMM', 
    leakage_cols: Optional[List[str]] = None,
    n_splits: int = 4,
    xgb_params: Optional[Dict[str, Any]] = None,
    use_mlflow: bool = False
) -> Tuple[xgb.XGBClassifier, List[str]]:
    """
    Performs an expanding-window walk-forward evaluation for XGBoost.
    Evaluates strictly on out-of-sample data and optionally logs metrics to MLflow.
    """
    logger.info(f"Starting Walk-Forward Validation ({n_splits} splits) for target: {target_col}")
    
    # Default parameters if none provided
    if xgb_params is None:
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 250,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

    # 1. Feature Isolation & Leakage Prevention
    if leakage_cols is None:
        # Standard leakage columns to drop
        leakage_cols = ['Future_Sq_Return', 'Rolling_Thresh_Low', 'Rolling_Thresh_High', 'Target_Regime', target_col]
        
    features = [c for c in df.columns if c not in leakage_cols]
    
    X = df[features]
    y = df[target_col]
    total_samples = len(df)
    block_size = int(total_samples / (n_splits + 1))
    
    all_preds = []
    all_targets = []
    fold_metrics = []

    # Optional MLflow Tracking
    if use_mlflow:
        mlflow.set_experiment("Volatility_Regime_Prediction")
        mlflow.start_run()
        mlflow.log_params(xgb_params)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("features_count", len(features))

    # 2. Walk Forward Loop
    for i in range(1, n_splits + 1):
        train_end_idx = i * block_size
        test_end_idx = train_end_idx + block_size if i < n_splits else total_samples
        
        X_train, X_test = X.iloc[:train_end_idx], X.iloc[train_end_idx:test_end_idx]
        y_train, y_test = y.iloc[:train_end_idx], y.iloc[train_end_idx:test_end_idx]
        
        logger.info(f"Fold {i}/{n_splits} | Train: 0 to {train_end_idx} | Test: {train_end_idx} to {test_end_idx}")
        
        # Balanced weights dynamically calculated for this fold's history
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Predict classes and probabilities
        preds = model.predict(X_test)
        prob_preds = model.predict_proba(X_test)
        
        # Fold Metrics
        fold_acc = accuracy_score(y_test, preds)
        fold_loss = log_loss(y_test, prob_preds)
        fold_metrics.append({'accuracy': fold_acc, 'log_loss': fold_loss})
        
        logger.info(f"Fold {i} Results -> Accuracy: {fold_acc:.4f} | Log Loss: {fold_loss:.4f}")

        # Generate and log the classification report
        report = classification_report(
            y_test, 
            preds, 
            labels=[0, 1, 2],
            target_names=['Class 0 (Calm)', 'Class 1 (Elevated)', 'Class 2 (Shock)'],
            zero_division=0
        )
        print(report) # Printed for immediate notebook visualization
        
        all_preds.extend(preds)
        all_targets.extend(y_test)

    # 3. Aggregated Global Evaluation
    global_acc = accuracy_score(all_targets, all_preds)
    logger.info("=" * 60)
    logger.info("AGGREGATED WALK-FORWARD PERFORMANCE")
    logger.info("=" * 60)
    
    # Generate and log the classification report
    report = classification_report(
        all_targets, 
        all_preds, 
        labels=[0, 1, 2],
        target_names=['Class 0 (Calm)', 'Class 1 (Elevated)', 'Class 2 (Shock)'],
        zero_division=0
    )
    print(report) # Printed for immediate notebook visualization
    
    # Finalize MLflow tracking
    if use_mlflow:
        mlflow.log_metric("global_accuracy", global_acc)
        mlflow.log_metric("avg_fold_log_loss", np.mean([m['log_loss'] for m in fold_metrics]))
        mlflow.end_run()

    # Return the model trained on the final fold for feature importance extraction
    return model, features

def plot_feature_importance(model: xgb.XGBClassifier, feature_names: List[str], top_n: int = 20) -> None:
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
