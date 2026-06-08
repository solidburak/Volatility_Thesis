import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from src import config

logger = logging.getLogger(__name__)

def run_ablation_study(
    df: pd.DataFrame, 
    target_col: str = 'Target_HMM', 
    n_splits: int = 4
) -> Dict[str, Any]:
    """
    Executes the Advisor's Ablation Study.
    Dynamically sorts features into groups based on config dictionaries.
    """
    logger.info("Starting Advisor Ablation Study...")
    
    # 1. Dynamic Feature Groupings
    # Pull base names directly from the config dictionaries
    macro_base_names = list(config.DAILY_MACRO.keys()) + list(config.MONTHLY_MACRO.keys())
    
    volatility_features = [c for c in df.columns if 'Vol_' in c or 'Sq_Log_Return' in c]
    return_features = [c for c in df.columns if 'Log_Return' in c and 'Sq_' not in c]
    
    # Identify macro columns by checking if any config base name is a substring of the column name
    macro_features = [
        c for c in df.columns 
        if any(macro_name in c for macro_name in macro_base_names) and c != target_col
    ]
    
    # Ensure no future leakage in base features
    leakage = [target_col]
    
    experiments = {
        "1_AutoRegressive_Only": return_features + volatility_features,
        "2_Macro_Only": return_features + macro_features,
        "3_Full_Model": return_features + volatility_features + macro_features
    }
    
    results = {}
    
    # ... [Keep the rest of the XGBoost training loop exactly the same] ...    
    for exp_name, features in experiments.items():
        logger.info(f"\nEvaluating: {exp_name} ({len(features)} features)")
        
        # Clean features to exist in df
        valid_features = [f for f in features if f in df.columns and f not in leakage]
        
        X = df[valid_features]
        y = df[target_col]
        
        total_samples = len(df)
        block_size = int(total_samples / (n_splits + 1))
        
        all_preds = []
        all_targets = []
        
        # Using a simple parameter set for rapid ablation testing
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            max_depth=3,
            learning_rate=0.05,
            n_estimators=100,
            random_state=42
        )
        
        for i in range(1, n_splits + 1):
            train_end_idx = i * block_size
            test_end_idx = train_end_idx + block_size if i < n_splits else total_samples
            
            X_train, X_test = X.iloc[:train_end_idx], X.iloc[train_end_idx:test_end_idx]
            y_train, y_test = y.iloc[:train_end_idx], y.iloc[train_end_idx:test_end_idx]
            
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            preds = model.predict(X_test)
            all_preds.extend(preds)
            all_targets.extend(y_test)
            
        acc = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
        
        results[exp_name] = {
            "accuracy": acc,
            "macro_f1": report['macro avg']['f1-score'],
            "model": model,
            "features": valid_features
        }
        logger.info(f"Result -> Accuracy: {acc:.4f} | Macro F1: {report['macro avg']['f1-score']:.4f}")
        
    return results

def plot_ablation_importances(results: Dict[str, Any], top_n: int = 15):
    """
    Plots the feature importance of the 3 experiments side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Advisor Ablation Study: Feature Importances Across Models", fontsize=16)
    
    for idx, (exp_name, data) in enumerate(results.items()):
        model = data['model']
        feature_names = data['features']
        
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[-top_n:]
        
        axes[idx].barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
        axes[idx].set_yticks(range(len(sorted_idx)))
        axes[idx].set_yticklabels(np.array(feature_names)[sorted_idx])
        axes[idx].set_title(f"{exp_name}\nAcc: {data['accuracy']:.2f} | F1: {data['macro_f1']:.2f}")
        axes[idx].grid(axis='x', alpha=0.3)
        
    plt.tight_layout()
    plt.show()
