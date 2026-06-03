# src/features.py
import pandas as pd
import numpy as np

def engineer_stationary_features(raw_df):
    """
    Transforms raw prices and economic levels into stationary signals 
    (Returns, Differences, MoM changes) and drops the raw data.
    """
    print("Engineering stationary features...")
    df = raw_df.copy()
    
    # 1. ETF Math (Log Returns & Squared Returns)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Sq_Log_Return'] = df['Log_Return'] ** 2
    
    # 2. Daily Macro Math (Percentage Changes)
    df['VIX_Change'] = df['VIX'].pct_change()
    df['Oil_Change'] = df['Oil_WTI'].pct_change()
    
    # 3. Monthly Macro Math (Differences and MoM)
    df['CPI_MoM'] = df['CPI'].pct_change() * 100
    df['Term_Spread_Diff'] = df['Term_Spread'].diff()
    df['FedFunds_Diff'] = df['FedFunds'].diff()
    df['Unemployment_Diff'] = df['Unemployment'].diff()
    
    # 4. THE PURGE: Drop non-stationary and raw columns
    columns_to_drop = [
        'Close', 'VIX', 'Oil_WTI', 'CPI', 
        'FedFunds', 'Unemployment', 'Term_Spread'
    ]
    
    # Safely drop columns
    cols_to_drop_safe = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop_safe)
    
    # Drop initial NaNs created by the shifting/differencing
    df = df.dropna()
    
    print("Stationary transformation complete. Raw columns purged.")
    return df

def create_lags(df, feature_cols, lags=[1, 2, 3, 5, 10]):
    """
    Creates lagged versions of specific features so ML models (like XGBoost)
    can 'see' historical context.
    """
    print(f"Generating lag features for: {lags} days...")
    df_engineered = df.copy()
    
    for col in feature_cols:
        if col in df_engineered.columns:
            for lag in lags:
                df_engineered[f'{col}_lag{lag}'] = df_engineered[col].shift(lag)
                
    return df_engineered.dropna()
