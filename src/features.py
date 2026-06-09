import pandas as pd
import numpy as np
import logging
from typing import List
from statsmodels.tsa.ar_model import AutoReg
from src import config

logger = logging.getLogger(__name__)

def generate_macro_surprises(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculates the 'Economic Shock' by fitting a rolling AR(1) model.
    Residual = Actual Print - Expected Print.
    """
    surprises = pd.Series(index=series.index, dtype=float)
    # We only want to fit the model when the underlying monthly data ACTUALLY changes
    # to avoid fitting on forward-filled noise.
    unique_changes = series[series.diff() != 0].dropna()
    
    for i in range(window, len(unique_changes)):
        train_data = unique_changes.iloc[i-window : i]
        target_idx = unique_changes.index[i]
        actual_val = unique_changes.iloc[i]
        
        try:
            # Simple AR(1) to define expectation
            model = AutoReg(train_data, lags=1).fit()
            expected_val = model.forecast(steps=1).iloc[0]
            surprises.loc[target_idx] = actual_val - expected_val
        except:
            surprises.loc[target_idx] = 0.0
            
    # Forward fill the surprises to match daily frequency
    return surprises.ffill().fillna(0)

def engineer_stationary_features(raw_df: pd.DataFrame, etf_name: str) -> pd.DataFrame:
    """
    Transforms raw prices and creates True Macro Shocks dynamically.
    Includes robust column renaming to prevent cache/ticker mismatches.
    """
    logger.info(f"Engineering features and macro surprises for {etf_name}...")
    df = raw_df.copy()
    close_col = f'{etf_name}_Close'
    
    # --- 0. Safety Net: Standardize Column Names ---
    # In case the dataframe has FRED tickers (CPIAUCSL) instead of keys (CPI)
    monthly_ticker_to_name = {v: k for k, v in config.MONTHLY_MACRO.items()}
    daily_ticker_to_name = {v: k for k, v in config.DAILY_MACRO.items()}
    
    df.rename(columns=monthly_ticker_to_name, inplace=True)
    df.rename(columns=daily_ticker_to_name, inplace=True)
    
    # --- 1. ETF Math ---
    df['Log_Return'] = np.log(df[close_col] / df[close_col].shift(1))
    df['Sq_Log_Return'] = df['Log_Return'] ** 2
    
    # --- 2. Dynamic Daily Macro Math ---
    for name in config.DAILY_MACRO.keys():
        if name in df.columns:
            df[f'{name}_Change'] = df[name].pct_change()
            
    # --- 3. Dynamic True Macro Shocks ---
    logger.info("Calculating Autoregressive Macro Surprises...")
    for name in config.MONTHLY_MACRO.keys():
        if name in df.columns:
            lag = config.MACRO_LAGS.get(name, 1)
            
            # Smart Routing: Rates use simple diffs, Indices use AR(1) surprises
            if name in ['FedFunds', 'Term_Spread']:
                df[f'{name}_Diff'] = df[name].diff().shift(lag)
            else:
                df[f'{name}_Surprise'] = generate_macro_surprises(df[name]).shift(lag)
                
    # --- 4. The Purge ---
    columns_to_drop = [close_col] + list(config.DAILY_MACRO.keys()) + list(config.MONTHLY_MACRO.keys())
    cols_to_drop_safe = [col for col in columns_to_drop if col in df.columns]
    
    df = df.drop(columns=cols_to_drop_safe).dropna()
    
    return df

def create_lags(df: pd.DataFrame, feature_cols: List[str], lags: List[int] | None = None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 2, 3, 5]
        
    df_engineered = df.copy()
    for col in feature_cols:
        if col in df_engineered.columns:
            for lag in lags:
                df_engineered[f'{col}_lag{lag}'] = df_engineered[col].shift(lag)
                
    return df_engineered.dropna()
