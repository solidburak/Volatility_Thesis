import pandas as pd
import numpy as np
import logging
from typing import List
from src import config

# --- Setup Module Logger ---
logger = logging.getLogger(__name__)

def engineer_stationary_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw prices and economic levels into stationary signals.
    Strictly applies reporting lags to macro data to prevent look-ahead bias.
    """
    logger.info("Engineering stationary features and applying macro lags...")
    df = raw_df.copy()
    
    # --- 1. ETF Math (Log Returns & Squared Returns) ---
    # R_t = ln(P_t / P_{t-1})
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Sq_Log_Return'] = df['Log_Return'] ** 2
    
    # --- 2. Daily Macro Math (Percentage Changes) ---
    df['VIX_Change'] = df['VIX'].pct_change()
    df['Oil_Change'] = df['Oil_WTI'].pct_change()
    
    # --- 3. Monthly Macro Math with STRICT Publication Lags ---
    # This solves the forward-filling look-ahead bias from the data loader.
    cpi_lag = config.MACRO_LAGS.get('CPI', 11)
    term_lag = config.MACRO_LAGS.get('Term_Spread', 1)
    fed_lag = config.MACRO_LAGS.get('FedFunds', 1)
    unemp_lag = config.MACRO_LAGS.get('Unemployment', 5)

    df['CPI_MoM'] = (df['CPI'].pct_change() * 100).shift(cpi_lag)
    df['Term_Spread_Diff'] = df['Term_Spread'].diff().shift(term_lag)
    df['FedFunds_Diff'] = df['FedFunds'].diff().shift(fed_lag)
    df['Unemployment_Diff'] = df['Unemployment'].diff().shift(unemp_lag)
    
    # --- 4. THE PURGE: Drop non-stationary and raw columns ---
    columns_to_drop = [
        'Close', 'VIX', 'Oil_WTI', 'CPI', 
        'FedFunds', 'Unemployment', 'Term_Spread'
    ]
    
    cols_to_drop_safe = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop_safe)
    
    # Drop the initial NaNs created by the math and the long lags
    df = df.dropna()
    
    logger.info(f"Stationary transformation complete. Final shape: {df.shape}")
    return df

def create_lags(df: pd.DataFrame, feature_cols: List[str], lags: List[int] = None) -> pd.DataFrame:
    """
    Creates lagged versions of specific features so ML models (like XGBoost)
    can 'see' historical context.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10]
        
    logger.info(f"Generating lag features for {len(feature_cols)} columns with lags: {lags} days...")
    df_engineered = df.copy()
    
    for col in feature_cols:
        if col in df_engineered.columns:
            for lag in lags:
                df_engineered[f'{col}_lag{lag}'] = df_engineered[col].shift(lag)
        else:
            logger.warning(f"Feature '{col}' not found in DataFrame. Skipping lags for this column.")
                
    final_df = df_engineered.dropna()
    logger.info(f"Lag generation complete. Output shape: {final_df.shape}")
    
    return final_df
