import pandas as pd
import numpy as np
import logging
import warnings
from hmmlearn import hmm
from typing import List
from sklearn.preprocessing import StandardScaler

# --- Setup Module Logger ---
logger = logging.getLogger(__name__)

# Suppress the tiny convergence warning from hmmlearn
warnings.filterwarnings("ignore", category=UserWarning, module='hmmlearn')
warnings.filterwarnings("ignore", category=RuntimeWarning)

def generate_smoothed_targets(
    df: pd.DataFrame, 
    lower_quant: float = 0.86, 
    upper_quant: float = 0.95, 
    window: int = 252, 
    floor: float = 0.0, 
    shock_smooth: int = 3, 
    elevated_smooth: int = 4
) -> pd.DataFrame:
    """
    Heuristic 10-day smoothed target generator based on rolling quantiles.
    """
    logger.info("Generating rolling heuristic regimes and smoothed targets...")
    df_local = df.copy()
    
    # 1. Target Variable: Tomorrow's Volatility
    df_local['Future_Sq_Return'] = df_local['Sq_Log_Return'].shift(-1)
    
    # 2. Dynamic Thresholds
    df_local['Rolling_Thresh_Low'] = df_local['Sq_Log_Return'].rolling(window=window).quantile(lower_quant)
    df_local['Rolling_Thresh_High'] = df_local['Sq_Log_Return'].rolling(window=window).quantile(upper_quant)
    
    df_local = df_local.dropna(subset=['Rolling_Thresh_Low', 'Rolling_Thresh_High', 'Future_Sq_Return'])
    
    # 3. Vectorized Classification
    conditions = [
        (df_local['Future_Sq_Return'] <= df_local['Rolling_Thresh_Low']) | (df_local["Future_Sq_Return"] <= floor),
        (df_local['Future_Sq_Return'] > df_local['Rolling_Thresh_Low']) & (df_local['Future_Sq_Return'] <= df_local['Rolling_Thresh_High']),
        df_local['Future_Sq_Return'] > df_local['Rolling_Thresh_High']
    ]
    
    choices = [0, 1, 2] 
    df_local['Target_Regime'] = np.select(conditions, choices, default=np.nan)
    df_local = df_local.dropna(subset=['Target_Regime'])
    df_local['Target_Regime'] = df_local['Target_Regime'].astype(int)

    # 4. Target Density Smoothing
    is_shock = (df_local['Target_Regime'] == 2).astype(int)
    is_elevated = (df_local['Target_Regime'] == 1).astype(int)

    shock_density = is_shock.rolling(10).sum()
    elevated_density = is_elevated.rolling(10).sum()

    df_local['Target_Smooth_10d'] = 0
    df_local.loc[elevated_density >= elevated_smooth, 'Target_Smooth_10d'] = 1
    df_local.loc[shock_density >= shock_smooth, 'Target_Smooth_10d'] = 2

    df_local = df_local.dropna(subset=['Target_Smooth_10d'])
    
    logger.info(f"Smoothed heuristic targets generated. Shape: {df_local[['Target_Smooth_10d']].shape}")
    return df_local[['Target_Smooth_10d']]


def generate_expanding_hmm_targets(
    df: pd.DataFrame, 
    feature_cols: List[str] = ['Log_Return', 'Vol_EGARCH'], 
    vol_col: str = 'Vol_EGARCH',
    n_components: int = 3,
    min_window: int = 252,
    refit_freq: int = 21
) -> pd.DataFrame:
    """
    Industry-Standard Expanding HMM target generator.
    Expands the historical window to ensure enough regime diversity for convergence,
    and refits periodically to optimize compute speed.
    """
    logger.info(f"Starting Expanding HMM (Min Window: {min_window}, Refit: {refit_freq} days)...")
    hmm_data = df.dropna(subset=feature_cols).copy()
    
    if vol_col not in feature_cols:
        raise ValueError(f"vol_col '{vol_col}' must be included in feature_cols.")
        
    vol_col_idx = feature_cols.index(vol_col)
    hmm_data['Target_HMM'] = np.nan
    n_obs = len(hmm_data)
    
    # Pre-allocate variables to hold the last known good mapping and scaler
    last_good_model = None
    last_good_scaler = None
    last_good_mapping = None
    
    for i in range(min_window, n_obs, refit_freq):
        # 1. EXPANDING WINDOW: From day 0 up to day i
        raw_train_window = hmm_data[feature_cols].iloc[0 : i].values
        chunk_end = min(i + refit_freq, n_obs)
        
        # 2. Standardize to stabilize the optimizer
        scaler = StandardScaler()
        train_window_scaled = scaler.fit_transform(raw_train_window)
        
        # 3. Fit Model
        model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type="full", 
            n_iter=200, 
            random_state=42
        )
        
        model.fit(train_window_scaled)
        
        # 4. Strict Convergence Check
        if model.monitor_.converged:
            # Map the states based on actual volatility
            train_states = model.predict(train_window_scaled)
            state_vols = {}
            for c in range(n_components):
                mask = (train_states == c)
                state_vols[c] = np.mean(raw_train_window[mask, vol_col_idx]) if np.any(mask) else -1.0
                
            sorted_states = sorted(state_vols.keys(), key=lambda k: state_vols[k])
            mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_states)}
            
            # Save these as our "Last Known Good" configuration
            last_good_model = model
            last_good_scaler = scaler
            last_good_mapping = mapping
        else:
            logger.warning(f"HMM failed to converge at index {i}. Using previous successful parameters.")
            if last_good_model is None:
                # If it fails on the very first window, we skip the chunk
                continue
        
        # 5. Predict the Out-Of-Sample Chunk using the Best Known Model
        active_model = last_good_model if last_good_model else model
        active_scaler = last_good_scaler if last_good_scaler else scaler
        active_mapping = last_good_mapping if last_good_mapping else {0:0, 1:1, 2:2}
        
        eval_data_raw = hmm_data[feature_cols].iloc[i : chunk_end].values
        eval_data_scaled = active_scaler.transform(eval_data_raw)
        
        raw_states = active_model.predict(eval_data_scaled)
        mapped_states = [active_mapping.get(state, np.nan) for state in raw_states]
        
        hmm_data.iloc[i : chunk_end, hmm_data.columns.get_loc('Target_HMM')] = mapped_states

    # Clean up, shift backward for prediction, and type correctly
    hmm_data['Target_HMM'] = hmm_data['Target_HMM'].shift(-1)
    
    final_df = hmm_data.dropna(subset=['Target_HMM']).copy()
    final_df['Target_HMM'] = final_df['Target_HMM'].astype(int)
    
    logger.info(f"Expanding HMM Target generation complete. Shape: {final_df[['Target_HMM']].shape}")
    
    return final_df[['Target_HMM']]
