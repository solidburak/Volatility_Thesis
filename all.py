==================================================
### FILE: config.py ###
==================================================

import os
import logging
from dotenv import load_dotenv
from typing import Dict

# --- Logging Configuration ---
# Industry standard: centralized logging setup replaces print() statements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Environment Variables ---
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    logging.warning("FRED_API_KEY is not set. FRED data ingestion will fail.")

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Project Parameters ---
START_DATE = '2000-01-01'
TARGET_ETF = 'XLK'

# --- Tickers & Macro Data ---
DAILY_MACRO: Dict[str, str] = {
    'VIX': '^VIX', 
    'Oil_WTI': 'CL=F'
}

MONTHLY_MACRO: Dict[str, str] = {
    'CPI': 'CPIAUCSL', 
    'FedFunds': 'FEDFUNDS', 
    'Unemployment': 'UNRATE',
    'Term_Spread': 'T10Y2Y'
}

# --- Macro Reporting Lags (in trading days) ---
# Configurable lags to prevent look-ahead bias. 
# You can update these manually based on exact historical publication schedules.
MACRO_LAGS: Dict[str, int] = {
    'CPI': 11,           # ~11 trading days into the following month
    'FedFunds': 1,       # Known immediately, 1 day lag to be safe for market open
    'Unemployment': 5,   # First Friday of the following month (~5 trading days)
    'Term_Spread': 1     # Daily series, 1 day lag is safe
}



==================================================
### FILE: __init__.py ###
==================================================

# src/__init__.py
"""
Volatility Regime Detection Project
Core source code package.
"""
# We leave this intentionally blank to avoid circular dependencies.
# Modules should be imported explicitly in your notebooks 
# (e.g., 'from src import data_loader')



==================================================
### FILE: data_loader.py ###
==================================================

import os
import time
import logging
import pandas as pd
import yfinance as yf
from fredapi import Fred
from src import config

# --- Setup Module Logger ---
logger = logging.getLogger(__name__)

def get_fred_series_with_cache(
    fred_client: Fred, 
    series_id: str, 
    series_name: str, 
    start_date: str
) -> pd.DataFrame:
    """
    Fetches individual FRED series with local caching to prevent Rate Limit errors.
    """
    cache_path = os.path.join(config.RAW_DATA_DIR, f"fred_cache_{series_id}.parquet")
    
    if os.path.exists(cache_path):
        logger.info(f"Loaded {series_name} ({series_id}) from local cache.")
        return pd.read_parquet(cache_path)
    
    logger.info(f"Fetching {series_name} ({series_id}) from FRED API...")
    try:
        time.sleep(1.5)  # Polite delay to strictly avoid rate limits
        series = fred_client.get_series(series_id, observation_start=start_date)
        df = pd.DataFrame(series, columns=[series_name])
        
        # Save to cache
        df.to_parquet(cache_path)
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch {series_id}: {str(e)}")
        # Return an empty dataframe to keep the pipeline moving if one series fails
        return pd.DataFrame()

def fetch_raw_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    The master data ingestion function. 
    Downloads ETF and Macro data based on config.py and merges them into a daily index.
    """
    master_cache = os.path.join(config.RAW_DATA_DIR, "master_raw_data.parquet")
    
    if os.path.exists(master_cache) and not force_refresh:
        logger.info(f"Loading cached master raw data from {master_cache}...")
        df = pd.read_parquet(master_cache)
        df.index = pd.to_datetime(df.index)
        return df
        
    logger.info("Starting raw data ingestion pipeline...")
    
    # --- Initialize FRED ---
    if not config.FRED_API_KEY:
        raise ValueError("FRED_API_KEY is missing. Check your .env file or config.")
    fred = Fred(api_key=config.FRED_API_KEY)
    
    # --- 1. Download Target ETF (XLK) ---
    logger.info(f"Fetching Target ETF: {config.TARGET_ETF}...")
    etf_df = yf.download(config.TARGET_ETF, start=config.START_DATE, progress=False)
    
    if isinstance(etf_df.columns, pd.MultiIndex):
        etf_df.columns = etf_df.columns.get_level_values(0)
    
    # Isolate the Close price
    base_df = pd.DataFrame({'Close': etf_df['Close']})
    
    # --- 2. Download Daily Macro (VIX, Oil) ---
    logger.info("Fetching Daily Macro Indicators from yfinance...")
    daily_tickers = list(config.DAILY_MACRO.values())
    daily_macro_raw = yf.download(daily_tickers, start=config.START_DATE, progress=False)
    
    # Extract 'Close' and handle potential MultiIndex from yfinance
    if isinstance(daily_macro_raw.columns, pd.MultiIndex):
        daily_macro = daily_macro_raw['Close'].copy()
    else:
        daily_macro = daily_macro_raw.copy()
        
    # Rename columns to our clean mapping names
    ticker_to_name = {v: k for k, v in config.DAILY_MACRO.items()}
    daily_macro.rename(columns=ticker_to_name, inplace=True)
    
    # --- 3. Download Monthly Macro (FRED) ---
    logger.info("Fetching Macro Indicators from FRED...")
    fred_data = {}
    for name, series_id in config.MONTHLY_MACRO.items():
        df_raw = get_fred_series_with_cache(fred, series_id, name, config.START_DATE)
        if not df_raw.empty:
            fred_data[name] = df_raw
            
    monthly_macro = pd.concat(fred_data.values(), axis=1)
    
    # --- 4. Merge Everything Together ---
    logger.info("Merging datasets to daily trading calendar...")
    # Join everything to the ETF trading days
    master_df = base_df.join(daily_macro, how='left')
    master_df = master_df.join(monthly_macro, how='left')
    
    # Forward-fill the monthly macro data so every trading day has the latest *reported* value
    # Note: Look-ahead bias shifting will be handled strictly in features.py
    master_df = master_df.ffill().dropna()
    
    # Save the un-transformed raw data
    master_df.to_parquet(master_cache)
    logger.info(f"Raw data ingestion complete. Saved to {master_cache}. Shape: {master_df.shape}")
    
    return master_df



==================================================
### FILE: features.py ###
==================================================

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



==================================================
### FILE: volatility.py ###
==================================================

import pandas as pd
import numpy as np
import warnings
import logging
from arch import arch_model

# --- Setup Module Logger ---
logger = logging.getLogger(__name__)

# Suppress arch convergence warnings to keep the logs clean during expanding windows
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def fit_static_models_for_research(returns_series: pd.Series) -> pd.DataFrame:
    """
    Fits GARCH(1,1), EGARCH, and GJR-GARCH on the entire dataset.
    Strictly for AIC/BIC comparison in EDA. Do not use outputs as ML features.
    """
    logger.info("Fitting static GARCH models for AIC/BIC comparison...")
    returns_scaled = returns_series.dropna() * 100
    results = {}
    
    # Standard GARCH
    garch = arch_model(returns_scaled, vol='GARCH', p=1, q=1, mean='Constant')
    garch_fit = garch.fit(disp='off', show_warning=False)
    results['GARCH(1,1)'] = {'AIC': garch_fit.aic, 'BIC': garch_fit.bic}
    
    # EGARCH (Exponential)
    egarch = arch_model(returns_scaled, vol='EGARCH', p=1, o=1, q=1, mean='Constant')
    egarch_fit = egarch.fit(disp='off', show_warning=False)
    results['EGARCH(1,1)'] = {'AIC': egarch_fit.aic, 'BIC': egarch_fit.bic}
    
    # GJR-GARCH (Threshold)
    gjr = arch_model(returns_scaled, vol='GARCH', p=1, o=1, q=1, mean='Constant')
    gjr_fit = gjr.fit(disp='off', show_warning=False)
    results['GJR-GARCH(1,1)'] = {'AIC': gjr_fit.aic, 'BIC': gjr_fit.bic}
    
    comparison_df = pd.DataFrame(results).T.sort_values(by='AIC')
    return comparison_df

def generate_expanding_garch(
    df: pd.DataFrame, 
    return_col: str = 'Log_Return', 
    window_size: int = 252, 
    refit_freq: int = 21
) -> pd.DataFrame:
    """
    Industry-Standard ML-Safe GARCH & EGARCH.
    Balances strict out-of-sample ML safety with computational speed by 
    re-fitting the optimizer periodically.
    """
    logger.info(f"Generating Expanding GARCH & EGARCH (Window: {window_size}, Refit: {refit_freq} days)...")
    df_local = df.copy()
    
    # Scale returns by 100 for optimizer convergence stability
    returns = df_local[return_col] * 100
    n_obs = len(returns)
    
    # Pre-allocate output series with NaNs
    vol_garch = pd.Series(np.nan, index=returns.index)
    vol_egarch = pd.Series(np.nan, index=returns.index)
    
    # Loop through the timeline in chunks of 'refit_freq'
    for i in range(window_size, n_obs, refit_freq):
        
        train_window = returns.iloc[i - window_size : i]
        chunk_end = min(i + refit_freq, n_obs)
        eval_data = returns.iloc[i - window_size : chunk_end]
        
        # -------------------------------------------------------------
        # A. STANDARD GARCH(1,1)
        # -------------------------------------------------------------
        train_mod_g = arch_model(train_window, vol='GARCH', p=1, q=1, mean='Constant', rescale=False)
        try:
            train_res_g = train_mod_g.fit(disp='off', show_warning=False)
            eval_mod_g = arch_model(eval_data, vol='GARCH', p=1, q=1, mean='Constant', rescale=False)
            fixed_res_g = eval_mod_g.fix(train_res_g.params)
            
            chunk_vol_g = np.asarray(fixed_res_g.conditional_volatility)[window_size:]
            vol_garch.iloc[i : chunk_end] = chunk_vol_g
        except Exception as e:
            logger.debug(f"GARCH optimizer failed at index {i}: {e}")
            pass
            
        # -------------------------------------------------------------
        # B. EGARCH(1,1) - Critical for the Leverage Effect
        # -------------------------------------------------------------
        train_mod_eg = arch_model(train_window, vol='EGARCH', p=1, o=1, q=1, mean='Constant', rescale=False)
        try:
            train_res_eg = train_mod_eg.fit(disp='off', show_warning=False)
            eval_mod_eg = arch_model(eval_data, vol='EGARCH', p=1, o=1, q=1, mean='Constant', rescale=False)
            fixed_res_eg = eval_mod_eg.fix(train_res_eg.params)
            
            chunk_vol_eg = np.asarray(fixed_res_eg.conditional_volatility)[window_size:]
            vol_egarch.iloc[i : chunk_end] = chunk_vol_eg
        except Exception as e:
            logger.debug(f"EGARCH optimizer failed at index {i}: {e}")
            pass

    # Unscale the final volatility to match original return decimals
    df_local['Vol_GARCH'] = vol_garch / 100
    df_local['Vol_EGARCH'] = vol_egarch / 100

    # -------------------------------------------------------------
    # C. DYNAMIC WINSORIZATION (Replaces the hard 0.20 cap)
    # -------------------------------------------------------------
    # Cap mathematical explosions at 3x the maximum absolute daily return 
    # seen in the trailing window. This handles crashes natively.
    trailing_max_return = df_local[return_col].abs().rolling(window=window_size, min_periods=window_size//2).max()
    dynamic_cap = trailing_max_return * 3.0
    
    df_local['Vol_GARCH'] = df_local['Vol_GARCH'].clip(upper=dynamic_cap)
    df_local['Vol_EGARCH'] = df_local['Vol_EGARCH'].clip(upper=dynamic_cap)
    
    final_df = df_local.dropna(subset=['Vol_GARCH', 'Vol_EGARCH'])
    logger.info(f"Out-of-sample generation complete. Shape: {final_df.shape}")
    
    return final_df



==================================================
### FILE: regimes.py ###
==================================================

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



==================================================
### FILE: models_xgb.py ###
==================================================

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



==================================================
### FILE: models_lstm.py ###
==================================================

# src/models_lstm.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Dataset Class (Sliding Window Generation)
# ---------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, features, target, sequence_length):
        """
        features: numpy array shape (Time, Features)
        target: numpy array shape (Time,)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, index):
        # Grab a window of past 'sequence_length' days
        x = self.features[index : index + self.sequence_length]
        # Predict the target for the day immediately AFTER the window
        y = self.target[index + self.sequence_length]
        return x, y

# ---------------------------------------------------------
# 2. LSTM Architecture
# ---------------------------------------------------------
class RegimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=3, dropout=0.3):
        super(RegimeLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The LSTM Core
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the *last* time step only
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out) # Returns shape: (batch_size, 3)
        
        return out

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def train_and_evaluate_lstm(df, feature_cols, target_col='Target_Smooth_10d', seq_length=21, epochs=30, lr=0.001, test_size=0.2):
    """
    Handles scaling, splitting, weighting, and training the PyTorch LSTM.
    """
    print(f"Preparing Data for LSTM (Sequence Length: {seq_length} days)...")
    
    # 1. Isolate Data
    # Drop rows with NaNs to ensure contiguous sequences
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 2. Chronological Split
    split_idx = int(len(df) * (1 - test_size))
    
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 3. Scaling (Critical for Neural Networks)
    # Fit scaler ONLY on training data to prevent leakage
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 4. Create DataLoaders
    train_dataset = SequenceDataset(X_train, y_train, seq_length)
    test_dataset = SequenceDataset(X_test, y_test, seq_length)
    
    # shuffle=True is generally safe here because the sequences themselves preserve time
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 5. Handle Class Imbalance with PyTorch Weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print("\nCalculated Class Weights (Calm, Elevated, Shock):")
    print(np.round(class_weights, 4))
    
    # 6. Initialize Model
    model = RegimeLSTM(
        input_size=len(feature_cols), 
        hidden_size=32, 
        num_layers=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Added weight decay for regularization
    
    # 7. Training Phase
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {total_loss/len(train_loader):.4f}")
            
    # 8. Evaluation Phase
    print("\n" + "="*50)
    print("LSTM Evaluation on Unseen Test Data")
    print("="*50)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            
            _, preds = torch.max(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=['Class 0 (Calm)', 'Class 1 (Elevated)', 'Class 2 (Shock)'],
        zero_division=0
    ))
    
    return model, scaler


==================================================
### FILE: 01_volatility_analysis.ipynb ###
==================================================

# 1. Imports and Setup
import sys
import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch
from IPython.display import display

# Ensure the notebook can find the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src import data_loader, features, volatility, config

# 2. Ingest Data & Engineer Stationary Features
# The macro lags configured in config.py are automatically applied here!
print(f"Loading data for {config.TARGET_ETF}...")
raw_df = data_loader.fetch_raw_data(force_refresh=False)
stat_df = features.engineer_stationary_features(raw_df)

# 3. Prove Volatility Clustering (ARCH-LM Test)
returns = stat_df['Log_Return']
returns_mean_adj = returns - returns.mean()
sq_returns = stat_df['Sq_Log_Return']

lm_stat, p_value, f_stat, fp_value = het_arch(returns_mean_adj, nlags=5)
print(f"ARCH-LM Test p-value: {p_value:.4e}")
if p_value < 0.05:
    print("Conclusion: Volatility clustering is present. GARCH modeling is justified.")

# 4. Visualize the Persistence of Volatility
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(returns.index, returns, color='#1f77b4', linewidth=0.5)
axes[0].set_title(f"{config.TARGET_ETF} Daily Log Returns (Visual Clustering)")

plot_acf(sq_returns, lags=50, ax=axes[1], color='#d62728')
axes[1].set_title("ACF of Squared Returns (Persistence)")
plt.tight_layout()
plt.show()

# 5. Model Selection (AIC/BIC Leaderboard)
# We use the static fitting function strictly for this theoretical comparison
leaderboard = volatility.fit_static_models_for_research(returns)
print("\n--- GARCH Model Leaderboard ---")
display(leaderboard)





==================================================
### FILE: 02_regime_engineering.ipynb ###
==================================================

# 1. Imports
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src import data_loader, features, volatility, regimes, config

# 2. Standard Ingest and ML-Safe Features
raw_df = data_loader.fetch_raw_data()
stat_df = features.engineer_stationary_features(raw_df)

# Note: Dynamic Winsorization natively handles Black Swans now
garch_df = volatility.generate_expanding_garch(stat_df, window_size=252, refit_freq=21)

# 3. Generate Targets
smooth_df = regimes.generate_smoothed_targets(
    garch_df, lower_quant=0.85, upper_quant=0.95, window=252
)

# New: Industry-Standard Expanding HMM
# Using a 504-day (2 year) initial window to give the HMM plenty of history 
# before it starts making out-of-sample predictions.
hmm_df = regimes.generate_expanding_hmm_targets(
    garch_df, 
    feature_cols=['Log_Return', 'Vol_EGARCH'], 
    vol_col='Vol_EGARCH',
    n_components=3,
    min_window=504,
    refit_freq=21
)

# 4. Robust Concatenation via Inner Join
comparison_df = pd.concat([garch_df['Log_Return'], smooth_df, hmm_df], axis=1, join='inner')

# Print Final Distributions
print("="*50)
print("Target Class Distributions (Strictly Out-of-Sample)")
print("="*50)
print("\nMethod A: 10-day Smoothed Targets:")
print(comparison_df['Target_Smooth_10d'].value_counts().sort_index())

print("\nMethod B: Expanding 3-State HMM Targets:")
print(comparison_df['Target_HMM'].value_counts().sort_index())


# ---------------------------------------------------------
# Plotting Function
# ---------------------------------------------------------
def plot_regime_comparison(df, title_suffix="XLK"):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    color_map = {1: ('orange', 0.3), 2: ('red', 0.5)}
    
    # TOP: Smoothed
    axes[0].plot(df.index, df['Log_Return'], color='black', linewidth=0.5)
    diff_smooth = df['Target_Smooth_10d'].diff().fillna(0)
    changes_smooth = df[diff_smooth != 0].index.tolist()
    changes_smooth.insert(0, df.index[0])
    changes_smooth.append(df.index[-1])
    
    for i in range(len(changes_smooth)-1):
        start, end = changes_smooth[i], changes_smooth[i+1]
        regime = df.loc[start, 'Target_Smooth_10d']
        if regime in color_map:
            color, alpha = color_map[regime]
            axes[0].axvspan(start, end, color=color, alpha=alpha)
            
    axes[0].set_title(f"Method A: Manual Rolling Thresholds ({title_suffix})", fontsize=14)
    axes[0].set_ylabel("Log Return")
    
    # BOTTOM: HMM
    axes[1].plot(df.index, df['Log_Return'], color='black', linewidth=0.5)
    diff_hmm = df['Target_HMM'].diff().fillna(0)
    changes_hmm = df[diff_hmm != 0].index.tolist()
    changes_hmm.insert(0, df.index[0])
    changes_hmm.append(df.index[-1])
    
    for i in range(len(changes_hmm)-1):
        start, end = changes_hmm[i], changes_hmm[i+1]
        regime = df.loc[start, 'Target_HMM']
        if regime in color_map:
            color, alpha = color_map[regime]
            axes[1].axvspan(start, end, color=color, alpha=alpha)
            
    axes[1].set_title(f"Method B: Expanding Hidden Markov Model ({title_suffix})", fontsize=14)
    axes[1].set_ylabel("Log Return")
    
    plt.tight_layout()
    plt.show()

plot_regime_comparison(comparison_df)


# ---------------------------------------------------------
# Testing the 4-State HMM Mapped to 3 Classes
# ---------------------------------------------------------
print("\n--- Generating 4-State HMM ---")
hmm_4_df = regimes.generate_expanding_hmm_targets(
    garch_df, 
    feature_cols=['Log_Return', 'Vol_EGARCH'], 
    vol_col='Vol_EGARCH',
    n_components=4,
    min_window=504,
    refit_freq=21
)

# Merge
comparison_4_df = pd.concat([garch_df['Log_Return'], smooth_df, hmm_4_df], axis=1, join='inner')

# Map the 4-state HMM down to 3 classes
# Because the src code guarantees State 0 is always lowest vol and State 3 is highest,
# this mapping is completely safe and won't be scrambled by label switching.
mapping_dict = {
    0: 0,  # Dead Calm -> Calm
    1: 0,  # Normal -> Calm
    2: 1,  # Elevated
    3: 2   # Shock
}

comparison_4_df['Target_HMM'] = comparison_4_df['Target_HMM'].map(mapping_dict)

print("\n" + "="*50)
print("FINAL MACHINE LEARNING TARGET DISTRIBUTION (4-State mapped to 3)")
print("="*50)
print(comparison_4_df['Target_HMM'].value_counts().sort_index())

plot_regime_comparison(comparison_4_df, title_suffix="XLK - 4-State Mapped to 3")





==================================================
### FILE: 03_baseline_xgboost.ipynb ###
==================================================

from src import data_loader, features, volatility, regimes, models_xgb

# 1. Ingest & Transform
raw_df = data_loader.fetch_raw_data()
stat_df = features.engineer_stationary_features(raw_df)
garch_df = volatility.generate_expanding_garch(stat_df)

# 2. HMM Target Generation (Now immune to Label Switching)
hmm_df = regimes.generate_expanding_hmm_targets(
    garch_df,
    min_window=504,
    feature_cols=['Log_Return', 'Vol_EGARCH'], 
    vol_col='Vol_EGARCH',
    n_components=3
)

# 3. Merge & Lag
master_df = garch_df.join(hmm_df, how='inner')
cols_to_lag = [
    'Log_Return', 'Sq_Log_Return', 'Vol_GARCH', 'Vol_EGARCH', 
    'VIX_Change', 'Oil_Change', 'CPI_MoM', 'FedFunds_Diff', 'Term_Spread_Diff'
]
final_df = features.create_lags(master_df, feature_cols=cols_to_lag)

# 4. Train & Evaluate
final_model, feature_names = models_xgb.train_xgb_walk_forward(
    df=final_df, 
    target_col='Target_HMM', 
    n_splits=4,
    use_mlflow=False # Set to True once you `pip install mlflow`
)

# 5. Visualize
models_xgb.plot_feature_importance(final_model, feature_names)
