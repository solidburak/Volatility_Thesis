# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

# Directory Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Project Parameters
START_DATE = '2000-01-01'
TARGET_ETF = 'XLK'

# Macroeconomic Tickers (yfinance)
DAILY_MACRO = {
    'VIX': '^VIX', 
    'Oil_WTI': 'CL=F'
}

# Macroeconomic Series (FRED)
MONTHLY_MACRO = {
    'CPI': 'CPIAUCSL', 
    'FedFunds': 'FEDFUNDS', 
    'Unemployment': 'UNRATE',
    'Term_Spread': 'T10Y2Y'
}

# src/data_loader.py
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from src import config  # Importing our centralized parameters

def get_fred_series_with_cache(fred_client, series_id, start_date):
    """
    Fetches individual FRED series with local caching to prevent Rate Limit errors.
    """
    cache_path = os.path.join(config.RAW_DATA_DIR, f"fred_cache_{series_id}.parquet")
    
    if os.path.exists(cache_path):
        print(f"  -> Loaded {series_id} from local cache.")
        return pd.read_parquet(cache_path)
    
    print(f"  -> Fetching {series_id} from FRED API...")
    try:
        time.sleep(1) # Polite 1-second delay for API
        series = fred_client.get_series(series_id, observation_start=start_date)
        df = pd.DataFrame(series, columns=[series_id])
        df.to_parquet(cache_path)
        return df
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {series_id}: {e}")
        return pd.DataFrame()

def fetch_raw_data(force_refresh=False):
    """
    The master data ingestion function. 
    Downloads ETF and Macro data based on config.py and merges them.
    """
    master_cache = os.path.join(config.RAW_DATA_DIR, "master_raw_data.parquet")
    
    if os.path.exists(master_cache) and not force_refresh:
        print(f"Loading cached raw data from {master_cache}...")
        df = pd.read_parquet(master_cache)
        df.index = pd.to_datetime(df.index)
        return df
        
    print("Starting raw data ingestion pipeline...")
    
    # Initialize FRED
    if not config.FRED_API_KEY:
        raise ValueError("FRED_API_KEY is missing. Check your .env file.")
    fred = Fred(api_key=config.FRED_API_KEY)
    
    # 1. Download Target ETF (XLK)
    print(f"Fetching ETF: {config.TARGET_ETF}...")
    etf_df = yf.download(config.TARGET_ETF, start=config.START_DATE, progress=False)
    if isinstance(etf_df.columns, pd.MultiIndex):
        etf_df.columns = etf_df.columns.get_level_values(0)
    
    # We only need the Close price for the raw dataset
    base_df = pd.DataFrame({'Close': etf_df['Close']})
    
    # 2. Download Daily Macro (VIX, Oil)
    print("Fetching Daily Macro Indicators...")
    daily_tickers = list(config.DAILY_MACRO.values())
    daily_macro = yf.download(daily_tickers, start=config.START_DATE, progress=False)['Close']
    
    # Rename columns to our clean names
    ticker_to_name = {v: k for k, v in config.DAILY_MACRO.items()}
    daily_macro.rename(columns=ticker_to_name, inplace=True)
    
    # 3. Download Monthly Macro (FRED)
    print("Fetching FRED Macro Data...")
    fred_data = {}
    for name, series_id in config.MONTHLY_MACRO.items():
        df_raw = get_fred_series_with_cache(fred, series_id, config.START_DATE)
        if not df_raw.empty:
            fred_data[name] = df_raw.rename(columns={series_id: name})
            
    monthly_macro = pd.concat(fred_data.values(), axis=1)
    
    # 4. Merge Everything Together
    print("Merging datasets...")
    master_df = base_df.join(daily_macro, how='left')
    master_df = master_df.join(monthly_macro, how='left').ffill()
    master_df = master_df.dropna()
    
    # Save the un-transformed raw data
    master_df.to_parquet(master_cache)
    print(f"Raw data ingestion complete. Saved to {master_cache}")
    
    return master_df

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

# src/volatility.py
import pandas as pd
import numpy as np
import warnings
from arch import arch_model

# Suppress convergence warnings to keep the console clean during expanding windows
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def fit_static_models_for_research(returns_series):
    """
    Fits GARCH(1,1), EGARCH, and GJR-GARCH on the entire dataset.
    DO NOT use these outputs for ML features (Data Leakage). 
    Use this strictly for AIC/BIC comparison in your EDA notebook.
    """
    print("Fitting static models for AIC/BIC comparison...")
    returns_scaled = returns_series.dropna() * 100
    results = {}
    
    # Standard GARCH
    garch = arch_model(returns_scaled, vol='GARCH', p=1, q=1, mean='Constant').fit(disp='off')
    results['GARCH(1,1)'] = {'AIC': garch.aic, 'BIC': garch.bic}
    
    # EGARCH (Exponential)
    egarch = arch_model(returns_scaled, vol='EGARCH', p=1, o=1, q=1, mean='Constant').fit(disp='off')
    results['EGARCH(1,1)'] = {'AIC': egarch.aic, 'BIC': egarch.bic}
    
    # GJR-GARCH (Threshold)
    gjr = arch_model(returns_scaled, vol='GARCH', p=1, o=1, q=1, mean='Constant').fit(disp='off')
    results['GJR-GARCH(1,1)'] = {'AIC': gjr.aic, 'BIC': gjr.bic}
    
    comparison_df = pd.DataFrame(results).T.sort_values(by='AIC')
    return comparison_df

def generate_expanding_garch(df, return_col='Log_Return', window_size=252, refit_freq=21):
    """
    INDUSTRY STANDARD ML-SAFE GARCH & EGARCH.
    Balances strict out-of-sample ML safety with computational speed by 
    re-fitting the optimizer periodically (e.g., every 21 days) rather than daily.
    """
    print(f"Generating Industry Standard GARCH & EGARCH (Window: {window_size}, Refit: {refit_freq} days)...")
    df = df.copy()
    
    # Scale returns by 100 for optimizer convergence
    returns = df[return_col] * 100
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
            
            # FIX= Force to numpy array, safely slice, and assign
            chunk_vol_g = np.asarray(fixed_res_g.conditional_volatility)[window_size:]
            vol_garch.iloc[i : chunk_end] = chunk_vol_g
        except Exception:
            pass
            
        # -------------------------------------------------------------
        # B. EGARCH(1,1) - Critical for the Leverage Effect
        # -------------------------------------------------------------
        train_mod_eg = arch_model(train_window, vol='EGARCH', p=1, o=1, q=1, mean='Constant', rescale=False)
        try:
            train_res_eg = train_mod_eg.fit(disp='off', show_warning=False)
            
            eval_mod_eg = arch_model(eval_data, vol='EGARCH', p=1, o=1, q=1, mean='Constant', rescale=False)
            fixed_res_eg = eval_mod_eg.fix(train_res_eg.params)
            
            # FIX= Force to numpy array, safely slice, and assign
            chunk_vol_eg = np.asarray(fixed_res_eg.conditional_volatility)[window_size:]
            vol_egarch.iloc[i : chunk_end] = chunk_vol_eg
        except Exception:
            pass

    # Unscale the final volatility to match original return decimals
    vol_garch_unscaled = vol_garch / 100
    vol_egarch_unscaled = vol_egarch / 100

    # Cap OOS mathematical explosions at a realistic historical maximum (20% daily vol)
    df['Vol_GARCH'] = vol_garch_unscaled.clip(upper=0.20)
    df['Vol_EGARCH'] = vol_egarch_unscaled.clip(upper=0.20)
    
    print("Out-of-sample GARCH/EGARCH generation complete.")
    
    return df.dropna(subset=['Vol_GARCH', 'Vol_EGARCH'])

# src/regimes.py
import pandas as pd
import numpy as np
from hmmlearn import hmm
import warnings

# Suppress the tiny convergence warning from hmmlearn to keep console clean
warnings.filterwarnings("ignore", category=UserWarning, module='hmmlearn')

def generate_smoothed_targets(df, lower_quant=0.86, upper_quant=0.95, window=252, floor=0.0, shock_smooth=3, elevated_smooth=4):
    """
    Independent version of the heuristic 10-day smoothed target generator.
    """
    print("Generating rolling regimes and smoothed targets...")
    df_local = df.copy()
    
    # 1. Target Variable: Tomorrow's Volatility
    df_local['Future_Sq_Return'] = df_local['Sq_Log_Return'].shift(-1)
    
    # 2. Dynamic Thresholds
    df_local['Rolling_Thresh_Low'] = df_local['Sq_Log_Return'].rolling(window=window).quantile(lower_quant)
    df_local['Rolling_Thresh_High'] = df_local['Sq_Log_Return'].rolling(window=window).quantile(upper_quant)
    
    # Drop rows missing threshold baselines or future returns
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

    # 4. Target Smoothing (Your Custom Density Logic)
    is_shock = (df_local['Target_Regime'] == 2).astype(int)
    is_elevated = (df_local['Target_Regime'] == 1).astype(int)

    shock_density = is_shock.rolling(10).sum()
    elevated_density = is_elevated.rolling(10).sum()

    df_local['Target_Smooth_10d'] = 0
    df_local.loc[elevated_density >= elevated_smooth, 'Target_Smooth_10d'] = 1
    df_local.loc[shock_density >= shock_smooth, 'Target_Smooth_10d'] = 2

    # Drop early warmup rows for density calculation
    df_local = df_local.dropna(subset=['Target_Smooth_10d'])
    
    # Return ONLY the final target column mapped to the date index
    return df_local[['Target_Smooth_10d']]


def generate_hmm_targets(df, feature_cols=['Log_Return', 'Vol_EGARCH'], n_components=3):
    """
    Independent statistical HMM target generator with safe type-casting.
    """
    print(f"\n--- HMM Generation Process ---")
    df_local = df.copy()
    
    # 1. Isolate clean, contiguous historical sequence
    hmm_data = df_local.dropna(subset=feature_cols).copy()
    X = hmm_data[feature_cols].values
    
    # 2. Fit Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_components, 
        covariance_type="full", 
        n_iter=1000, 
        random_state=42
    )
    
    model.fit(X)
    hidden_states = model.predict(X)
    
    # Force numpy output into standard python integers
    hmm_data['Temp_State'] = [int(x) for x in hidden_states]
    
    # 3. Dynamic State Sorting (0=Calm, 1=Elevated, 2=Shock)
    state_vols = {}
    for i in range(n_components):
        # Calculate mean vol, default to -1 if state is somehow empty
        mean_vol = hmm_data[hmm_data['Temp_State'] == i]['Vol_EGARCH'].mean()
        state_vols[i] = mean_vol if pd.notna(mean_vol) else -1.0
        print(f"HMM State {i} Average Volatility: {state_vols[i]:.4f}")
        
    # Sort states by their mean volatility
    sorted_states = sorted(list(state_vols.keys()), key=lambda k: state_vols[k])
    print(f"Mapping Order (Lowest Vol to Highest Vol): {sorted_states}")
    
    # Create strictly typed mapping dictionary
    mapping = {int(old_label): int(new_label) for new_label, old_label in enumerate(sorted_states)}
    
    # Apply mapping using a safe lambda function
    hmm_data['Target_HMM'] = hmm_data['Temp_State'].apply(lambda x: mapping.get(x, np.nan))
    
    # 4. Shift Target Backward (Predicting Tomorrow's State)
    hmm_data['Target_HMM'] = hmm_data['Target_HMM'].shift(-1)
    hmm_data = hmm_data.dropna(subset=['Target_HMM'])
    hmm_data['Target_HMM'] = hmm_data['Target_HMM'].astype(int)
    
    print(f"HMM Target generation complete. Shape: {hmm_data[['Target_HMM']].shape}\n")
    
    return hmm_data[['Target_HMM']]

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

# 01_volatility_analysis.ipynb

# 1. Imports and Setup
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_arch
from IPython.display import display

# Ensure the notebook can find the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src import data_loader, features, volatility, config

# 2. Ingest Data
print(f"Loading data for {config.TARGET_ETF}...")
raw_df = data_loader.fetch_raw_data(force_refresh=False)
stat_df = features.engineer_stationary_features(raw_df)

# 3. Prove Volatility Clustering (ARCH-LM Test)
returns = stat_df['Log_Return']
returns_mean_adj = returns - returns.mean()
sq_returns = stat_df['Sq_Log_Return']

lm_stat, p_value, f_stat, fp_value = het_arch(returns_mean_adj, nlags=5)
print(f"ARCH-LM Test p-value: {p_value:.4e} (If < 0.05, GARCH is justified)")

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

# 02_regime_engineering.ipynb

# 1. Imports
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src import data_loader, features, volatility, regimes, config

# 1. Standard Ingest and Base Features
raw_df = data_loader.fetch_raw_data()
stat_df = features.engineer_stationary_features(raw_df)
garch_df = volatility.generate_expanding_garch(stat_df, min_obs=252)

# 2. Generate Targets
smooth_df = regimes.generate_smoothed_targets(garch_df, lower_quant=0.85, upper_quant=0.95, window=252,
                                             floor=0.0, shock_smooth=3, elevated_smooth=3)
hmm_df = regimes.generate_hmm_targets(garch_df, feature_cols=['Log_Return', 'Vol_EGARCH'])

# 3. Robust Concatenation via Inner Join
# We explicitly add garch_df['Log_Return'] here so the plotting code can see it!
comparison_df = pd.concat([garch_df['Log_Return'], smooth_df, hmm_df], axis=1, join='inner')

# 4. Ensure clean integer types for the targets
comparison_df['Target_Smooth_10d'] = comparison_df['Target_Smooth_10d'].astype(int)
comparison_df['Target_HMM'] = comparison_df['Target_HMM'].astype(int)

# 5. Print Final Distributions
print("="*50)
print("Target Class Distributions Aligned Successfully")
print("="*50)
print("\nMethod A: 10-day Smoothed Targets:")
print(comparison_df['Target_Smooth_10d'].value_counts().sort_index())

print("\nMethod B: Statistical HMM Targets:")
print(comparison_df['Target_HMM'].value_counts().sort_index())


def plot_regime_comparison(df, title_suffix="XLK"):
    """
    Generates a stacked comparison chart of heuristic vs. statistical regimes.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Define colors for regimes (1 = Elevated (Orange), 2 = Shock (Red))
    color_map = {1: ('orange', 0.2), 2: ('red', 0.4)}
    
    # ---------------------------------------------------------
    # TOP PLOT: Smoothed Targets
    # ---------------------------------------------------------
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
            
    axes[0].set_title(f"Method A: Manual Rolling Thresholds & Smoothing ({title_suffix})", fontsize=14)
    axes[0].set_ylabel("Log Return")
    axes[0].grid(alpha=0.3)
    
    # ---------------------------------------------------------
    # BOTTOM PLOT: HMM Targets
    # ---------------------------------------------------------
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
            
    axes[1].set_title(f"Method B: Unsupervised Hidden Markov Model ({title_suffix})", fontsize=14)
    axes[1].set_ylabel("Log Return")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Execute the plotting function
plot_regime_comparison(comparison_df)

# 1. Generate the 4-State HMM
# We change n_components to 4
hmm_4_df = regimes.generate_hmm_targets(garch_df, feature_cols=['Log_Return', 'Vol_EGARCH'], n_components=4)

# 2. Add it to our master comparison dataframe
comparison_df = pd.concat([garch_df['Log_Return'], smooth_df, hmm_4_df], axis=1, join='inner')
comparison_df['Target_Smooth_10d'] = comparison_df['Target_Smooth_10d'].astype(int)
comparison_df['Target_HMM'] = comparison_df['Target_HMM'].astype(int)

# 3. Print the new distributions
print("\n" + "="*50)
print("Testing 4-State HMM Distribution")
print("="*50)
print(comparison_df['Target_HMM'].value_counts().sort_index())

# Map the 4-state HMM down to 3 classes for Machine Learning
mapping_dict = {
    0: 0,  # Calm (Lowest vol: 0.82) -> Becomes Class 0
    1: 0,  # Calm (Normal vol: 1.22) -> Becomes Class 0
    2: 1,  # Elevated (High vol: 1.76) -> Becomes Class 1
    3: 2   # Shock (Crash vol: 2.76) -> Becomes Class 2
}

# Apply the mapping
comparison_df['Target_HMM'] = comparison_df['Target_HMM'].map(mapping_dict)

print("="*50)
print("FINAL MACHINE LEARNING TARGET DISTRIBUTION (HMM)")
print("="*50)
print(comparison_df['Target_HMM'].value_counts().sort_index())

plot_regime_comparison(comparison_df)

# 03_baseline_xgboost.ipynb

# 1. Imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src import data_loader, features, volatility, regimes, models_xgb

# 2. The Complete Modular Pipeline Execution
raw_df = data_loader.fetch_raw_data()
stat_df = features.engineer_stationary_features(raw_df)

# Generate ML-Safe GARCH
garch_df = volatility.generate_expanding_garch(stat_df, min_obs=252)

# Generate Targets
regime_df = regimes.generate_smoothed_targets(
    garch_df, lower_quant=0.85, upper_quant=0.95, floor=0.0
)

# Generate Lags for XGBoost
cols_to_lag = [
    'Log_Return', 'Sq_Log_Return', 'Vol_GARCH', 'Vol_EGARCH', 
    'VIX_Change', 'Oil_Change', 'CPI_MoM', 'FedFunds_Diff', 'Term_Spread_Diff'
]
final_df = features.create_lags(regime_df, feature_cols=cols_to_lag)

# 3. Train and Evaluate
model, feature_names, X_test, y_test = models_xgb.train_xgboost_baseline(final_df)

# 4. Feature Importance
models_xgb.plot_feature_importance(model, feature_names)

# In 03_baseline_xgboost.ipynb
from src import data_loader, features, volatility, regimes, models_xgb

# 1. Core Pipeline
raw_df = data_loader.fetch_raw_data()
stat_df = features.engineer_stationary_features(raw_df)
garch_df = volatility.generate_expanding_garch(stat_df, min_obs=252)
regime_df = regimes.generate_smoothed_targets(garch_df, lower_quant=0.85, upper_quant=0.95, floor=0.0)

# 2. Lag Engineering
cols_to_lag = ['Log_Return', 'Sq_Log_Return', 'Vol_GARCH', 'Vol_EGARCH', 
               'VIX_Change', 'Oil_Change', 'CPI_MoM', 'FedFunds_Diff', 'Term_Spread_Diff']
final_df = features.create_lags(regime_df, feature_cols=cols_to_lag)

# 3. Run the Walk-Forward Validation (4 splits = 20% chunks)
final_model, feature_names = models_xgb.train_xgb_walk_forward(final_df, n_splits=4)

# 4. Look at feature importance from the most mature model fold
models_xgb.plot_feature_importance(final_model, feature_names)

# 03_baseline_xgboost.ipynb
from src import data_loader, features, volatility, regimes, models_xgb
import pandas as pd

# 1. Standard Data Ingest & Math
raw_df = data_loader.fetch_raw_data()
stat_df = features.engineer_stationary_features(raw_df)
garch_df = volatility.generate_expanding_garch(stat_df)

# 2. Generate the 4-State HMM
print("\n--- Generating HMM Targets ---")
hmm_4_df = regimes.generate_hmm_targets(garch_df, feature_cols=['Log_Return', 'Vol_EGARCH'], n_components=4)

# 3. Safe Merge & Mapping to 3 ML Classes
master_df = garch_df.copy()
master_df['Target_HMM'] = hmm_4_df['Target_HMM']
master_df = master_df.dropna(subset=['Target_HMM'])

# Collapse the two lowest volatility states into Class 0 (Calm)
mapping_dict = {
    0: 0,  # Dead Calm -> Calm
    1: 0,  # Normal -> Calm
    2: 1,  # Elevated
    3: 2   # Shock
}
master_df['Target_HMM_ML'] = master_df['Target_HMM'].map(mapping_dict).astype(int)
master_df = master_df.drop('Target_HMM', axis=1)

# 4. Lag Engineering for XGBoost
cols_to_lag = [
    'Log_Return', 'Sq_Log_Return', 'Vol_GARCH', 'Vol_EGARCH', 
    'VIX_Change', 'Oil_Change', 'CPI_MoM', 'FedFunds_Diff', 'Term_Spread_Diff'
]
final_df = features.create_lags(master_df, feature_cols=cols_to_lag)

# 5. Walk-Forward Validation
# We pass our newly mapped HMM column as the target
final_model, feature_names = models_xgb.train_xgb_walk_forward(
    df=final_df, 
    target_col='Target_HMM_ML', 
    n_splits=4
)

models_xgb.plot_feature_importance(final_model, feature_names)
