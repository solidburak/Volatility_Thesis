# pipeline2.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from arch import arch_model
from dotenv import load_dotenv
import time # Added to handle API rate limits gracefully

def get_fred_series_with_cache(fred_client, series_id, start_date, data_dir):
    """
    Helper function to cache individual FRED series to avoid Rate Limit errors.
    """
    cache_path = os.path.join(data_dir, f"fred_cache_{series_id}.parquet")
    
    # 1. Check if we already have the raw data downloaded
    if os.path.exists(cache_path):
        print(f"  -> Loaded {series_id} from local cache.")
        return pd.read_parquet(cache_path)
    
    # 2. If not, fetch from API with a polite delay
    print(f"  -> Fetching {series_id} from FRED API...")
    try:
        # Adding a 1-second delay to respect rate limits
        time.sleep(1) 
        series = fred_client.get_series(series_id, observation_start=start_date)
        df = pd.DataFrame(series, columns=[series_id])
        
        # Save to local cache immediately
        df.to_parquet(cache_path)
        return df
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {series_id}: {e}")
        return pd.DataFrame() # Return empty df so the pipeline doesn't crash entirely

def fetch_and_prepare_data(force_refresh=False):
    """
    Downloads XLK and Macro data, fits GARCH models, aligns daily/monthly frequencies,
    adds the Yield Curve Term Spread, and aggressively drops non-stationary columns.
    """
    load_dotenv()
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY not found.")
        
    fred = Fred(api_key=FRED_API_KEY)
    DATA_DIR = 'data'
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_file = os.path.join(DATA_DIR, "processed_master_data.parquet")
    
    # If force_refresh is True, we only delete the FINAL master cache. 
    # We keep the intermediate FRED caches unless you manually delete them.
    if os.path.exists(cache_file) and not force_refresh:
        print(f"Loading cached processed master data from {cache_file}...")
        df = pd.read_parquet(cache_file)
        df.index = pd.to_datetime(df.index)
        return df
        
    print("Starting data pipeline...")
    START_DATE = '2000-01-01'
    
    # 1. Download XLK
    print("Fetching XLK from yfinance...")
    xlk_df = yf.download('XLK', start=START_DATE, progress=False)
    if isinstance(xlk_df.columns, pd.MultiIndex):
        xlk_df.columns = xlk_df.columns.get_level_values(0)
        
    xlk_df['Log_Return'] = np.log(xlk_df['Close'] / xlk_df['Close'].shift(1))
    xlk_df = xlk_df.dropna().copy()
    xlk_df['Sq_Log_Return'] = xlk_df['Log_Return'] ** 2
    
    # 2. Volatility Models
    print("Fitting GARCH/EGARCH models (this takes a moment)...")
    returns_scaled = xlk_df['Log_Return'] * 100
    
    garch_mod = arch_model(returns_scaled, vol='GARCH', p=1, q=1, mean='Constant')
    xlk_df['Vol_GARCH'] = garch_mod.fit(disp='off').conditional_volatility
    
    egarch_mod = arch_model(returns_scaled, vol='EGARCH', p=1, o=1, q=1, mean='Constant')
    xlk_df['Vol_EGARCH'] = egarch_mod.fit(disp='off').conditional_volatility
    
    # 3. Daily Macro (VIX, Oil)
    print("Fetching Daily Macro (VIX, Oil)...")
    macro_tickers = {'VIX': '^VIX', 'Oil_WTI': 'CL=F'}
    daily_macro = yf.download(list(macro_tickers.values()), start=START_DATE, progress=False)['Close']
    daily_macro.columns = list(macro_tickers.keys())
    
    daily_macro['VIX_Change'] = daily_macro['VIX'].pct_change()
    daily_macro['Oil_Change'] = daily_macro['Oil_WTI'].pct_change()
    
    # 4. Monthly Macro & Yield Curve (FRED)
    print("Fetching FRED Macro Data (using intermediate cache)...")
    fred_series_map = {
        'CPI': 'CPIAUCSL', 
        'FedFunds': 'FEDFUNDS', 
        'Unemployment': 'UNRATE',
        'Term_Spread': 'T10Y2Y' 
    }
    
    fred_data = {}
    for name, series_id in fred_series_map.items():
        # Using our new caching helper function
        df_raw = get_fred_series_with_cache(fred, series_id, START_DATE, DATA_DIR)
        
        if df_raw.empty:
            continue
            
        # Rename the column from the raw ID (e.g., CPIAUCSL) to our clean name (e.g., CPI)
        df_macro = df_raw.rename(columns={series_id: name})
        
        # Apply the stationary math transformations
        if name == 'CPI':
            df_macro['CPI_MoM'] = df_macro[name].pct_change() * 100
        elif name == 'Term_Spread':
            df_macro['Term_Spread_Diff'] = df_macro[name].diff()
        else:
            df_macro[f'{name}_Diff'] = df_macro[name].diff()
            
        fred_data[name] = df_macro
        
    monthly_macro = pd.concat(fred_data.values(), axis=1)
    
    # 5. Merge datasets
    print("Aligning and merging all features...")
    master_df = xlk_df.join(daily_macro, how='left')
    master_df = master_df.join(monthly_macro, how='left').ffill()
    master_df = master_df.dropna()
    
    # 6. THE PURGE: Drop non-stationary and raw columns to prevent model cheating
    columns_to_drop = [
        'Open', 'High', 'Low', 'Close', 'Volume',  
        'VIX', 'Oil_WTI',                          
        'CPI', 'FedFunds', 'Unemployment', 'Term_Spread' 
    ]
    
    cols_to_drop_safe = [col for col in columns_to_drop if col in master_df.columns]
    master_df = master_df.drop(columns=cols_to_drop_safe)
    
    # Save the final processed dataset
    master_df.to_parquet(cache_file)
    print(f"Pipeline complete. Stationary master dataset saved to {cache_file}")
    
    return master_df

def create_multiclass_target(df, lower_quant=0.60, upper_quant=0.88, window=252, floor=0.0005, shock_smooth=3, elevated_smooth=4):
    """
    Applies a rolling historical window to define dynamic regime thresholds.
    """
    df = df.copy()
    
    df['Future_Sq_Return'] = df['Sq_Log_Return'].shift(-1)
    
    df['Rolling_Thresh_Low'] = df['Sq_Log_Return'].rolling(window=window).quantile(lower_quant)
    df['Rolling_Thresh_High'] = df['Sq_Log_Return'].rolling(window=window).quantile(upper_quant)
    
    df = df.dropna()
    
    import numpy as np
    conditions = [
        (df['Future_Sq_Return'] <= df['Rolling_Thresh_Low']) | (df["Future_Sq_Return"] <= floor) ,
        (df['Future_Sq_Return'] > df['Rolling_Thresh_Low']) & (df['Future_Sq_Return'] <= df['Rolling_Thresh_High']),
        df['Future_Sq_Return'] > df['Rolling_Thresh_High']
    ]
    
    choices = [0, 1, 2] 
    df['Target_Regime'] = np.select(conditions, choices, default=np.nan)
    df['Target_Regime'] = df['Target_Regime'].astype(int)

    is_shock = (df['Target_Regime'] == 2).astype(int)
    is_elevated = (df['Target_Regime'] == 1).astype(int)

    shock_density = is_shock.rolling(10).sum()
    elevated_density = is_elevated.rolling(10).sum()

    df['Target_Smooth_10d'] = 0
    df.loc[elevated_density >= elevated_smooth, 'Target_Smooth_10d'] = 1
    df.loc[shock_density >= shock_smooth, 'Target_Smooth_10d'] = 2

    df = df.dropna(subset=['Target_Smooth_10d'])
    
    return df
