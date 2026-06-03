# pipeline.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from arch import arch_model
from dotenv import load_dotenv

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
    
    if os.path.exists(cache_file) and not force_refresh:
        print(f"Loading cached processed master data from {cache_file}...")
        df = pd.read_parquet(cache_file)
        df.index = pd.to_datetime(df.index)
        return df
        
    print("Starting data pipeline...")
    
    # 1. Download XLK
    xlk_df = yf.download('XLK', start='2000-01-01', progress=False)
    if isinstance(xlk_df.columns, pd.MultiIndex):
        xlk_df.columns = xlk_df.columns.get_level_values(0)
        
    xlk_df['Log_Return'] = np.log(xlk_df['Close'] / xlk_df['Close'].shift(1))
    xlk_df = xlk_df.dropna().copy()
    xlk_df['Sq_Log_Return'] = xlk_df['Log_Return'] ** 2
    
    # 2. Volatility Models
    returns_scaled = xlk_df['Log_Return'] * 100
    
    garch_mod = arch_model(returns_scaled, vol='GARCH', p=1, q=1, mean='Constant')
    xlk_df['Vol_GARCH'] = garch_mod.fit(disp='off').conditional_volatility
    
    egarch_mod = arch_model(returns_scaled, vol='EGARCH', p=1, o=1, q=1, mean='Constant')
    xlk_df['Vol_EGARCH'] = egarch_mod.fit(disp='off').conditional_volatility
    
    # 3. Daily Macro (VIX, Oil)
    macro_tickers = {'VIX': '^VIX', 'Oil_WTI': 'CL=F'}
    daily_macro = yf.download(list(macro_tickers.values()), start='2000-01-01', progress=False)['Close']
    daily_macro.columns = list(macro_tickers.keys())
    
    daily_macro['VIX_Change'] = daily_macro['VIX'].pct_change()
    daily_macro['Oil_Change'] = daily_macro['Oil_WTI'].pct_change()
    
    # 4. Monthly Macro & Yield Curve (FRED)
    # T10Y2Y is the 10-Year minus 2-Year Treasury Constant Maturity (Term Spread)
    fred_series = {
        'CPI': 'CPIAUCSL', 
        'FedFunds': 'FEDFUNDS', 
        'Unemployment': 'UNRATE',
        'Term_Spread': 'T10Y2Y' 
    }
    
    fred_data = {}
    for name, series_id in fred_series.items():
        series = fred.get_series(series_id, observation_start='2000-01-01')
        df_macro = pd.DataFrame(series, columns=[name])
        
        if name == 'CPI':
            df_macro['CPI_MoM'] = df_macro[name].pct_change() * 100
        elif name == 'Term_Spread':
            # Term spread is already a percentage yield difference, but we want the day-to-day change
            df_macro['Term_Spread_Diff'] = df_macro[name].diff()
        else:
            df_macro[f'{name}_Diff'] = df_macro[name].diff()
            
        fred_data[name] = df_macro
        
    monthly_macro = pd.concat(fred_data.values(), axis=1)
    
    # 5. Merge datasets
    master_df = xlk_df.join(daily_macro, how='left')
    master_df = master_df.join(monthly_macro, how='left').ffill()
    master_df = master_df.dropna()
    
    # 6. THE PURGE: Drop non-stationary and raw columns to prevent model cheating
    columns_to_drop = [
        'Open', 'High', 'Low', 'Close', 'Volume',  # Raw ETF price data
        'VIX', 'Oil_WTI',                          # Raw daily macro levels
        'CPI', 'FedFunds', 'Unemployment', 'Term_Spread' # Raw FRED levels
    ]
    
    # Only drop columns that actually exist in the dataframe to avoid KeyErrors
    cols_to_drop_safe = [col for col in columns_to_drop if col in master_df.columns]
    master_df = master_df.drop(columns=cols_to_drop_safe)
    
    master_df.to_parquet(cache_file)
    print(f"Pipeline complete. Stationary master dataset saved to {cache_file}")
    return master_df

def create_multiclass_target(df, lower_quant=0.60, upper_quant=0.88, window=252, floor=0.0005, shock_smooth=3, elevated_smooth=4):
    """
    Applies a rolling historical window to define dynamic regime thresholds.
    Eliminates look-ahead bias by comparing tomorrow's realized variance
    only against the distribution of the preceding 'window' days.
    """
    df = df.copy()
    
    # 1. Target Variable: Tomorrow's Volatility
    df['Future_Sq_Return'] = df['Sq_Log_Return'].shift(-1)
    
    # 2. Dynamic Thresholds: Based on TODAY'S 252-day lookback of realized volatility
    # We do not use Future_Sq_Return here to prevent data leakage!
    df['Rolling_Thresh_Low'] = df['Sq_Log_Return'].rolling(window=window).quantile(lower_quant)
    df['Rolling_Thresh_High'] = df['Sq_Log_Return'].rolling(window=window).quantile(upper_quant)
    
    # Drop rows with NaNs (This will naturally drop the first 252 days used for the initial warmup, 
    # as well as the final day which has no 'Future' return)
    df = df.dropna()
    
    # 3. Vectorized Regime Classification (Much faster than .apply for dynamic rows)
    import numpy as np
    conditions = [
        (df['Future_Sq_Return'] <= df['Rolling_Thresh_Low']) | (df["Future_Sq_Return"] <= floor) ,
        (df['Future_Sq_Return'] > df['Rolling_Thresh_Low']) & (df['Future_Sq_Return'] <= df['Rolling_Thresh_High']),
        df['Future_Sq_Return'] > df['Rolling_Thresh_High']
    ]
    
    # 0 = Calm, 1 = Elevated, 2 = Shock
    choices = [0, 1, 2] 
    
    df['Target_Regime'] = np.select(conditions, choices, default=np.nan)
    
    # Convert to integer (np.select sometimes casts as float)
    df['Target_Regime'] = df['Target_Regime'].astype(int)

    # Target smoothing
    is_shock = (df['Target_Regime'] == 2).astype(int)
    is_elevated = (df['Target_Regime'] == 1).astype(int)

    shock_density = is_shock.rolling(10).sum()
    elevated_density = is_elevated.rolling(10).sum()

    shock_threshold = shock_smooth
    elevated_threshold = elevated_smooth

    df['Target_Smooth_10d'] = 0
    df.loc[elevated_density >= elevated_threshold, 'Target_Smooth_10d'] = 1
    df.loc[shock_density >= shock_threshold, 'Target_Smooth_10d'] = 2

    df = df.dropna(subset='Target_Smooth_10d')
    
    return df

def create_static_multiclass_target(df, lower_quant, upper_quant):
    """
    Applies user-defined quantile thresholds to future squared returns 
    to create the Target_Regime column.
    """
    df = df.copy()
    df['Future_Sq_Return'] = df['Sq_Log_Return'].shift(-1)
    df = df.dropna()
    
    thresh_low = df['Future_Sq_Return'].quantile(lower_quant)
    thresh_high = df['Future_Sq_Return'].quantile(upper_quant)
    
    def classify(val):
        if val <= thresh_low:
            return 0  # Calm
        elif val <= thresh_high:
            return 1  # Elevated
        else:
            return 2  # Shock
            
    df['Target_Regime'] = df['Future_Sq_Return'].apply(classify)
    return df, thresh_low, thresh_high
