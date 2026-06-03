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
