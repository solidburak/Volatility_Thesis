import os
import time
import logging
import pandas as pd
import yfinance as yf
from fredapi import Fred
from src import config
import pickle
from typing import Dict, List

logger = logging.getLogger(__name__)

def get_fred_series_with_cache(fred_client: Fred, series_id: str, series_name: str, start_date: str) -> pd.DataFrame:
    cache_path = os.path.join(config.RAW_DATA_DIR, f"fred_cache_{series_id}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    
    logger.info(f"Fetching {series_name} ({series_id}) from FRED API...")
    try:
        time.sleep(1.5) 
        series = fred_client.get_series(series_id, observation_start=start_date)
        df = series.to_frame(name=series_name)
        df.to_parquet(cache_path)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch {series_id}: {str(e)}")
        return pd.DataFrame()

def fetch_raw_data(force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Refactored to return a dictionary of DataFrames, one for each Target ETF.
    This makes the jump to Multi-Sector VAR modeling seamless later.
    """
    cache_path = os.path.join(config.RAW_DATA_DIR, "sector_datasets_cache.pkl")
    
    # 1. Check Cache
    if not force_refresh and os.path.exists(cache_path):
        logger.info("Loading cached multi-sector data...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    fred = Fred(api_key=config.FRED_API_KEY)
    
# 1. Download Macro Data (Only needs to happen once)
    daily_tickers = list(config.DAILY_MACRO.values())
    daily_macro_raw = yf.download(daily_tickers, start=config.START_DATE, progress=False)
    
    if isinstance(daily_macro_raw.columns, pd.MultiIndex):
        daily_macro = daily_macro_raw['Close'].copy()
    else:
        daily_macro = daily_macro_raw.copy()
        
    ticker_to_name = {v: k for k, v in config.DAILY_MACRO.items()}
    daily_macro.rename(columns=ticker_to_name, inplace=True)
    
    fred_data = {}
    for name, series_id in config.MONTHLY_MACRO.items():
        df_raw = get_fred_series_with_cache(fred, series_id, name, config.START_DATE)
        if not df_raw.empty:
            fred_data[name] = df_raw
    monthly_macro = pd.concat(fred_data.values(), axis=1)
    
    # 2. Process each ETF independently
    sector_datasets = {}
    for etf in config.TARGET_ETFS:
        logger.info(f"Processing data for {etf}...")
        etf_df = yf.download(etf, start=config.START_DATE, progress=False)
        
        if isinstance(etf_df.columns, pd.MultiIndex):
            etf_df.columns = etf_df.columns.get_level_values(0)
            
        base_df = pd.DataFrame({f'{etf}_Close': etf_df['Close']})
        
        # Merge Macro to Sector
        master_df = base_df.join(daily_macro, how='left')
        master_df = master_df.join(monthly_macro, how='left')
        master_df = master_df.ffill().dropna()
        
        sector_datasets[etf] = master_df

    # 3. Save Cache before returning
    with open(cache_path, 'wb') as f:
        pickle.dump(sector_datasets, f)
        
    return sector_datasets
