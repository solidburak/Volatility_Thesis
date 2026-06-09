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

def fetch_raw_data(force_refresh: bool = False) -> dict:
    """
    Fetches multi-sector data with smart cache invalidation.
    Validates current config against cached config snapshot.
    """
    cache_path = os.path.join(config.RAW_DATA_DIR, "sector_datasets_cache.pkl")
    
    # Define the exact state of the config that matters for data ingestion
    current_config_state = {
        'START_DATE': config.START_DATE,
        'TARGET_ETFS': sorted(config.TARGET_ETFS),
        'DAILY_MACRO': sorted(list(config.DAILY_MACRO.keys())),
        'MONTHLY_MACRO': sorted(list(config.MONTHLY_MACRO.keys()))
    }

    # 1. Check Cache and Validate Config Snapshot
    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_payload = pickle.load(f)
                
            # Check if it's our new payload format with metadata
            if isinstance(cached_payload, dict) and 'metadata' in cached_payload:
                if cached_payload['metadata'] == current_config_state:
                    logger.info("Configuration matches cache. Loading data from disk...")
                    return cached_payload['datasets']
                else:
                    logger.info("Config parameters changed. Cache invalidated. Forcing refresh...")
            else:
                logger.info("Legacy cache format detected. Forcing refresh...")
        except Exception as e:
            logger.warning(f"Failed to read cache ({e}). Forcing refresh...")

    logger.info("Fetching fresh multi-sector data from APIs...")
    
    fred = Fred(api_key=config.FRED_API_KEY)
    
    # --- 1. Download Macro Data ---
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
    
    # --- 2. Process each ETF independently ---
    sector_datasets = {}
    for etf in config.TARGET_ETFS:
        logger.info(f"Downloading historical data for {etf}...")
        etf_df = yf.download(etf, start=config.START_DATE, progress=False)
        
        if isinstance(etf_df.columns, pd.MultiIndex):
            etf_df.columns = etf_df.columns.get_level_values(0)
            
        base_df = pd.DataFrame({f'{etf}_Close': etf_df['Close']})
        
        master_df = base_df.join(daily_macro, how='left')
        master_df = master_df.join(monthly_macro, how='left')
        master_df = master_df.ffill().dropna()
        
        sector_datasets[etf] = master_df

    # --- 3. Save Cache with Config Snapshot ---
    cache_payload = {
        'metadata': current_config_state,
        'datasets': sector_datasets
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_payload, f)
        
    return sector_datasets
