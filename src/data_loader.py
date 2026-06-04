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
