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
