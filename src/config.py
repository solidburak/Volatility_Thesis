import os
import logging
from dotenv import load_dotenv
from typing import Dict, List

# --- Logging Configuration ---
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

# MULTI-SECTOR PREPARATION: List of targets instead of a single string.
# We keep it to 1 for now per instructions, but the pipeline handles it as a list.
TARGET_ETFS: List[str] = ['XLK'] 

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
MACRO_LAGS: Dict[str, int] = {
    'CPI': 11,           
    'FedFunds': 1,       
    'Unemployment': 5,   
    'Term_Spread': 1     
}
