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
TARGET_ETFS: List[str] = ['XLK', 'XLE', 'XLF', 'XLV'] 

# --- Tickers & Macro Data ---
DAILY_MACRO: Dict[str, str] = {
    'VIX': '^VIX', 
    'Oil_WTI': 'CL=F'
}

MONTHLY_MACRO: Dict[str, str] = {
    # Baseline
    'CPI': 'CPIAUCSL', 
    'FedFunds': 'FEDFUNDS', 
    'Unemployment': 'UNRATE',
    'Term_Spread': 'T10Y2Y',
    
    # Flannery & Protopapadakis (2002) Additions
    'PPI': 'WPSFD49207',       # Producer Price Index: Finished Goods
    'Housing_Starts': 'HOUST', # New Privately-Owned Housing Units Started
    'Trade_Balance': 'BOPGSTB',# Trade Balance: Goods and Services
    'M1_Money': 'M1SL',         # M1 Real Money Stock}

    # --- YENİ EKLENENLER ---
    'NFP': 'PAYEMS',           # Non-Farm Payrolls (Tarım Dışı İstihdam)
    'Retail_Sales': 'RSAFS'    # Advance Retail Sales
}

# --- Macro Reporting Lags (in trading days) ---
MACRO_LAGS: Dict[str, int] = {
    # Baseline Lags
    'CPI': 11,           # ~11 trading days into the following month
    'FedFunds': 1,       # Known immediately, 1 day lag to be safe
    'Unemployment': 5,   # First Friday of the following month (~5 trading days)
    'Term_Spread': 1,    # Daily series, 1 day lag is safe
    
    # Flannery 2002 Addition Lags
    'PPI': 10,             # Usually released 1 day before CPI
    'Housing_Starts': 14,  # Usually released around the 18th-20th of the following month
    'Trade_Balance': 25,   # Usually released in the first week of the *second* month following
    'M1_Money': 20,         # Usually released late in the following month

    # --- YENİ EKLENENLERİN GECİKMELERİ ---
    'NFP': 5,            # Her ayın ilk Cuması (Yaklaşık 5 iş günü)
    'Retail_Sales': 10   # Ayın ortasında açıklanır (Yaklaşık 10 iş günü)
}
