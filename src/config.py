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
