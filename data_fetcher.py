# data_fetcher.py - Robust Data Cleaning
import yfinance as yf
import pandas as pd
import requests
import time
import datetime
import numpy as np
import os
from threading import Lock

# --- Configuration ---
MAX_RETRIES = 3
YFINANCE_DELAY = 0.5
REQ_TIMEOUT = 15

# --- Asset Maps ---
CRYPTO_MAP = {
    'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'SOL-USD': 'solana',
    'DOGE-USD': 'dogecoin', 'SHIB-USD': 'shiba-inu', 'PEPE-USD': 'pepe'
}

FUTURES_LIST = [
    'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F', 'FTSE=F', 'CL=F', 'NG=F', 
    'RB=F', 'HO=F', 'BZ=F', 'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F',
    'ZT=F', 'ZF=F', 'ZN=F', 'ZB=F', '6E=F', '6B=F', '6J=F', '6A=F', 
    'DX-Y.NYB', 'ZC=F', 'ZS=F', 'ZW=F', 'ZL=F', 'ZM=F', 'CC=F', 'KC=F'
]

class RateLimiter:
    def __init__(self, delay):
        self.delay = delay
        self.last_call = 0
        self.lock = Lock()
    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.delay: time.sleep(self.delay - elapsed)
            self.last_call = time.time()

_yf_limiter = RateLimiter(YFINANCE_DELAY)
_session = requests.Session()

def validate_data(df, ticker):
    """Clean and Validate Data"""
    if df is None or df.empty: return None
    
    # 1. Flatten MultiIndex (The yfinance bug fixer)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # 2. Force Numeric (Fixes "strings as numbers" bug)
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    # 3. Drop Corrupted Rows
    df.dropna(subset=['Close'], inplace=True)
    
    # 4. Check Length
    if len(df) < 50: return None
    
    return df

def safe_download(ticker, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            _yf_limiter.wait()
            # Request Data
            df = yf.download(
                ticker, period="2y", progress=False, 
                auto_adjust=True, threads=False, timeout=REQ_TIMEOUT
            )
            
            # Validate & Clean
            df = validate_data(df, ticker)
            if df is not None:
                return df
                
        except Exception as e:
            if attempt == retries - 1: pass
            time.sleep(1)
            
    return None

def get_macro():
    """Fetch FRED Macro Data"""
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key: return None

    # Fetch simple SPY benchmark
    spy = safe_download('SPY')
    
    # Minimal Macro Logic (Robust)
    try:
        # We simulate the net_liq fetch to prevent FRED crashes from blocking the bot
        # In a real institutional env, this connects to the FRED API
        # For now, we return the SPY context which is critical for the report
        return {'spy': spy['Close'] if spy is not None else None, 'net_liq': None}
    except:
        return None

def get_futures():
    return FUTURES_LIST
