# data_fetcher.py - The "Golden" Version (Real Data + Robust Cleaning)
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
    """Clean and Validate Data - FIXES THE ZEROS"""
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
    
    # 4. Check Length (Need enough data for ADX/DeMark)
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

def fetch_fred_series(series_id, api_key, start_date):
    """Helper to fetch FRED data"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id, 'api_key': api_key, 'file_type': 'json',
        'observation_start': start_date
    }
    try:
        r = _session.get(url, params=params, timeout=REQ_TIMEOUT)
        data = r.json()
        if 'observations' not in data: return None
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df['value'].replace('.', np.nan).astype(float).rename(series_id)
    except:
        return None

def get_macro():
    """Fetch Real FRED Macro Data"""
    api_key = os.environ.get('FRED_API_KEY')
    # If no key, return partial data so bot doesn't crash
    if not api_key: 
        spy = safe_download('SPY')
        return {'spy': spy['Close'] if spy is not None else None}

    start = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Fetch core liquidity components
    walcl = fetch_fred_series('WALCL', api_key, start)      # Fed Assets
    tga = fetch_fred_series('WTREGEN', api_key, start)      # TGA
    rrp = fetch_fred_series('RRPONTSYD', api_key, start)    # Reverse Repo
    
    result = {}
    
    # Calculate Net Liquidity if possible
    if walcl is not None and tga is not None and rrp is not None:
        # Align dates
        df = pd.concat([walcl, tga, rrp], axis=1).dropna()
        # WALCL is millions, others billions. Convert WALCL to billions.
        result['net_liq'] = (df['WALCL'] / 1000) - df['WTREGEN'] - df['RRPONTSYD']
    else:
        result['net_liq'] = None

    # Fetch SPY for context
    spy = safe_download('SPY')
    result['spy'] = spy['Close'] if spy is not None else None
    
    return result

def get_futures():
    return FUTURES_LIST
