# data_fetcher.py - The "Nuclear" Flattening Fix
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

def safe_download(ticker, retries=MAX_RETRIES):
    """
    Downloads data and aggressively flattens it to ensure
    indicators can ALWAYS see the numbers.
    """
    for attempt in range(retries):
        try:
            _yf_limiter.wait()
            
            # Use Ticker object for cleaner raw data
            dat = yf.Ticker(ticker)
            df = dat.history(period="2y", auto_adjust=True)
            
            if df is None or df.empty:
                continue

            # --- THE FIX: Aggressive Flattening ---
            
            # 1. Reset Index to make Date a column (removes complexity)
            df = df.reset_index()
            
            # 2. Rename columns to standard lower case to find them easily
            df.columns = [c.lower() for c in df.columns]
            
            # 3. Rename back to Capitalized standard for indicators
            # We map whatever we found ('date', 'close', etc) to the Right Names
            rename_map = {
                'date': 'Date', 'open': 'Open', 'high': 'High', 
                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            }
            df = df.rename(columns=rename_map)
            
            # 4. Set Date back as index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df = df.set_index('Date')
            
            # 5. Force Float Conversion (The "Anti-Text" Shield)
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for c in cols:
                if c in df.columns:
                    # Coerce errors to NaN, then fill NaNs with previous value
                    df[c] = pd.to_numeric(df[c], errors='coerce').ffill()

            # 6. Check Length
            if len(df) < 50:
                continue
                
            return df
                
        except Exception as e:
            time.sleep(1)
            
    return None

def fetch_fred_series(series_id, api_key, start_date):
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
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key: 
        spy = safe_download('SPY')
        return {'spy': spy['Close'] if spy is not None else None}

    start = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
    walcl = fetch_fred_series('WALCL', api_key, start)
    tga = fetch_fred_series('WTREGEN', api_key, start)
    rrp = fetch_fred_series('RRPONTSYD', api_key, start)
    
    result = {}
    if walcl is not None and tga is not None and rrp is not None:
        df = pd.concat([walcl, tga, rrp], axis=1).dropna()
        result['net_liq'] = (df['WALCL'] / 1000) - df['WTREGEN'] - df['RRPONTSYD']
    else:
        result['net_liq'] = None

    spy = safe_download('SPY')
    result['spy'] = spy['Close'] if spy is not None else None
    return result

def get_futures():
    return FUTURES_LIST
