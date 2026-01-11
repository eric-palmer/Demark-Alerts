# data_fetcher.py - Institutional Grade Data Ingestion
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
YFINANCE_DELAY = 0.5  # Seconds between calls to avoid blacklisting
COINGECKO_DELAY = 1.5 # Strict rate limit for free tier
REQ_TIMEOUT = 15      # Seconds before timing out a request

# --- Asset Maps ---
CRYPTO_MAP = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana',
    'DOGE-USD': 'dogecoin',
    'SHIB-USD': 'shiba-inu',
    'PEPE-USD': 'pepe'
}

FUTURES_LIST = [
    'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F', 'FTSE=F',
    'CL=F', 'NG=F', 'RB=F', 'HO=F', 'BZ=F',
    'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F',
    'ZT=F', 'ZF=F', 'ZN=F', 'ZB=F',
    '6E=F', '6B=F', '6J=F', '6A=F', 'DX-Y.NYB',
    'ZC=F', 'ZS=F', 'ZW=F', 'ZL=F', 'ZM=F',
    'CC=F', 'KC=F', 'SB=F', 'CT=F', 'LE=F', 'HE=F'
]

# --- Infrastructure ---
class RateLimiter:
    """Thread-safe rate limiter to protect against API bans"""
    def __init__(self, delay):
        self.delay = delay
        self.last_call = 0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_call = time.time()

_yf_limiter = RateLimiter(YFINANCE_DELAY)
_cg_limiter = RateLimiter(COINGECKO_DELAY)
_session = requests.Session()

def get_session():
    return _session

# --- Core Functions ---

def validate_data(df, ticker):
    """Institutional data integrity checks"""
    if df is None or df.empty:
        return False
    
    # Check 1: Minimum history for indicators (200 SMA requires ~200 bars)
    if len(df) < 50: 
        return False
        
    # Check 2: Freshness (Data must be recent)
    last_date = df.index[-1]
    # Allow for weekends/holidays (4 days max lag)
    if (datetime.datetime.now() - last_date).days > 5:
        # Special exception for some futures/indexes that might delay
        if ticker not in FUTURES_LIST: 
            return False

    # Check 3: Data gaps (Nulls in Close)
    if df['Close'].isnull().sum() > len(df) * 0.05:
        return False

    return True

def safe_download(ticker, retries=MAX_RETRIES):
    """Robust download with YFinance priority and Crypto fallback"""
    
    # 1. Try YFinance first (Preferred source)
    for attempt in range(retries):
        try:
            _yf_limiter.wait()
            
            # exponential backoff on retries
            if attempt > 0:
                time.sleep(2 * attempt)
                print(f"   Retry {attempt}/{retries} for {ticker}...")

            # Download without threading to prevent race conditions
            df = yf.download(
                ticker, 
                period="2y", 
                progress=False, 
                auto_adjust=True, 
                threads=False, 
                timeout=REQ_TIMEOUT
            )

            # Clean MultiIndex if present (common yfinance issue)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if validate_data(df, ticker):
                return df
                
        except Exception as e:
            if attempt == retries - 1:
                # Don't print error on first fail, only on final fail
                pass

    # 2. Fallback: CoinGecko (Only for known cryptos)
    if ticker in CRYPTO_MAP:
        return _fetch_coingecko(ticker)
        
    return None

def _fetch_coingecko(ticker):
    """Backup crypto data source"""
    coin_id = CRYPTO_MAP.get(ticker)
    if not coin_id:
        return None

    try:
        _cg_limiter.wait()
        
        # Get OHLC (Open, High, Low, Close)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=730"
        r = _session.get(url, timeout=REQ_TIMEOUT)
        if r.status_code != 200:
            return None
            
        data = r.json()
        if not data:
            return None

        df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # CoinGecko OHLC doesn't include volume, fetch separately or assume 0
        # For simplicity in fallback, we assume 0 or average to prevent indicator crash
        df['Volume'] = 0 
        
        return df if validate_data(df, ticker) else None
        
    except Exception:
        return None

def get_futures():
    """Returns the static list of futures tickers"""
    return FUTURES_LIST

# --- Macro Economics (FRED) ---

def fetch_fred_series(series_id, api_key, start_date):
    """Helper to fetch a single FRED series"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date,
        'observation_end': datetime.datetime.now().strftime('%Y-%m-%d')
    }
    
    try:
        r = _session.get(url, params=params, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        
        if 'observations' not in data:
            return None

        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Clean data (replace '.' with NaN and convert to float)
        s = df['value'].replace('.', np.nan).astype(float)
        return s.rename(series_id)
        
    except Exception as e:
        print(f"   FRED Error ({series_id}): {e}")
        return None

def get_macro():
    """
    Fetches macro liquidity and regime data.
    Returns a dictionary of metrics or None if critical data fails.
    """
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        print("   ⚠️ FRED API Key missing")
        return None

    print("   Fetching Macro Data...")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Critical Series for Net Liquidity
    # WALCL: Fed Total Assets
    # WTREGEN: Treasury General Account
    # RRPONTSYD: Reverse Repo
    series_map = {
        'WALCL': 'FedAssets',
        'WTREGEN': 'TGA',
        'RRPONTSYD': 'RRP',
        'DGS10': '10Y',
        'DGS2': '2Y',
        'T5YIE': 'InflationExpectations'
    }

    data_frames = []
    for sid, name in series_map.items():
        s = fetch_fred_series(sid, api_key, start_date)
        if s is not None:
            data_frames.append(s)
            
    if not data_frames:
        return None

    # Merge all series on Date index
    macro_df = pd.concat(data_frames, axis=1)
    macro_df = macro_df.resample('D').ffill().dropna() # Forward fill weekends
    
    result = {}
    
    # 1. Calculate Net Liquidity (The "Howell" Metric)
    # Net Liq = Fed Assets - TGA - RRP
    # Note: WALCL is in millions, others usually billions. 
    # FRED units: WALCL (Millions), WTREGEN (Billions), RRP (Billions)
    try:
        if 'WALCL' in macro_df.columns and 'WTREGEN' in macro_df.columns and 'RRPONTSYD' in macro_df.columns:
            # Convert WALCL to Billions to match others
            fed_assets_bn = macro_df['WALCL'] / 1000 
            net_liq = fed_assets_bn - macro_df['WTREGEN'] - macro_df['RRPONTSYD']
            result['net_liq'] = net_liq
            result['fed_assets'] = fed_assets_bn
    except Exception as e:
        print(f"   Calc Error (Net Liq): {e}")

    # 2. Yield Curve
    try:
        if 'DGS10' in macro_df.columns and 'DGS2' in macro_df.columns:
            result['term_premia'] = macro_df['DGS10'] - macro_df['DGS2']
    except:
        pass

    # 3. Inflation
    if 'T5YIE' in macro_df.columns:
        result['inflation'] = macro_df['T5YIE']

    # 4. SPY Benchmark (for relative strength)
    spy = safe_download('SPY')
    if spy is not None:
        result['spy'] = spy['Close']

    print(f"   ✓ Macro Data Loaded ({len(result)} metrics)")
    return result
