# data_fetcher.py - Institutional Router (Full OHLC Fix)
import pandas as pd
import time
import os
import yfinance as yf
from tiingo import TiingoClient

# --- Configuration ---
PAUSE_SEC = 0.2

def get_tiingo_client():
    """Create a fresh client every time to prevent timeouts"""
    api_key = os.environ.get('TIINGO_API_KEY')
    return TiingoClient({'api_key': api_key, 'session': True}) if api_key else None

def fetch_tiingo(ticker, client):
    """Institutional Source: Tiingo (Full OHLC)"""
    try:
        # 1. Crypto Handling (btcusd)
        if '-USD' in ticker:
            sym = ticker.replace('-USD', '').lower() + 'usd'
            data = client.get_crypto_price_history(tickers=[sym], startDate='2022-01-01', resampleFreq='1day')
            df = pd.DataFrame(data[0].get('priceData', []))
            rename = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        
        # 2. Stock Handling
        else:
            # FIX: Request FULL data (no metric_name limit)
            df = client.get_dataframe(ticker, startDate='2022-01-01')
            
            # Map Adjusted columns to Standard names
            rename = {
                'adjOpen': 'Open', 'adjHigh': 'High', 'adjLow': 'Low', 
                'adjClose': 'Close', 'adjVolume': 'Volume',
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            }
        
        df = df.rename(columns=rename)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Validation: Must have High/Low for ADX/DeMark
        if 'High' not in df.columns: return None
        
        return df.set_index('Date')
    except Exception:
        return None

def fetch_fallback(ticker):
    """Backup Source: Yahoo (Nuclear Flat)"""
    try:
        dat = yf.Ticker(ticker)
        df = dat.history(period="2y", auto_adjust=True)
        if df.empty: return None
        
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        rename = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        df = df.rename(columns=rename)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df.set_index('Date')
            
        # Force numeric
        for c in ['Open', 'High', 'Low', 'Close']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except: return None

def safe_download(ticker, client=None):
    """
    Smart Router: 
    1. Futures (=F) -> Yahoo (Tiingo doesn't have them)
    2. Stocks/Crypto -> Tiingo (Best Quality)
    3. Fallback -> Yahoo
    """
    # Futures Check
    if any(x in ticker for x in ['=F', 'DX-Y', '=X']):
        return fetch_fallback(ticker)

    df = None
    
    # Priority: Tiingo
    if client:
        time.sleep(PAUSE_SEC)
        df = fetch_tiingo(ticker, client)
    
    # Fallback: Yahoo
    if df is None or len(df) < 5:
        df = fetch_fallback(ticker)
        
    # Validation
    if df is not None:
        df = df.ffill().dropna()
        if len(df) > 30:
            return df
            
    return None

def get_macro():
    """Context Fetcher"""
    spy = fetch_fallback('SPY')
    api_key = os.environ.get('FRED_API_KEY')
    result = {'net_liq': None, 'spy': None}
    
    if spy is not None: result['spy'] = spy['Close']
    
    if api_key:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            import requests 
            def get(sid):
                r = requests.get(url, params={'series_id': sid, 'api_key': api_key, 'file_type': 'json'}, timeout=10)
                df = pd.DataFrame(r.json()['observations'])
                return pd.to_numeric(df['value'], errors='coerce')
            
            walcl = get('WALCL')
            tga = get('WTREGEN')
            rrp = get('RRPONTSYD')
            
            if all(x is not None for x in [walcl, tga, rrp]):
                result['net_liq'] = (walcl.iloc[-1]/1000) - tga.iloc[-1] - rrp.iloc[-1]
        except: pass
    return result
