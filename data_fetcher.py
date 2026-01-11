# data_fetcher.py - Institutional Router (Deduplicated)
import pandas as pd
import time
import os
import yfinance as yf
from tiingo import TiingoClient

# --- Configuration ---
PAUSE_SEC = 0.2

def get_tiingo_client():
    api_key = os.environ.get('TIINGO_API_KEY')
    return TiingoClient({'api_key': api_key, 'session': True}) if api_key else None

def fetch_tiingo(ticker, client):
    """Institutional Source: Tiingo (Clean & Deduped)"""
    try:
        # 1. Crypto Handling
        if '-USD' in ticker:
            sym = ticker.replace('-USD', '').lower() + 'usd'
            data = client.get_crypto_price_history(tickers=[sym], startDate='2022-01-01', resampleFreq='1day')
            df = pd.DataFrame(data[0].get('priceData', []))
            
            # Crypto is simple, just map keys
            rename = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            df = df.rename(columns=rename)
        
        # 2. Stock Handling (The Fix for Duplicates)
        else:
            df = client.get_dataframe(ticker, startDate='2022-01-01')
            
            # PRIORITY: Keep Adjusted columns, drop raw ones to prevent 'High' vs 'High' collision
            if 'adjClose' in df.columns:
                df = df[['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # Direct rename
            else:
                # Fallback if no adj columns (rare)
                rename = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
                df = df.rename(columns=rename)

        df['Date'] = pd.to_datetime(df.index).tz_localize(None)
        df = df.set_index('Date')
        
        # Final Verification
        if 'High' not in df.columns or 'Close' not in df.columns: return None
        
        # Remove any lingering duplicates just in case
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
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
            
        for c in ['Open', 'High', 'Low', 'Close']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Deduplicate
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except: return None

def safe_download(ticker, client=None):
    """Smart Router"""
    # Futures -> Yahoo
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
        
    if df is not None:
        df = df.ffill().dropna()
        if len(df) > 30: return df
            
    return None

def get_macro():
    """Context Fetcher (Crash Proof)"""
    spy = fetch_fallback('SPY')
    api_key = os.environ.get('FRED_API_KEY')
    result = {'net_liq': None, 'spy': None}
    
    if spy is not None and 'Close' in spy.columns:
        result['spy'] = spy['Close']
    
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
