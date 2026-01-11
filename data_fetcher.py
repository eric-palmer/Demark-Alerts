# data_fetcher.py - Institutional Data (Raw & Filled)
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
    """Institutional Source: Tiingo"""
    try:
        # 1. Crypto Handling (btcusd)
        if '-USD' in ticker:
            sym = ticker.replace('-USD', '').lower() + 'usd'
            data = client.get_crypto_price_history(tickers=[sym], startDate='2022-01-01', resampleFreq='1day')
            df = pd.DataFrame(data[0].get('priceData', []))
            rename = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            df = df.rename(columns=rename)
        
        # 2. Stock Handling (FORCE RAW DATA)
        else:
            # We request everything but manually select the raw columns
            df = client.get_dataframe(ticker, startDate='2022-01-01')
            # Select RAW columns to match broker prices
            # If keys missing, fallback to whatever is there
            cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[cols]
            
            rename = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            df = df.rename(columns=rename)

        df['Date'] = pd.to_datetime(df.index).tz_localize(None)
        df = df.set_index('Date')
        
        # Force numeric & Fill Gaps (Crucial for ADX)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        return df.ffill().dropna()
    except Exception:
        return None

def fetch_fallback(ticker):
    """Backup Source: Yahoo"""
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
        
        return df.ffill().dropna()
    except: return None

def safe_download(ticker, client=None):
    if any(x in ticker for x in ['=F', 'DX-Y', '=X']):
        return fetch_fallback(ticker)

    df = None
    if client:
        time.sleep(PAUSE_SEC)
        df = fetch_tiingo(ticker, client)
    
    if df is None or len(df) < 5:
        df = fetch_fallback(ticker)
        
    if df is not None:
        # Final safety: Ensure we have enough rows for ADX(14)
        if len(df) > 20: 
            return df
            
    return None

def get_macro():
    """Market Radar Inputs"""
    api_key = os.environ.get('FRED_API_KEY')
    result = {'growth': None, 'inflation': None, 'net_liq': None}
    
    if api_key:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            import requests 
            def get(sid):
                r = requests.get(url, params={'series_id': sid, 'api_key': api_key, 'file_type': 'json'}, timeout=10)
                if r.status_code != 200: return None
                df = pd.DataFrame(r.json()['observations'])
                return pd.to_numeric(df['value'], errors='coerce')
            
            # Growth (ISM)
            ism = get('NAPM')
            if ism is not None and not ism.empty: result['growth'] = ism
            
            # Inflation (Breakevens)
            inf = get('T5YIE')
            if inf is not None and not inf.empty: result['inflation'] = inf

            # Liquidity
            walcl = get('WALCL')
            tga = get('WTREGEN')
            rrp = get('RRPONTSYD')
            
            if all(x is not None and not x.empty for x in [walcl, tga, rrp]):
                # Slice to min length to align
                min_len = min(len(walcl), len(tga), len(rrp))
                result['net_liq'] = (walcl.iloc[-min_len:].values/1000) - tga.iloc[-min_len:].values - rrp.iloc[-min_len:].values
                result['net_liq'] = pd.Series(result['net_liq'])
        except: pass
    return result
