# data_fetcher.py - Institutional Router (Brute Force Mapping)
import pandas as pd
import time
import os
import datetime
import yfinance as yf
from tiingo import TiingoClient

# --- Configuration ---
PAUSE_SEC = 0.2

def get_tiingo_client():
    api_key = os.environ.get('TIINGO_API_KEY')
    return TiingoClient({'api_key': api_key, 'session': True}) if api_key else None

def standardize_columns(df):
    """
    STRATEGY: Brute Force Column Mapping
    Don't guess names. Find the column that matches our needs.
    """
    df.columns = [c.lower() for c in df.columns] # Normalize to lowercase
    
    # Define targets and their possible aliases
    mapping = {
        'Open': ['adjopen', 'open'],
        'High': ['adjhigh', 'high'],
        'Low': ['adjlow', 'low'],
        'Close': ['adjclose', 'close'],
        'Volume': ['adjvolume', 'volume']
    }
    
    final_df = pd.DataFrame(index=df.index)
    
    for target, aliases in mapping.items():
        found = False
        for alias in aliases:
            if alias in df.columns:
                final_df[target] = df[alias]
                found = True
                break
        
        # Critical Safety: If 'High' is missing but 'Close' exists, 
        # approximate High=Close to prevent ADX crash (Better than NaN)
        if not found:
            if target in ['High', 'Low', 'Open'] and 'close' in df.columns:
                final_df[target] = df['close']
            else:
                final_df[target] = 0.0
                
    return final_df

def fetch_tiingo(ticker, client):
    try:
        # 1. Fetch 2 Years (Warmup Data)
        start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
        
        # 2. Raw Request
        if '-USD' in ticker:
            sym = ticker.replace('-USD', '').lower() + 'usd'
            data = client.get_crypto_price_history(tickers=[sym], startDate=start_date, resampleFreq='1day')
            df = pd.DataFrame(data[0].get('priceData', []))
            # Crypto needs date set to index before mapping
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.set_index('date')
        else:
            df = client.get_dataframe(ticker, startDate=start_date)
            # Stocks index is already date, just localize
            df.index = df.index.tz_localize(None)

        # 3. Brute Force Map
        df = standardize_columns(df)
        
        # 4. Force Float & Fill Gaps
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        return df.interpolate(method='time').ffill().bfill()
    except Exception:
        return None

def fetch_fallback(ticker):
    try:
        dat = yf.Ticker(ticker)
        df = dat.history(period="2y", auto_adjust=True)
        if df.empty: return None
        
        df = df.reset_index()
        # Ensure date column exists
        if 'Date' in df.columns:
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df = df.set_index('date')
            
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Use same mapper
        df = standardize_columns(df)
        
        return df.interpolate(method='time').ffill().bfill()
    except: return None

def safe_download(ticker, client=None):
    # Futures -> Yahoo
    if any(x in ticker for x in ['=F', 'DX-Y', '=X']):
        return fetch_fallback(ticker)

    df = None
    if client:
        time.sleep(PAUSE_SEC)
        df = fetch_tiingo(ticker, client)
    
    if df is None or len(df) < 5:
        df = fetch_fallback(ticker)
        
    if df is not None:
        if len(df) > 30: return df
            
    return None

def get_macro():
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
            
            # STRATEGY: Increase smoothing to 10 days to match Market Radar stability
            ism = get('NAPM')
            if ism is not None: 
                result['growth'] = ism.rolling(3).mean() # Quarterly smooth
            
            inf = get('T5YIE')
            if inf is not None: 
                result['inflation'] = inf.rolling(10).mean() # 10-day smooth

            walcl = get('WALCL'); tga = get('WTREGEN'); rrp = get('RRPONTSYD')
            if all(x is not None for x in [walcl, tga, rrp]):
                min_len = min(len(walcl), len(tga), len(rrp))
                net = (walcl.iloc[-min_len:].values/1000) - tga.iloc[-min_len:].values - rrp.iloc[-min_len:].values
                result['net_liq'] = pd.Series(net).rolling(10).mean()
        except: pass
    return result
