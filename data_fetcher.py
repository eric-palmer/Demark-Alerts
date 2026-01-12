# data_fetcher.py - Institutional Router (Data Health Check)
import pandas as pd
import time
import os
import datetime
import yfinance as yf
from tiingo import TiingoClient

PAUSE_SEC = 0.2

def get_tiingo_client():
    api_key = os.environ.get('TIINGO_API_KEY')
    return TiingoClient({'api_key': api_key, 'session': True}) if api_key else None

def fetch_tiingo(ticker, client):
    try:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
        
        # 1. Crypto
        if '-USD' in ticker:
            sym = ticker.replace('-USD', '').lower() + 'usd'
            data = client.get_crypto_price_history(tickers=[sym], startDate=start_date, resampleFreq='1day')
            df = pd.DataFrame(data[0].get('priceData', []))
            rename = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
            df = df.rename(columns=rename)
        
        # 2. Stocks
        else:
            df = client.get_dataframe(ticker, startDate=start_date)
            
            # HEALTH CHECK: Do Adjusted columns actually have data?
            # Check if 'adjClose' exists and is not mostly NaN
            use_adj = False
            if 'adjClose' in df.columns:
                if df['adjClose'].isnull().sum() < len(df) * 0.5: # If >50% valid
                    use_adj = True
            
            if use_adj:
                # Use Adjusted
                df = df[['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            else:
                # Fallback to RAW (We know this works from your log)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        df['Date'] = pd.to_datetime(df.index).tz_localize(None)
        df = df.set_index('Date')
        
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        return df.interpolate(method='time').ffill().bfill()
    except Exception as e:
        print(f"Tiingo Error {ticker}: {e}")
        return None

def fetch_fallback(ticker):
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
        
        return df.interpolate(method='time').ffill().bfill()
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
            
            ism = get('NAPM')
            if ism is not None: result['growth'] = ism
            inf = get('T5YIE')
            if inf is not None: result['inflation'] = inf
            walcl = get('WALCL'); tga = get('WTREGEN'); rrp = get('RRPONTSYD')
            if all(x is not None for x in [walcl, tga, rrp]):
                min_len = min(len(walcl), len(tga), len(rrp))
                result['net_liq'] = (walcl.iloc[-min_len:].values/1000) - tga.iloc[-min_len:].values - rrp.iloc[-min_len:].values
                result['net_liq'] = pd.Series(result['net_liq'])
        except: pass
    return result
