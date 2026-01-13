# data_fetcher.py - Institutional Data (Deep History)
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

def standardize_columns(df):
    """Maps Adjusted Columns to Standard Names"""
    df.columns = [c.lower() for c in df.columns]
    final_df = pd.DataFrame(index=df.index)
    
    # Priority: Adjusted Data
    mapping = {
        'Open': 'adjopen', 'High': 'adjhigh', 'Low': 'adjlow', 'Close': 'adjclose', 'Volume': 'adjvolume'
    }
    raw_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }
    
    for target, adj_col in mapping.items():
        if adj_col in df.columns:
            final_df[target] = df[adj_col]
        elif raw_mapping[target] in df.columns:
            final_df[target] = df[raw_mapping[target]]
        else:
            final_df[target] = 0.0
            
    return final_df

def fetch_tiingo(ticker, client):
    try:
        # FETCH 4 YEARS (Required for accurate Weekly DeMark 13 counts)
        start_date = (datetime.datetime.now() - datetime.timedelta(days=1460)).strftime('%Y-%m-%d')
        
        if '-USD' in ticker:
            sym = ticker.replace('-USD', '').lower() + 'usd'
            data = client.get_crypto_price_history(tickers=[sym], startDate=start_date, resampleFreq='1day')
            df = pd.DataFrame(data[0].get('priceData', []))
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df = df.set_index('date')
        else:
            df = client.get_dataframe(ticker, startDate=start_date)
            df.index = df.index.tz_localize(None)

        df = standardize_columns(df)
        
        # Zero protection
        mask = df['High'] <= 0
        if mask.any():
            df.loc[mask, 'High'] = df.loc[mask, 'Close']
            df.loc[mask, 'Low'] = df.loc[mask, 'Close']

        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        return df.bfill().ffill()
    except: return None

def safe_download(ticker, client=None):
    if any(x in ticker for x in ['=F', 'DX-Y', '=X']): return None 
    if client:
        time.sleep(PAUSE_SEC)
        return fetch_tiingo(ticker, client)
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
            if ism is not None: result['growth'] = ism.rolling(3).mean()
            inf = get('T5YIE')
            if inf is not None: result['inflation'] = inf.rolling(10).mean()
            walcl = get('WALCL'); tga = get('WTREGEN'); rrp = get('RRPONTSYD')
            if all(x is not None for x in [walcl, tga, rrp]):
                min_len = min(len(walcl), len(tga), len(rrp))
                net = (walcl.iloc[-min_len:].values/1000) - tga.iloc[-min_len:].values - rrp.iloc[-min_len:].values
                result['net_liq'] = pd.Series(net).rolling(10).mean()
        except: pass
    return result
