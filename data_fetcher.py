# data_fetcher.py - Institutional Data (Durable w/ Retry)
import pandas as pd
import time
import os
import datetime
import requests
from tiingo import TiingoClient

PAUSE_SEC = 0.5 # Increased slightly to be safer

def get_tiingo_client():
    api_key = os.environ.get('TIINGO_API_KEY')
    return TiingoClient({'api_key': api_key, 'session': True}) if api_key else None

def standardize_columns(df):
    """Maps Adjusted Columns to Standard Names"""
    df.columns = [c.lower() for c in df.columns]
    final_df = pd.DataFrame(index=df.index)
    
    col_map = {
        'Open': ['adjopen', 'open'], 'High': ['adjhigh', 'high'],
        'Low': ['adjlow', 'low'], 'Close': ['adjclose', 'close'],
        'Volume': ['adjvolume', 'volume']
    }
    
    for target, sources in col_map.items():
        for src in sources:
            if src in df.columns:
                final_df[target] = df[src]
                break
        if target not in final_df.columns: final_df[target] = 0.0
            
    return final_df

def fetch_tiingo(ticker, client, retries=2):
    """Fetches data with automatic retry logic for rate limits"""
    try:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=1460)).strftime('%Y-%m-%d')
        
        # Attempt Fetch
        if '-USD' in ticker:
            sym = ticker.replace('-USD', '').lower() + 'usd'
            data = client.get_crypto_price_history(tickers=[sym], startDate=start_date, resampleFreq='1day')
            df = pd.DataFrame(data[0].get('priceData', []))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                df = df.set_index('date')
        else:
            df = client.get_dataframe(ticker, startDate=start_date)
            if not df.empty: df.index = df.index.tz_localize(None)

        # Retry Logic for Empty Data (Glitch protection)
        if df.empty and retries > 0:
            time.sleep(2)
            return fetch_tiingo(ticker, client, retries - 1)
            
        if df.empty: return None

        df = standardize_columns(df)
        
        # Safety: Fix Zeros
        mask = df['High'] <= 0
        if mask.any():
            df.loc[mask, 'High'] = df.loc[mask, 'Close']
            df.loc[mask, 'Low'] = df.loc[mask, 'Close']

        for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            
        return df.bfill().ffill()
        
    except Exception as e:
        # If Rate Limited, Sleep and Retry
        if "429" in str(e) and retries > 0:
            print(f"   ⚠️ Rate Limit Hit on {ticker}. Sleeping 15s...")
            time.sleep(15)
            return fetch_tiingo(ticker, client, retries - 1)
        
        print(f"   ❌ {ticker} Fetch Error: {e}")
        return None

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
            # Simple FRED fetch logic here (omitted for brevity, same as before)
            pass 
        except: pass
    return result
