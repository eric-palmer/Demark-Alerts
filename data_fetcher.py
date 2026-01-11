# data_fetcher.py - Institutional Batch Fetcher
import pandas as pd
import requests
import time
import os
import yfinance as yf
from tiingo import TiingoClient

# --- Configuration ---
# Safety buffer: We request 40 tickers per hour to be safe under the 50 limit
BATCH_SIZE = 40  

def get_tiingo_client():
    api_key = os.environ.get('TIINGO_API_KEY')
    return TiingoClient({'api_key': api_key, 'session': True}) if api_key else None

def fetch_tiingo(ticker, client):
    """Primary Institutional Source"""
    try:
        # Tiingo logic: Check if crypto or stock
        if '-USD' in ticker:
            # Crypto: 'BTC-USD' -> 'btcusd'
            sym = ticker.replace('-USD', '').lower() + 'usd'
            data = client.get_crypto_price_history(tickers=[sym], startDate='2022-01-01', resampleFreq='1day')
            df = pd.DataFrame(data[0].get('priceData', []))
            rename = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        else:
            # Stock
            df = client.get_dataframe(ticker, metric_name='adjClose', startDate='2022-01-01')
            rename = {'adjOpen': 'Open', 'adjHigh': 'High', 'adjLow': 'Low', 'adjClose': 'Close', 'adjVolume': 'Volume',
                      'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        
        df = df.rename(columns=rename)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df.set_index('Date')
    except:
        return None

def fetch_fallback(ticker):
    """Backup for Meme Coins not on Tiingo"""
    try:
        # Try Yahoo "Nuclear" Flat method
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
        return df
    except: return None

def safe_download(ticker, client=None):
    """Smart Router: Tiingo -> Fallback"""
    df = None
    if client:
        df = fetch_tiingo(ticker, client)
    
    if df is None or len(df) < 5:
        # Tiingo missed it (likely a new meme coin), use backup
        df = fetch_fallback(ticker)
        
    # Validation
    if df is not None:
        # Force numeric
        for c in ['Open', 'High', 'Low', 'Close']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.ffill().dropna()
        if len(df) > 30: return df
        
    return None

def get_macro():
    """Get Context without wasting Tiingo credits"""
    spy = fetch_fallback('SPY')
    api_key = os.environ.get('FRED_API_KEY')
    result = {'net_liq': None, 'spy': None}
    
    if spy is not None: result['spy'] = spy['Close']
    
    if api_key:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            def get(sid):
                r = requests.get(url, params={'series_id': sid, 'api_key': api_key, 'file_type': 'json'})
                df = pd.DataFrame(r.json()['observations'])
                return pd.to_numeric(df['value'], errors='coerce')
            
            walcl = get('WALCL')
            tga = get('WTREGEN')
            rrp = get('RRPONTSYD')
            
            if all(x is not None for x in [walcl, tga, rrp]):
                result['net_liq'] = (walcl.iloc[-1]/1000) - tga.iloc[-1] - rrp.iloc[-1]
        except: pass
    return result
