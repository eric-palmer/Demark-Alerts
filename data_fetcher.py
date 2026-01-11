# data_fetcher.py - Data download functions with enhanced error handling
import yfinance as yf
import pandas as pd
import requests
import time
import datetime
import numpy as np
import os
from utils import get_session

CRYPTO_MAP = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana',
    'DOGE-USD': 'dogecoin',
    'SHIB-USD': 'shiba-inu',
    'PEPE-USD': 'pepe'
}

COINGECKO_DELAY = 1.5
_coingecko_last = 0

def safe_download(ticker, retries=3):
    """Download ticker data with fallback to CoinGecko for crypto"""
    global _coingecko_last
    session = get_session()
    
    # Try yfinance with longer delays and better error handling
    for attempt in range(retries):
        try:
            # Add delay to avoid rate limiting
            if attempt > 0:
                print(f"    Retry {attempt+1} for {ticker}")
                time.sleep(5)
            
            df = yf.download(
                ticker,
                period="2y",
                progress=False,
                auto_adjust=True,
                threads=False,  # Disable threading to avoid issues
                timeout=30,  # Longer timeout
                show_errors=False
            )
            
            if df.empty or len(df) < 50:
                if attempt < retries - 1:
                    continue
            else:
                # Normalize MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Validate data quality
                if len(df) >= 50 and df['Close'].notna().sum() > len(df) * 0.9:
                    return df
                    
        except Exception as e:
            error_msg = str(e)[:100]
            if attempt < retries - 1:
                time.sleep(5)
    
    # Fallback to CoinGecko for crypto
    if ticker in CRYPTO_MAP:
        time_since = time.time() - _coingecko_last
        if time_since < COINGECKO_DELAY:
            time.sleep(COINGECKO_DELAY - time_since)
        
        try:
            id_ = CRYPTO_MAP[ticker]
            
            # Fetch OHLC
            url = f"https://api.coingecko.com/api/v3/coins/{id_}/ohlc?vs_currency=usd&days=730"
            r = session.get(url, timeout=15)
            _coingecko_last = time.time()
            
            if r.status_code != 200:
                return None
            
            data = r.json()
            df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Fetch volume
            time.sleep(COINGECKO_DELAY)
            url_vol = f"https://api.coingecko.com/api/v3/coins/{id_}/market_chart?vs_currency=usd&days=730&interval=daily"
            r_vol = session.get(url_vol, timeout=15)
            _coingecko_last = time.time()
            
            if r_vol.status_code == 200:
                vol_data = r_vol.json()
                times = [v[0] for v in vol_data.get('total_volumes', [])]
                vols = [v[1] for v in vol_data.get('total_volumes', [])]
                df_vol = pd.DataFrame(
                    {'Volume': vols},
                    index=pd.to_datetime(times, unit='ms')
                )
                df = df.join(df_vol, how='left')
            
            df['Volume'] = df.get('Volume', 0).fillna(0)
            
            if len(df) >= 50:
                return df
                
        except Exception as e:
            pass
    
    return None

def get_futures():
    """Return list of key futures contracts"""
    return [
        'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F', 'FTSE=F',
        'CL=F', 'NG=F', 'RB=F', 'HO=F', 'BZ=F',
        'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F',
        'ZT=F', 'ZF=F', 'ZN=F', 'ZB=F',
        '6E=F', '6B=F', '6J=F', '6A=F', 'DX-Y.NYB',
        'ZC=F', 'ZS=F', 'ZW=F', 'ZL=F', 'ZM=F',
        'CC=F', 'KC=F', 'SB=F', 'CT=F', 'LE=F', 'HE=F'
    ]

def fetch_fred(series_id, start, api_key, session, retries=3):
    """Fetch FRED economic data series"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start.strftime('%Y-%m-%d'),
        'observation_end': datetime.datetime.now().strftime('%Y-%m-%d')
    }
    
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            
            data = r.json()
            
            if 'error_message' in data:
                raise ValueError(f"FRED error: {data['error_message']}")
            
            obs = data.get('observations', [])
            if not obs:
                raise ValueError(f"No data for {series_id}")
            
            df = pd.DataFrame(obs)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df[['value']].rename(columns={'value': series_id})
            df = df.replace('.', np.nan).astype(float)
            
            return df
            
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
    
    return pd.DataFrame()

def get_macro():
    """Fetch macro economic data from FRED and market benchmarks"""
    try:
        api_key = os.environ.get('FRED_API_KEY')
        
        if not api_key:
            print("ERROR: FRED_API_KEY not set")
            return None
        
        start = datetime.datetime.now() - datetime.timedelta(days=730)
        series = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'DGS10', 'DGS2', 'T5YIE']
        session = get_session()
        
        dfs = []
        for s in series:
            df = fetch_fred(s, start, api_key, session)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            print("ERROR: All FRED series failed")
            return None
        
        fred = pd.concat(dfs, axis=1)
        fred = fred.resample('D').ffill().dropna()
        
        result = {}
        
        # Calculate liquidity metrics
        if 'WALCL' in fred.columns and 'WTREGEN' in fred.columns and 'RRPONTSYD' in fred.columns:
            result['net_liq'] = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
            result['fed_assets'] = fred['WALCL'] / 1000
        
        # Yield curve
        if 'DGS10' in fred.columns and 'DGS2' in fred.columns:
            result['term_premia'] = fred['DGS10'] - fred['DGS2']
        
        # Inflation expectations
        if 'T5YIE' in fred.columns:
            result['inflation'] = fred['T5YIE']
        
        # Fetch SPY with retries
        print("  Fetching SPY...")
        spy = None
        for attempt in range(5):
            spy = safe_download('SPY')
            if spy is not None:
                print(f"  ✓ SPY fetched (attempt {attempt + 1})")
                break
            else:
                if attempt < 4:
                    print(f"  SPY retry {attempt + 1}/5...")
                    time.sleep(5)
        
        if spy is None:
            print("  WARNING: SPY unavailable after 5 attempts")
        else:
            result['spy'] = spy['Close']
        
        if not result:
            return None
        
        print(f"  ✓ Macro ready with {len(result)} metrics")
        return result
        
    except Exception as e:
        print(f"Macro error: {e}")
        import traceback
        traceback.print_exc()
        return None
