# data_fetcher.py - Data download functions
import yfinance as yf
import pandas as pd
import requests
import time
import datetime
import numpy as np
from utils import get_session

CRYPTO_MAP = {
    'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'SOL-USD': 'solana',
    'DOGE-USD': 'dogecoin', 'SHIB-USD': 'shiba-inu', 'PEPE-USD': 'pepe'
}
COINGECKO_DELAY = 1.5
_coingecko_last = 0

def safe_download(ticker, retries=3):
    global _coingecko_last
    session = get_session()
    
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period="2y", progress=False, auto_adjust=True, session=session, timeout=15)
            if df.empty or len(df) < 50:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) >= 50 and df['Close'].notna().sum() > len(df) * 0.9:
                    return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
    
    if ticker in CRYPTO_MAP:
        time_since = time.time() - _coingecko_last
        if time_since < COINGECKO_DELAY:
            time.sleep(COINGECKO_DELAY - time_since)
        try:
            id_ = CRYPTO_MAP[ticker]
            url = f"https://api.coingecko.com/api/v3/coins/{id_}/ohlc?vs_currency=usd&days=730"
            r = session.get(url, timeout=15)
            _coingecko_last = time.time()
            if r.status_code != 200:
                return None
            data = r.json()
            df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            time.sleep(COINGECKO_DELAY)
            url_vol = f"https://api.coingecko.com/api/v3/coins/{id_}/market_chart?vs_currency=usd&days=730&interval=daily"
            r_vol = session.get(url_vol, timeout=15)
            _coingecko_last = time.time()
            if r_vol.status_code == 200:
                vol_data = r_vol.json()
                times = [v[0] for v in vol_data.get('total_volumes', [])]
                vols = [v[1] for v in vol_data.get('total_volumes', [])]
                df_vol = pd.DataFrame({'Volume': vols}, index=pd.to_datetime(times, unit='ms'))
                df = df.join(df_vol, how='left')
            df['Volume'] = df.get('Volume', 0).fillna(0)
            if len(df) >= 50:
                return df
        except Exception as e:
            print(f"CoinGecko error {ticker}: {e}")
    return None

def get_futures():
    return ['ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'CL=F', 'NG=F', 'GC=F', 'SI=F', 'ZN=F', 'ZB=F', '6E=F']

def fetch_fred(series_id, start, api_key, session, retries=3):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id, 'api_key': api_key, 'file_type': 'json',
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
    import os
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
            raise ValueError("All FRED series failed")
        fred = pd.concat(dfs, axis=1)
        fred = fred.resample('D').ffill().dropna()
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        term = fred['DGS10'] - fred['DGS2']
        spy = safe_download('SPY')
        if spy is None:
            raise ValueError("SPY unavailable")
        return {
            'net_liq': net_liq, 'term_premia': term,
            'inflation': fred['T5YIE'], 'fed_assets': fred['WALCL'] / 1000,
            'spy': spy['Close']
        }
    except Exception as e:
        print(f"Macro error: {e}")
        return None
