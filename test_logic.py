# test_logic.py - Single File Diagnostic (No Imports)
import os
import time
import datetime
import pandas as pd
import numpy as np
from tiingo import TiingoClient

print("🚀 STARTING DIAGNOSTIC RUN...")

# --- 1. SETUP CLIENT ---
api_key = os.environ.get('TIINGO_API_KEY')
if not api_key:
    print("❌ ERROR: TIINGO_API_KEY is missing.")
    exit()
client = TiingoClient({'api_key': api_key, 'session': True})

# --- 2. MATH FUNCTIONS (Defined Locally to avoid Import Errors) ---
def sanitize(series):
    return series.fillna(0)

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return sanitize((100 - (100 / (1 + rs))).fillna(50))

def calc_adx(df, period=14):
    up = df['High'].diff(); down = -df['Low'].diff()
    p_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    m_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
    tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    tr = tr.replace(0, 0.0001)
    atr = tr.ewm(alpha=1/period).mean()
    p_di = 100 * (p_dm.ewm(alpha=1/period).mean() / atr)
    m_di = 100 * (m_dm.ewm(alpha=1/period).mean() / atr)
    dx = 100 * abs(p_di - m_di) / (p_di + m_di)
    return sanitize(dx.ewm(alpha=1/period).mean())

def calc_demark(df):
    c = df['Close'].values
    buy_setup = np.zeros(len(c), dtype=int)
    sell_setup = np.zeros(len(c), dtype=int)
    # Setup Logic
    for i in range(4, len(c)):
        if c[i] < c[i-4]: buy_setup[i] = buy_setup[i-1] + 1
        else: buy_setup[i] = 0
        if c[i] > c[i-4]: sell_setup[i] = sell_setup[i-1] + 1
        else: sell_setup[i] = 0
    return buy_setup, sell_setup

# --- 3. EXECUTION ---
def test_ticker(ticker):
    print(f"\n🔎 TESTING {ticker}...")
    try:
        # Fetch
        start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
        df = client.get_dataframe(ticker, startDate=start_date)
        
        # Standardize Columns
        df.columns = [c.lower() for c in df.columns]
        rename_map = {'adjclose': 'Close', 'adjhigh': 'High', 'adjlow': 'Low', 'adjopen': 'Open', 'adjvolume': 'Volume',
                      'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'}
        
        # Smart Rename
        final_df = pd.DataFrame(index=df.index)
        for k, v in rename_map.items():
            if k in df.columns and v not in final_df.columns:
                final_df[v] = df[k]
        
        df = final_df.dropna()
        print(f"   ✅ Data Fetched: {len(df)} rows. Last Price: {df['Close'].iloc[-1]}")
        
        # Run Math
        df['RSI'] = calc_rsi(df['Close'])
        df['ADX'] = calc_adx(df)
        bs, ss = calc_demark(df)
        df['Buy_Setup'] = bs
        df['Sell_Setup'] = ss
        
        # Inspect Results
        last = df.iloc[-1]
        print(f"   📊 RESULTS:")
        print(f"      - RSI: {last['RSI']:.2f}")
        print(f"      - ADX: {last['ADX']:.2f}")
        print(f"      - DeMark Buy: {int(last['Buy_Setup'])}")
        print(f"      - DeMark Sell: {int(last['Sell_Setup'])}")
        
        # Print Raw Data for Verification
        print("\n   Last 5 Days (Raw Data):")
        print(df[['Close', 'RSI', 'ADX', 'Buy_Setup', 'Sell_Setup']].tail(5))
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

# Run Tests
test_ticker('SLV')
test_ticker('DJT')
print("\n✅ DIAGNOSTIC COMPLETE")
