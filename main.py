import yfinance as yf
import pandas as pd
import pandas_datareader.data as web 
import requests
import os
import time
import datetime
import numpy as np
import sys

# --- CONFIGURATION ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']

# Institutional Universe (Batched)
STRATEGIC_TICKERS = [
    'DJT', 'DOGE-USD', 'SHIB-USD', 'PEPE-USD', # Meme
    'BTC-USD', 'ETH-USD', 'SOL-USD', # Coins
    'BTDR', 'MARA', 'RIOT', 'HUT', 'CLSK', 'IREN', 'CIFR', 'CORZ', 'WULF', 'BITF', # Miners
    'IBIT', 'ETHA', 'BITQ', 'MSTR', 'COIN', 'HOOD', # Crypto Proxies
    'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F', 'PA=F', 'GLD', 'SLV', 'URA', 'CCJ', # Commodities/Energy
    'NVDA', 'SMH', 'TSM', 'AVGO', 'MSFT', 'GOOG', 'META', 'AMZN', 'AAPL', 'TSLA', 'NFLX', # Mag 7
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', # Sectors
    'BLK', 'BX', 'KKR', 'PLTR', 'SOFI' # Financials
]

FUTURES = ['ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F', 'ZB=F', 'ZN=F', 'DX-Y.NYB']

# --- TELEGRAM SENDER ---
def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: 
        print("⚠️ No Telegram Token/Chat ID found.")
        return
    
    # Chunking
    for i in range(0, len(message), 4000):
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": message[i:i+4000], "parse_mode": "Markdown"})
            time.sleep(1)
        except Exception as e: 
            print(f"Telegram Error: {e}")

# --- DATA ENGINE (BATCHED) ---
def get_batch_data(tickers):
    """Downloads all tickers in ONE request to avoid bans."""
    print(f"📥 Downloading {len(tickers)} tickers...")
    try:
        # User Agent to prevent 403s
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})
        
        data = yf.download(tickers, period="1y", group_by='ticker', progress=False, auto_adjust=True, threads=True, session=session)
        return data
    except Exception as e:
        print(f"❌ Batch Download Failed: {e}")
        return pd.DataFrame()

def get_macro_data():
    """Fetches FRED data with fallback."""
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=730)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD', 'DGS10', 'DGS2', 'T5YIE'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        return {
            'net_liq': net_liq, 
            'term_premia': fred['DGS10'] - fred['DGS2'], 
            'inflation': fred['T5YIE']
        }
    except Exception as e:
        print(f"⚠️ Macro Data Failed: {e}")
        return None

# --- TECHNICAL INDICATORS (SHANNON / DEMARK / SQUEEZE) ---
def calculate_indicators(df):
    if df.empty or len(df) < 200: return None
    
    # 1. Brian Shannon's "Anchored" & Moving Averages
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()   # Momentum (Near Term)
    df['SMA_10'] = df['Close'].rolling(10).mean()                # Fast Trend
    df['SMA_20'] = df['Close'].rolling(20).mean()                # Intermediate Trend
    df['SMA_50'] = df['Close'].rolling(50).mean()                # Institutional Defense
    
    # Approximate YTD VWAP (Start from index ~252 trading days ago or start of year)
    # Since we downloaded 1y, we'll use the whole dataframe as a proxy for a "Major Low" anchor
    df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical'] * df['Volume']
    df['Total_VP'] = df['VP'].cumsum()
    df['Total_Vol'] = df['Volume'].cumsum()
    df['AVWAP'] = df['Total_VP'] / df['Total_Vol'] # Anchored to start of data (1 Year)

    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. DeMark (9 Setup)
    df['Setup'] = np.where(df['Close'] < df['Close'].shift(4), 1, 0)
    df['Buy_Seq'] = df['Setup'].rolling(9).sum()
    df['Sell_Setup'] = np.where(df['Close'] > df['Close'].shift(4), 1, 0)
    df['Sell_Seq'] = df['Sell_Setup'].rolling(9).sum()
    
    # 4. Volatility Squeeze
    std = df['Close'].rolling(20).std()
    upper_bb = df['SMA_20'] + (std * 2)
    lower_bb = df['SMA_20'] - (std * 2)
    
    tr = df['High'] - df['Low']
    atr = tr.rolling(20).mean()
    upper_kc = df['SMA_20'] + (atr * 1.5)
    lower_kc = df['SMA_20'] - (atr * 1.5)
    
    df['Squeeze'] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    df['ATR'] = atr
    
    return df

# --- ANALYSIS ENGINE ---
def analyze_single_ticker(ticker, df):
    try:
        # Extract ticker data
        if isinstance(df.columns, pd.MultiIndex):
            if ticker not in df.columns.get_level_values(0): return None
            t_df = df[ticker].copy()
        else:
            t_df = df.copy()
            
        t_df = t_df.dropna()
        t_df = calculate_indicators(t_df)
        
        if t_df is None: return None
        
        last = t_df.iloc[-1]
        prev = t_df.iloc[-2]
        
        # --- SHANNON SIGNALS ---
        # Near Term: Price vs 5 EMA
        near_term = "BULLISH" if last['Close'] > last['EMA_5'] else "BEARISH"
        
        # Long Term: Price vs AVWAP / 50 SMA
        long_term = "ACCUMULATION"
        if last['Close'] > last['SMA_50'] and last['SMA_10'] > last['SMA_20']:
            long_term = "STAGE 2 (Uptrend)"
        elif last['Close'] < last['SMA_50']:
            long_term = "STAGE 4 (Downtrend)"
            
        # Shannon "Buy" Setup: 10 crosses above 20, while above 50
        shannon_buy = (prev['SMA_10'] < prev['SMA_20']) and (last['SMA_10'] > last['SMA_20']) and (last['Close'] > last['SMA_50'])
        
        # --- ALERTS ---
        signals = []
        if shannon_buy: signals.append("ALPHATRENDS BUY (10/20 Cross)")
        if last['Buy_Seq'] == 9: signals.append("DEMARK BUY 9")
        if last['Sell_Seq'] == 9: signals.append("DEMARK SELL 9")
        if last['Squeeze']: signals.append("VOLATILITY SQUEEZE")
        if last['RSI'] < 30: signals.append("RSI OVERSOLD")
        
        # Targets
        target = last['Close'] + (last['ATR'] * 2)
        stop = last['Close'] - (last['ATR'] * 1.5)
        
        return {
            'ticker': ticker,
            'price': last['Close'],
            'signals': signals,
            'near_term': near_term,
            'long_term': long_term,
            'avwap': last['AVWAP'],
            'target': target,
            'stop': stop,
            'rsi': last['RSI']
        }
    except: return None

# --- MAIN RUNNER ---
if __name__ == "__main__":
    print("1. Fetching Macro...")
    macro = get_macro_data()
    
    macro_msg = "🌍 **GLOBAL MACRO INSIGHTS** 🌍\n"
    if macro:
        liq_roc = macro['net_liq'].pct_change(63).iloc[-1]
        tp_trend = "RISING (Bearish)" if macro['term_premia'].iloc[-1] > macro['term_premia'].iloc[-20] else "FALLING (Bullish)"
        regime = "RISK ON 🟢" if liq_roc > 0 else "RISK OFF 🔴"
        macro_msg += f"📊 **Regime:** {regime}\n"
        macro_msg += f"   └ Liq Trend: {liq_roc*100:.2f}%\n"
        macro_msg += f"   └ Term Premia: {tp_trend}\n"
    else:
        macro_msg += "⚠️ Macro Data Unavailable (Timeout)\n"
    
    send_telegram_alert(macro_msg)
    
    print("2. Batch Downloading Market Data...")
    all_tickers = list(set(CURRENT_PORTFOLIO + STRATEGIC_TICKERS + FUTURES))
    market_data = get_batch_data(all_tickers)
    
    # --- PORTFOLIO ANALYSIS ---
    p_msg = "💼 **CURRENT PORTFOLIO (Technical Health)** 💼\n"
    for t in CURRENT_PORTFOLIO:
        res = analyze_single_ticker(t, market_data)
        if res:
            p_msg += f"🔹 **{t}**: ${res['price']:.2f}\n"
            p_msg += f"   └ Near Term (5EMA): {res['near_term']}\n"
            p_msg += f"   └ Cycle (Shannon): {res['long_term']}\n"
            p_msg += f"   └ AVWAP (1Y): ${res['avwap']:.2f}\n"
            p_msg += f"   🎯 Target: ${res['target']:.2f} | 🛑 Stop: ${res['stop']:.2f}\n\n"
        else:
            p_msg += f"⚠️ {t}: Data Unavailable\n"
    send_telegram_alert(p_msg)
    
    # --- MARKET SCANNER ---
    print("3. Scanning for Alpha...")
    alerts = []
    
    for t in all_tickers:
        res = analyze_single_ticker(t, market_data)
        if res and res['signals']: # Only report active signals
            alerts.append(res)
                
    # Sort by number of signals
    alerts.sort(key=lambda x: len(x['signals']), reverse=True)
    
    a_msg = "🔔 **ALPHATRENDS & DEMARK SIGNALS** 🔔\n"
    if alerts:
        for a in alerts[:20]: # Top 20
            icon = "🔥" if len(a['signals']) > 1 else "⚡"
            a_msg += f"{icon} **{a['ticker']}**: ${a['price']:.2f}\n"
            a_msg += f"   └ {', '.join(a['signals'])}\n"
            a_msg += f"   └ Trend: {a['long_term']}\n"
            a_msg += f"   └ ⏳ Timing: {a['near_term']} (vs 5EMA)\n"
    else:
        a_msg += "No major signals found today."
        
    send_telegram_alert(a_msg)
    print("✅ Done.")
