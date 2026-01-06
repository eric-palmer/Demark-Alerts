import yfinance as yf
import pandas as pd
import requests
import os
import time
import io
import numpy as np

# --- CONFIGURATION ---
STRATEGIC_TICKERS = ['IBIT', 'ETHA', 'GLD', 'SLV', 'PALL', 'PPLT']

# --- MACRO REGIME ENGINE ---
def get_market_regime():
    """
    Analyzes 'Market Radar' style liquidity proxies to determine Risk Regime.
    """
    try:
        # Tickers: SPX (Eq), HYG (Credit), BTC (Liquidity), DXY (Currency), TNX (Rates)
        tickers = ['^GSPC', 'HYG', 'BTC-USD', 'DX-Y.NYB', '^TNX']
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
        
        sma50 = data.rolling(window=50).mean().iloc[-1]
        sma200 = data.rolling(window=200).mean().iloc[-1]
        price = data.iloc[-1]
        
        score = 0
        reasons = []
        
        # A. LIQUIDITY (Bitcoin)
        if price['BTC-USD'] > sma50['BTC-USD']:
            score += 25
            reasons.append("🟢 Crypto Liquidity Expanding")
        else:
            reasons.append("🔴 Crypto Liquidity Contracting")
            
        # B. CREDIT (HYG)
        if price['HYG'] > sma200['HYG']:
            score += 25
            reasons.append("🟢 Credit Markets Healthy")
        else:
            reasons.append("🔴 Credit Stress Detected")
            
        # C. CURRENCY (DXY)
        if price['DX-Y.NYB'] < sma200['DX-Y.NYB']:
            score += 25
            reasons.append("🟢 Dollar Weak (Pro-Liquidity)")
        else:
            reasons.append("🔴 Dollar Strong (Tightening)")
            
        # D. EQUITY TREND (SPX)
        if price['^GSPC'] > sma200['^GSPC']:
            score += 25
            reasons.append("🟢 Stocks in Bull Trend")
        else:
            reasons.append("🔴 Stocks in Bear Trend")

        if score >= 75:
            regime = "RISK ON (Aggressive)"
            color = "🟢"
        elif score >= 50:
            regime = "NEUTRAL (Selective)"
            color = "🟡"
        else:
            regime = "RISK OFF (Defensive)"
            color = "🔴"
            
        return {"regime": regime, "color": color, "score": score, "reasons": reasons}

    except Exception as e:
        print(f"Macro Error: {e}")
        return None

# --- TICKER SCRAPERS ---
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.read_html(io.StringIO(response.text))[0]
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except: return []

def get_nasdaq_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            if 'Ticker' in table.columns: return table['Ticker'].tolist()
            if 'Symbol' in table.columns: return table['Symbol'].tolist()
        return []
    except: return []

# --- DEMARK ENGINE (UPDATED 13 PERFECTION) ---
def calculate_demark_indicators(df):
    # Setup Logic
    df['Close_4'] = df['Close'].shift(4)
    df['Buy_Setup'] = 0
    df['Sell_Setup'] = 0
    
    # Countdown Logic
    df['Buy_Countdown'] = 0
    df['Sell_Countdown'] = 0
    df['Buy_13_Perfected'] = False
    df['Sell_13_Perfected'] = False

    buy_seq = 0
    sell_seq = 0
    
    # State for Countdown
    active_buy_countdown = False
    buy_count = 0
    buy_countdown_idxs = [] # Track indices to check perfection against Bar 8

    active_sell_countdown = False
    sell_count = 0
    sell_countdown_idxs = []

    closes = df['Close'].values
    closes_4 = df['Close_4'].values
    lows = df['Low'].values
    highs = df['High'].values
    
    for i in range(4, len(df)):
        # --- SETUP (9) ---
        if closes[i] < closes_4[i]:
            buy_seq += 1
        else:
            buy_seq = 0
        df.iloc[i, df.columns.get_loc('Buy_Setup')] = buy_seq
        
        if closes[i] > closes_4[i]:
            sell_seq += 1
        else:
            sell_seq = 0
        df.iloc[i, df.columns.get_loc('Sell_Setup')] = sell_seq
        
        # --- TRIGGER COUNTDOWN ---
        if buy_seq == 9:
            active_buy_countdown = True
            buy_count = 0 
            buy_countdown_idxs = []
            active_sell_countdown = False 
            
        if sell_seq == 9:
            active_sell_countdown = True
            sell_count = 0 
            sell_countdown_idxs = []
            active_buy_countdown = False 

        # --- COUNTDOWN (13) ---
        if active_buy_countdown:
            if closes[i] <= lows[i-2]:
                buy_count += 1
                buy_countdown_idxs.append(i)
                df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_count
                
                if buy_count == 13:
                    # PERFECTION CHECK: Low of 13 <= Low of Bar 8 (Index 7)
                    if len(buy_countdown_idxs) >= 8:
                        idx_8 = buy_countdown_idxs[7]
                        if lows[i] <= lows[idx_8]:
                            df.iloc[i, df.columns.get_loc('Buy_13_Perfected')] = True
                    active_buy_countdown = False

        if active_sell_countdown:
            if closes[i] >= highs[i-2]:
                sell_count += 1
                sell_countdown_idxs.append(i)
                df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_count
                
                if sell_count == 13:
                    # PERFECTION CHECK: High of 13 >= High of Bar 8 (Index 7)
                    if len(sell_countdown_idxs) >= 8:
                        idx_8 = sell_countdown_idxs[7]
                        if highs[i] >= highs[idx_8]:
                            df.iloc[i, df.columns.get_loc('Sell_13_Perfected')] = True
                    active_sell_countdown = False

    return df

def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if 'Close' not in df.columns: return None

        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df = calculate_demark_indicators(df)
        
        last_row = df.iloc[-1]
        price = last_row['Close']
        sma_200 = last_row['SMA_200']
        
        signal = None
        
        # --- 13 CHECK (PRIORITY) ---
        if last_row['Buy_Countdown'] == 13:
            perfected = last_row['Buy_13_Perfected']
            stop = min(df['Low'].iloc[-13:])
            signal = {
                'ticker': ticker, 'type': 'BUY', 'algo': 'COUNTDOWN 13',
                'price': price, 'perfected': perfected,
                'action': 'ACCUMULATE (Bottom)', 'timing': 'Trend Reversal (Weeks)',
                'stop': stop, 'target': price * 1.15, # 15% target for 13s
                'trend': 'Bullish Dip' if price > sma_200 else 'Counter-Trend'
            }
        elif last_row['Sell_Countdown'] == 13:
            perfected = last_row['Sell_13_Perfected']
            stop = max(df['High'].iloc[-13:])
            signal = {
                'ticker': ticker, 'type': 'SELL', 'algo': 'COUNTDOWN 13',
                'price': price, 'perfected': perfected,
                'action': 'DISTRIBUTE (Top)', 'timing': 'Trend Reversal (Weeks)',
                'stop': stop, 'target': price * 0.85, 
                'trend': 'Extension' if price > sma_200 else 'Bearish'
            }
            
        # --- 9 CHECK ---
        elif last_row['Buy_Setup'] == 9:
            l9, l8, l7, l6 = df['Low'].iloc[-1], df['Low'].iloc[-2], df['Low'].iloc[-3], df['Low'].iloc[-4]
            perfected = (l9 < l7 and l9 < l6) or (l8 < l7 and l8 < l6)
            stop = min(df['Low'].iloc[-9:])
            risk = price - stop
            if risk <= 0: risk = price * 0.01

            signal = {
                'ticker': ticker, 'type': 'BUY', 'algo': 'SETUP 9',
                'price': price, 'perfected': perfected,
                'action': 'SCALP LONG (Bounce)', 'timing': '1-4 Day Reaction',
                'stop': stop, 'target': price + (risk * 2),
                'trend': 'Bullish Dip' if price > sma_200 else 'Counter-Trend'
            }
            
        elif last_row['Sell_Setup'] == 9:
            h9, h8, h7, h6 = df['High'].iloc[-1], df['High'].iloc[-2], df['High'].iloc[-3], df['High'].iloc[-4]
            perfected = (h9 > h7 and h9 > h6) or (h8 > h7 and h8 > h6)
            stop = max(df['High'].iloc[-9:])
            risk = stop - price
            if risk <= 0: risk = price * 0.01

            signal = {
                'ticker': ticker, 'type': 'SELL', 'algo': 'SETUP 9',
                'price': price, 'perfected': perfected,
                'action': 'SCALP SHORT (Pullback)', 'timing': '1-4 Day Reaction',
                'stop': stop, 'target': price - (risk * 2),
                'trend': 'Bearish' if price < sma_200 else 'Extension'
            }
            
        return signal
    except Exception:
        return None

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    
    if len(message) > 4000:
        parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
        for part in parts:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": part, "parse_mode": "Markdown"})
    else:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                      json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Analyzing Macro Regime...")
    macro = get_market_regime()
    
    print("Fetching Tickers...")
    nasdaq = get_nasdaq_tickers()
    sp500 = get_sp500_tickers()
    full_universe = list(set(nasdaq + sp500 + STRATEGIC_TICKERS))
    print(f"Scanning {len(full_universe)} tickers...")
    
    signals = []
    
    for i, ticker in enumerate(full_universe):
        if i % 50 == 0: print(f"Processing {i}/{len(full_universe)}...")
        result = analyze_ticker(ticker)
        if result: signals.append(result)
        time.sleep(0.05)
    
    # --- BUILD REPORT ---
    msg = f"{macro['color']} **MARKET REGIME: {macro['regime']}**\n"
    msg += f"Score: {macro['score']}/100\n"
    for r in macro['reasons']: msg += f"• {r}\n"
    msg += "───────────────\n"
    
    if not signals:
        msg += "No DeMark signals found today."
        send_telegram_alert(msg)
    else:
        # Sort: 13s first, then 9s
        signals.sort(key=lambda x: (x['algo'] == 'SETUP 9', x['type']))
        
        for s in signals:
            icon = "🟢" if "BUY" in s['type'] else "🔴"
            perf_icon = "⭐" if s['perfected'] else "⚠️"
            
            msg += f"{icon} **{s['ticker']}** [{s['algo']}] {perf_icon}\n"
            msg += f"   📊 **Trend:** {s['trend']}\n"
            msg += f"   ⚡ **Action:** {s['action']}\n"
            msg += f"   💰 **Price:** ${s['price']:.2f}\n"
            msg += f"   🎯 **Target:** ${s['target']:.2f}\n"
            msg += f"   🛑 **Stop:** ${s['stop']:.2f}\n"
            msg += f"   ⏳ **Timing:** {s['timing']}\n"
            msg += "───────────────\n"

        msg += "\n_(⭐ = Perfected | ⚠️ = Unperfected)_"
        send_telegram_alert(msg)
    
    print("Done.")
