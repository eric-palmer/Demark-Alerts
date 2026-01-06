import yfinance as yf
import pandas as pd
import requests
import os
import time
import io
import numpy as np

# --- CONFIGURATION ---
STRATEGIC_TICKERS = ['IBIT', 'ETHA', 'GLD', 'SLV', 'PALL', 'PPLT']

# --- DATA FEEDS (Crypto, F&G, Top 100) ---

def get_crypto_fear_greed():
    """Fetches Crypto Fear & Greed Index from Alternative.me"""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = r.json()['data'][0]
        return int(data['value']), data['value_classification']
    except:
        return None, "Unavailable"

def get_stock_fear_greed():
    """
    Attempts to fetch CNN Fear & Greed. 
    Fallbacks to a calculated proxy (VIX-based) if API fails.
    """
    try:
        # Try direct CNN data endpoint
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", headers=headers, timeout=5)
        data = r.json()
        score = int(data['fear_and_greed']['score'])
        rating = data['fear_and_greed']['rating']
        return score, rating.capitalize()
    except:
        # Proxy Fallback: Inverse VIX + SPY Momentum
        try:
            df = yf.download(["^VIX", "^GSPC"], period="1mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # Simple Proxy: High VIX = Fear, Low VIX = Greed
            vix = df['Close']['^VIX'].iloc[-1]
            if vix > 30: return 20, "Extreme Fear (Proxy)"
            elif vix > 20: return 40, "Fear (Proxy)"
            elif vix < 15: return 80, "Greed (Proxy)"
            else: return 50, "Neutral (Proxy)"
        except:
            return None, "Unavailable"

def get_top_cryptos():
    """Fetches Top 100 Cryptos by Market Cap from CoinGecko"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 100, 'page': 1}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        # Convert to yfinance format (e.g., BTC -> BTC-USD)
        tickers = [f"{coin['symbol'].upper()}-USD" for coin in data]
        return tickers
    except Exception as e:
        print(f"Error fetching Top Cryptos: {e}")
        return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'BNB-USD', 'DOGE-USD', 'ADA-USD'] # Fallback

def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        df = pd.read_html(io.StringIO(r.text))[0]
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except: return []

def get_nasdaq_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(r.text))
        for table in tables:
            if 'Ticker' in table.columns: return table['Ticker'].tolist()
            if 'Symbol' in table.columns: return table['Symbol'].tolist()
        return []
    except: return []

# --- MARKET RADAR REGIME ENGINE ---

def get_market_regime():
    """
    Builds a 'Risk Regime' dashboard using Liquidity, Credit, and Fear/Greed.
    """
    try:
        # 1. Fetch Fear & Greed Indices
        crypto_score, crypto_label = get_crypto_fear_greed()
        stock_score, stock_label = get_stock_fear_greed()
        
        # 2. Fetch Macro Data
        tickers = ['^GSPC', 'HYG', 'BTC-USD', 'DX-Y.NYB']
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex): data = data['Close']
        
        sma50 = data.rolling(window=50).mean().iloc[-1]
        sma200 = data.rolling(window=200).mean().iloc[-1]
        price = data.iloc[-1]
        
        score = 0
        reasons = []
        
        # A. LIQUIDITY (Bitcoin)
        if price['BTC-USD'] > sma50['BTC-USD']:
            score += 20
            reasons.append("🟢 Crypto Liquidity: EXPANDING")
        else:
            reasons.append("🔴 Crypto Liquidity: CONTRACTING")
            
        # B. CREDIT (HYG)
        if price['HYG'] > sma200['HYG']:
            score += 20
            reasons.append("🟢 Credit Markets: RISK ON")
        else:
            reasons.append("🔴 Credit Markets: STRESSED")
            
        # C. CURRENCY (DXY)
        if price['DX-Y.NYB'] < sma200['DX-Y.NYB']:
            score += 20
            reasons.append("🟢 Dollar: WEAK (Pro-Liquidity)")
        else:
            reasons.append("🔴 Dollar: STRONG (Tightening)")
        
        # D. SENTIMENT (Fear & Greed)
        # Extreme Fear (<25) is bullish for contrarians, Extreme Greed (>75) is bearish
        if stock_score and stock_score < 25:
            score += 20
            reasons.append(f"🟢 Stocks: {stock_label} (Buy Signal)")
        elif stock_score and stock_score > 75:
            reasons.append(f"🔴 Stocks: {stock_label} (Overheated)")
        else:
            score += 10
            reasons.append(f"🟡 Stocks: {stock_label}")

        # Final Regime Classification
        if score >= 60:
            regime = "RISK ON (Aggressive)"
            color = "🟢"
        elif score >= 40:
            regime = "NEUTRAL (Selective)"
            color = "🟡"
        else:
            regime = "RISK OFF (Defensive)"
            color = "🔴"
            
        return {
            "regime": regime, "color": color, "score": score, "reasons": reasons,
            "crypto_fg": f"{crypto_score} ({crypto_label})",
            "stock_fg": f"{stock_score} ({stock_label})"
        }
    except Exception as e:
        print(f"Macro Error: {e}")
        return None

# --- INSTITUTIONAL DEMARK ENGINE ---

def calculate_demark_indicators(df):
    """
    Strict TD Sequential Logic:
    - Setup 9: Consecutive closes < close[4]
    - Countdown 13: Non-consecutive closes <= low[2]
    - KEY FIX: New Setup 9 in same direction RESTARTS the Countdown (Sequential Rule).
    """
    # Create necessary columns first
    df['Close_4'] = df['Close'].shift(4)
    df['Buy_Setup'] = 0
    df['Sell_Setup'] = 0
    df['Buy_Countdown'] = 0
    df['Sell_Countdown'] = 0
    df['Buy_13_Perfected'] = False
    df['Sell_13_Perfected'] = False

    buy_seq = 0
    sell_seq = 0
    
    # State tracking
    buy_countdown = 0
    sell_countdown = 0
    active_buy_cd = False
    active_sell_cd = False
    
    # To check perfection, we need to track the bar index of the 8th count
    buy_cd_idx = [] 
    sell_cd_idx = []

    closes = df['Close'].values
    closes_4 = df['Close_4'].values
    lows = df['Low'].values
    highs = df['High'].values
    
    for i in range(4, len(df)):
        # --- 1. SETUP PHASE ---
        # Buy Setup
        if closes[i] < closes_4[i]:
            buy_seq += 1
        else:
            buy_seq = 0
        df.iloc[i, df.columns.get_loc('Buy_Setup')] = buy_seq
        
        # Sell Setup
        if closes[i] > closes_4[i]:
            sell_seq += 1
        else:
            sell_seq = 0
        df.iloc[i, df.columns.get_loc('Sell_Setup')] = sell_seq
        
        # --- 2. TRANSITION LOGIC (The "Reset" Fix) ---
        # If a NEW Buy Setup 9 completes, we must RESTART any existing Buy Countdown
        if buy_seq == 9:
            active_buy_cd = True
            buy_countdown = 0 # Sequential Reset
            buy_cd_idx = []
            active_sell_cd = False # Cancel opposite
            
        if sell_seq == 9:
            active_sell_cd = True
            sell_countdown = 0 # Sequential Reset
            sell_cd_idx = []
            active_buy_cd = False # Cancel opposite

        # --- 3. COUNTDOWN PHASE ---
        if active_buy_cd:
            if closes[i] <= lows[i-2]:
                buy_countdown += 1
                buy_cd_idx.append(i)
                df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_countdown
                
                if buy_countdown == 13:
                    # Perfection: Low of 13 <= Close of Bar 8
                    if len(buy_cd_idx) >= 8:
                        idx_8 = buy_cd_idx[7] # 8th bar (index 7)
                        if lows[i] <= closes[idx_8]:
                            df.iloc[i, df.columns.get_loc('Buy_13_Perfected')] = True
                    active_buy_cd = False # Finished

        if active_sell_cd:
            if closes[i] >= highs[i-2]:
                sell_countdown += 1
                sell_cd_idx.append(i)
                df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_countdown
                
                if sell_countdown == 13:
                    # Perfection: High of 13 >= Close of Bar 8
                    if len(sell_cd_idx) >= 8:
                        idx_8 = sell_cd_idx[7]
                        if highs[i] >= closes[idx_8]:
                            df.iloc[i, df.columns.get_loc('Sell_13_Perfected')] = True
                    active_sell_cd = False # Finished

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
        
        # --- 13 CHECK (Rare & Powerful) ---
        if last_row['Buy_Countdown'] == 13:
            perfected = last_row['Buy_13_Perfected']
            signal = {
                'ticker': ticker, 'type': 'BUY', 'algo': 'COUNTDOWN 13',
                'price': price, 'perfected': perfected,
                'action': 'ACCUMULATE (Major Bottom)', 'timing': 'Trend Reversal (Weeks)',
                'stop': min(df['Low'].iloc[-13:]), 'target': price * 1.15,
                'trend': 'Bullish Dip' if price > sma_200 else 'Counter-Trend'
            }
        elif last_row['Sell_Countdown'] == 13:
            perfected = last_row['Sell_13_Perfected']
            signal = {
                'ticker': ticker, 'type': 'SELL', 'algo': 'COUNTDOWN 13',
                'price': price, 'perfected': perfected,
                'action': 'DISTRIBUTE (Major Top)', 'timing': 'Trend Reversal (Weeks)',
                'stop': max(df['High'].iloc[-13:]), 'target': price * 0.85,
                'trend': 'Extension' if price > sma_200 else 'Bearish'
            }
            
        # --- 9 CHECK (Tactical) ---
        elif last_row['Buy_Setup'] == 9:
            # Perfection: Low of 8 or 9 < Low of 6 and 7
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
    
    # Split if too long
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
    print("Initializing Market Radar...")
    macro = get_market_regime()
    
    print("Fetching Tickers (Stocks + Top 100 Crypto)...")
    nasdaq = get_nasdaq_tickers()
    sp500 = get_sp500_tickers()
    top_crypto = get_top_cryptos()
    
    full_universe = list(set(nasdaq + sp500 + STRATEGIC_TICKERS + top_crypto))
    print(f"Scanning {len(full_universe)} tickers...")
    
    signals = []
    
    for i, ticker in enumerate(full_universe):
        if i % 50 == 0: print(f"Processing {i}/{len(full_universe)}...")
        result = analyze_ticker(ticker)
        if result: signals.append(result)
        time.sleep(0.05) # Rate limit
    
    # --- BUILD REPORT ---
    msg = f"{macro['color']} **MARKET REGIME: {macro['regime']}**\n"
    msg += f"Scores: Stocks F&G {macro['stock_fg']} | Crypto F&G {macro['crypto_fg']}\n"
    msg += f"Context: {macro['score']}/100 Liquidity Score\n"
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
            msg += f"   📊 {s['trend']}\n"
            msg += f"   ⚡ {s['action']}\n"
            msg += f"   💰 ${s['price']:.2f} | 🎯 ${s['target']:.2f} | 🛑 ${s['stop']:.2f}\n"
            msg += f"   ⏳ {s['timing']}\n"
            msg += "───────────────\n"

        msg += "\n_(⭐ = Perfected | ⚠️ = Unperfected)_"
        send_telegram_alert(msg)
    
    print("Done.")
