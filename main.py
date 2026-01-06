import yfinance as yf
import pandas as pd
import requests
import os
import time
import io

# --- CONFIGURATION ---
STRATEGIC_TICKERS = ['IBIT', 'ETHA', 'GLD', 'SLV', 'PALL', 'PPLT']

# --- FUNCTIONS ---

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

def calculate_demark_indicators(df):
    """
    Advanced Logic: Calculates both Setup (9) and Countdown (13).
    """
    # 1. Setup Phase
    df['Close_4'] = df['Close'].shift(4)
    df['Buy_Setup'] = 0
    df['Sell_Setup'] = 0
    
    # 2. Countdown Phase
    df['Close_2'] = df['Close'].shift(2)
    df['Buy_Countdown'] = 0
    df['Sell_Countdown'] = 0

    buy_seq = 0
    sell_seq = 0
    
    # Countdown state trackers
    active_buy_countdown = False
    buy_count = 0
    
    active_sell_countdown = False
    sell_count = 0
    
    # We need to iterate to maintain state for the Countdown
    # (Countdown 13 can only start AFTER a Setup 9 completes)
    for i in range(4, len(df)):
        # --- SETUP LOGIC (9) ---
        # Buy Setup
        if df['Close'].iloc[i] < df['Close_4'].iloc[i]:
            buy_seq += 1
        else:
            buy_seq = 0
        df.iloc[i, df.columns.get_loc('Buy_Setup')] = buy_seq
        
        # Sell Setup
        if df['Close'].iloc[i] > df['Close_4'].iloc[i]:
            sell_seq += 1
        else:
            sell_seq = 0
        df.iloc[i, df.columns.get_loc('Sell_Setup')] = sell_seq
        
        # --- STATE MANAGEMENT ---
        # If Buy Setup completes, activate Buy Countdown / Cancel Sell Countdown
        if buy_seq == 9:
            active_buy_countdown = True
            buy_count = 0 # Reset count
            active_sell_countdown = False # Trend flipped
            
        # If Sell Setup completes, activate Sell Countdown / Cancel Buy Countdown
        if sell_seq == 9:
            active_sell_countdown = True
            sell_count = 0 # Reset count
            active_buy_countdown = False # Trend flipped

        # --- COUNTDOWN LOGIC (13) ---
        # Buy Countdown (Close <= Low 2 bars ago)
        if active_buy_countdown:
            # Note: DeMark standard version often waits for bar 9 of setup to finish before counting 1
            # Here we count subsequent bars.
            if df['Close'].iloc[i] <= df['Low'].iloc[i-2]:
                buy_count += 1
                df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_count
                if buy_count == 13:
                    active_buy_countdown = False # Reset after completion

        # Sell Countdown (Close >= High 2 bars ago)
        if active_sell_countdown:
            if df['Close'].iloc[i] >= df['High'].iloc[i-2]:
                sell_count += 1
                df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_count
                if sell_count == 13:
                    active_sell_countdown = False # Reset after completion

    return df

def analyze_ticker(ticker):
    try:
        # Download 2 years to allow for 13 counts to develop
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        
        if len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if 'Close' not in df.columns: return None

        # Add 200 SMA for Trend Context
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        df = calculate_demark_indicators(df)
        last_row = df.iloc[-1]
        price = last_row['Close']
        sma_200 = last_row['SMA_200']
        
        signal = None
        
        # --- DETECT 13 (MAJOR REVERSAL) ---
        if last_row['Buy_Countdown'] == 13:
            risk = price * 0.05 # Estimate risk on 13s (wider)
            signal = {
                'ticker': ticker, 'type': 'BUY_13', 'price': price,
                'desc': 'MAJOR BOTTOM', 'action': 'ACCUMULATE',
                'stop': min(df['Low'].iloc[-13:]), 'target': price + (price*0.10), # 10% move target
                'trend': 'ABOVE 200SMA (Bullish Dip)' if price > sma_200 else 'BELOW 200SMA (Counter-Trend)'
            }
        elif last_row['Sell_Countdown'] == 13:
            risk = price * 0.05
            signal = {
                'ticker': ticker, 'type': 'SELL_13', 'price': price,
                'desc': 'MAJOR TOP', 'action': 'DISTRIBUTE/SHORT',
                'stop': max(df['High'].iloc[-13:]), 'target': price - (price*0.10),
                'trend': 'ABOVE 200SMA (Extension)' if price > sma_200 else 'BELOW 200SMA (Bearish)'
            }
            
        # --- DETECT 9 (REACTION) ---
        # Only alert 9 if there isn't a 13 (13 takes priority)
        elif last_row['Buy_Setup'] == 9:
            # Perfection Logic
            l9, l8, l7, l6 = df['Low'].iloc[-1], df['Low'].iloc[-2], df['Low'].iloc[-3], df['Low'].iloc[-4]
            is_perfected = (l9 < l7 and l9 < l6) or (l8 < l7 and l8 < l6)
            
            stop_loss = min(df['Low'].iloc[-9:])
            risk = price - stop_loss
            if risk == 0: risk = price * 0.01 # Prevent div/0
            
            signal = {
                'ticker': ticker, 'type': 'BUY_9', 'price': price,
                'desc': 'EXHAUSTION', 'action': 'SCALP LONG',
                'stop': stop_loss, 'target': price + (risk * 2.0), # 2.0R Target
                'perfected': is_perfected,
                'trend': 'ABOVE 200SMA (Dip Buy)' if price > sma_200 else 'BELOW 200SMA (Reversal)'
            }
            
        elif last_row['Sell_Setup'] == 9:
            h9, h8, h7, h6 = df['High'].iloc[-1], df['High'].iloc[-2], df['High'].iloc[-3], df['High'].iloc[-4]
            is_perfected = (h9 > h7 and h9 > h6) or (h8 > h7 and h8 > h6)
            
            stop_loss = max(df['High'].iloc[-9:])
            risk = stop_loss - price
            if risk == 0: risk = price * 0.01

            signal = {
                'ticker': ticker, 'type': 'SELL_9', 'price': price,
                'desc': 'EXHAUSTION', 'action': 'SCALP SHORT',
                'stop': stop_loss, 'target': price - (risk * 2.0),
                'perfected': is_perfected,
                'trend': 'ABOVE 200SMA (Reversal)' if price > sma_200 else 'BELOW 200SMA (Rally Sell)'
            }
            
        return signal
    except Exception:
        return None

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    
    # Split message if too long for Telegram (4096 chars)
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
    print("Fetching Tickers...")
    nasdaq = get_nasdaq_tickers()
    sp500 = get_sp500_tickers()
    full_universe = list(set(nasdaq + sp500 + STRATEGIC_TICKERS))
    print(f"Scanning {len(full_universe)} tickers...")
    
    signals_9 = []
    signals_13 = []
    
    for i, ticker in enumerate(full_universe):
        if i % 50 == 0: print(f"Processing {i}/{len(full_universe)}...")
        result = analyze_ticker(ticker)
        if result:
            if '13' in result['type']:
                signals_13.append(result)
            else:
                signals_9.append(result)
        time.sleep(0.05)
    
    if not signals_9 and not signals_13:
        print("No signals found.")
    else:
        # --- INSTITUTIONAL REPORT ---
        msg = "🏦 **INSTITUTIONAL DEMARK REPORT** 🏦\n"
        
        # SECTION 1: THE BIG TRADES (13s)
        if signals_13:
            msg += "\n🚨 **MAJOR REVERSALS (COUNTDOWN 13)** 🚨\n"
            msg += "_Timing: Weeks/Months | High Conviction_\n"
            for s in signals_13:
                icon = "🟢" if "BUY" in s['type'] else "🔴"
                msg += f"───────────────\n"
                msg += f"{icon} **{s['ticker']}**: ${s['price']:.2f}\n"
                msg += f"📊 **Trend:** {s['trend']}\n"
                msg += f"⚡ **Action:** {s['action']}\n"
                msg += f"🎯 **Target:** ${s['target']:.2f}\n"
                msg += f"🛑 **Stop:** ${s['stop']:.2f}\n"

        # SECTION 2: THE SCALPS (9s)
        if signals_9:
            msg += "\n⚡ **TACTICAL SETUPS (SETUP 9)** ⚡\n"
            msg += "_Timing: 1-4 Days | Short Term Reaction_\n"
            for s in signals_9:
                icon = "🟢" if "BUY" in s['type'] else "🔴"
                perf = "⭐" if s['perfected'] else "⚠️" # Star for perfect
                msg += f"{icon} **{s['ticker']}** ({perf}): ${s['price']:.2f}\n"
                msg += f"   └ {s['trend']}\n"
                msg += f"   └ 🎯 ${s['target']:.2f} | 🛑 ${s['stop']:.2f}\n"

        msg += "\n_(⭐ = Perfected 9 | ⚠️ = Unperfected)_"
        
        print("Sending Alert...")
        send_telegram_alert(msg)
