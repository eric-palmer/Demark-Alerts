import yfinance as yf
import pandas as pd
import requests
import os
import time

# --- CONFIGURATION ---
STRATEGIC_TICKERS = ['IBIT', 'ETHA', 'GLD', 'SLV', 'PALL', 'PPLT']

# --- FUNCTIONS ---

def get_sp500_tickers():
    """Scrapes S&P 500 tickers from Wikipedia"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Clean tickers (e.g., BRK.B -> BRK-B)
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"Error getting S&P 500: {e}")
        return []

def get_nasdaq_tickers():
    """Scrapes Nasdaq 100 tickers from Wikipedia"""
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables = pd.read_html(url)
        for table in tables:
            if 'Ticker' in table.columns:
                return table['Ticker'].tolist()
            if 'Symbol' in table.columns:
                return table['Symbol'].tolist()
        return []
    except Exception as e:
        print(f"Error getting Nasdaq 100: {e}")
        return []

def calculate_demark(df):
    """
    Calculates TD Sequential Setup (9).
    """
    # Create shift column (4 bars back)
    df['Close_4'] = df['Close'].shift(4)
    
    # Initialize lists to hold the counts
    buy_setups = [0] * len(df)
    sell_setups = [0] * len(df)
    
    closes = df['Close'].values
    closes_4 = df['Close_4'].values
    
    buy_seq = 0
    sell_seq = 0

    # Start loop from index 4 
    for i in range(4, len(df)):
        # --- BUY SETUP (Price < Price 4 bars ago) ---
        if closes[i] < closes_4[i]:
            buy_seq += 1
        else:
            buy_seq = 0
        buy_setups[i] = buy_seq
        
        # --- SELL SETUP (Price > Price 4 bars ago) ---
        if closes[i] > closes_4[i]:
            sell_seq += 1
        else:
            sell_seq = 0
        sell_setups[i] = sell_seq

    df['TD_Buy_Setup'] = buy_setups
    df['TD_Sell_Setup'] = sell_setups
    
    return df

def analyze_ticker(ticker):
    try:
        # Download data (6 months is sufficient for a 9 count)
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        
        # Data validation
        if len(df) < 20: 
            return None
        
        # Handling MultiIndex columns (yfinance update fix)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except:
                pass

        if 'Close' not in df.columns:
            return None

        # Calculate DeMark Indicators
        df = calculate_demark(df)
        
        last_row = df.iloc[-1]
        signal = None
        
        # --- BUY SIGNAL LOGIC (TD 9) ---
        if last_row['TD_Buy_Setup'] == 9:
            # Perfection Rule: Low of bar 8 or 9 < Low of bars 6 and 7
            l9 = df['Low'].iloc[-1]
            l8 = df['Low'].iloc[-2]
            l7 = df['Low'].iloc[-3]
            l6 = df['Low'].iloc[-4]
            
            is_perfected = (l9 < l7 and l9 < l6) or (l8 < l7 and l8 < l6)
            
            # TD Risk Level (Stop Loss): Lowest Low of the 9 bars
            setup_lows = df['Low'].iloc[-9:]
            stop_loss = min(setup_lows)
            
            signal = {
                'ticker': ticker,
                'type': 'BUY',
                'price': last_row['Close'],
                'perfected': is_perfected,
                'stop_loss': stop_loss
            }

        # --- SELL SIGNAL LOGIC (TD 9) ---
        elif last_row['TD_Sell_Setup'] == 9:
            # Perfection Rule: High of bar 8 or 9 > High of bars 6 and 7
            h9 = df['High'].iloc[-1]
            h8 = df['High'].iloc[-2]
            h7 = df['High'].iloc[-3]
            h6 = df['High'].iloc[-4]
            
            is_perfected = (h9 > h7 and h9 > h6) or (h8 > h7 and h8 > h6)
            
            # TD Risk Level (Stop Loss): Highest High of the 9 bars
            setup_highs = df['High'].iloc[-9:]
            stop_loss = max(setup_highs)
            
            signal = {
                'ticker': ticker,
                'type': 'SELL',
                'price': last_row['Close'],
                'perfected': is_perfected,
                'stop_loss': stop_loss
            }
            
        return signal

    except Exception:
        return None

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("Telegram credentials missing.")
        return
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Fetching Tickers...")
    nasdaq = get_nasdaq_tickers()
    sp500 = get_sp500_tickers()
    
    # Combine and ensure unique list
    full_universe = list(set(nasdaq + sp500 + STRATEGIC_TICKERS))
    print(f"Scanning {len(full_universe)} tickers...")
    
    buy_signals = []
    sell_signals = []
    
    for i, ticker in enumerate(full_universe):
        # Progress indicator for long lists
        if i % 50 == 0:
            print(f"Processing {i}/{len(full_universe)}...")
            
        result = analyze_ticker(ticker)
        
        if result:
            if result['type'] == 'BUY':
                buy_signals.append(result)
            else:
                sell_signals.append(result)
        
        # Small sleep to avoid Yahoo Finance rate limits (429 errors)
        time.sleep(0.05)
    
    # Build Output Message
    if not buy_signals and not sell_signals:
        print("No signals found today.")
    else:
        msg = "🔔 **DE-MARK HIGH VALUE ALERTS** 🔔\n\n"
        
        if buy_signals:
            msg += "🟢 **BUY SIGNALS (TD 9)**\n"
            for s in buy_signals:
                perf_icon = "✅" if s['perfected'] else "⚠️"
                msg += f"• **{s['ticker']}**: ${s['price']:.2f} {perf_icon}\n"
                msg += f"  └ Stop Loss: ${s['stop_loss']:.2f}\n"
            msg += "\n"
            
        if sell_signals:
            msg += "🔴 **SELL SIGNALS (TD 9)**\n"
            for s in sell_signals:
                perf_icon = "✅" if s['perfected'] else "⚠️"
                msg += f"• **{s['ticker']}**: ${s['price']:.2f} {perf_icon}\n"
                msg += f"  └ Stop Loss: ${s['stop_loss']:.2f}\n"
        
        msg += "\n_(✅ = Perfected Setup | ⚠️ = Unperfected)_"
        
        print("Sending Alert...")
        send_telegram_alert(msg)
        print("Done.")
