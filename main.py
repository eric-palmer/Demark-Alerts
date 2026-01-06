import yfinance as yf
import pandas as pd
import requests
import os
import time
import io  # Required for the fix

# --- CONFIGURATION ---
STRATEGIC_TICKERS = ['IBIT', 'ETHA', 'GLD', 'SLV', 'PALL', 'PPLT']

# --- FUNCTIONS ---

def get_sp500_tickers():
    """Scrapes S&P 500 tickers from Wikipedia using headers to bypass 403"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Fake a browser header to avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        response = requests.get(url, headers=headers)
        # Use io.StringIO to avoid pandas warnings/errors
        tables = pd.read_html(io.StringIO(response.text))
        
        df = tables[0]
        tickers = df['Symbol'].tolist()
        return [t.replace('.', '-') for t in tickers]
    except Exception as e:
        print(f"Error getting S&P 500: {e}")
        return []

def get_nasdaq_tickers():
    """Scrapes Nasdaq 100 tickers from Wikipedia using headers to bypass 403"""
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        
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
    """Calculates TD Sequential Setup (9)."""
    df['Close_4'] = df['Close'].shift(4)
    
    buy_setups = [0] * len(df)
    sell_setups = [0] * len(df)
    
    closes = df['Close'].values
    closes_4 = df['Close_4'].values
    
    buy_seq = 0
    sell_seq = 0

    for i in range(4, len(df)):
        if closes[i] < closes_4[i]:
            buy_seq += 1
        else:
            buy_seq = 0
        buy_setups[i] = buy_seq
        
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
        df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
        
        if len(df) < 20: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if 'Close' not in df.columns: return None

        df = calculate_demark(df)
        last_row = df.iloc[-1]
        signal = None
        
        # --- BUY SIGNAL (TD 9) ---
        if last_row['TD_Buy_Setup'] == 9:
            l9, l8, l7, l6 = df['Low'].iloc[-1], df['Low'].iloc[-2], df['Low'].iloc[-3], df['Low'].iloc[-4]
            is_perfected = (l9 < l7 and l9 < l6) or (l8 < l7 and l8 < l6)
            stop_loss = min(df['Low'].iloc[-9:])
            
            signal = {
                'ticker': ticker, 'type': 'BUY',
                'price': last_row['Close'], 'perfected': is_perfected, 'stop_loss': stop_loss
            }

        # --- SELL SIGNAL (TD 9) ---
        elif last_row['TD_Sell_Setup'] == 9:
            h9, h8, h7, h6 = df['High'].iloc[-1], df['High'].iloc[-2], df['High'].iloc[-3], df['High'].iloc[-4]
            is_perfected = (h9 > h7 and h9 > h6) or (h8 > h7 and h8 > h6)
            stop_loss = max(df['High'].iloc[-9:])
            
            signal = {
                'ticker': ticker, 'type': 'SELL',
                'price': last_row['Close'], 'perfected': is_perfected, 'stop_loss': stop_loss
            }
            
        return signal
    except Exception:
        return None

def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("❌ Error: Telegram credentials missing.")
        return
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
    except Exception as e:
        print(f"❌ Telegram Connection Error: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Fetching Tickers (Simulating Browser)...")
    nasdaq = get_nasdaq_tickers()
    sp500 = get_sp500_tickers()
    
    full_universe = list(set(nasdaq + sp500 + STRATEGIC_TICKERS))
    print(f"Scanning {len(full_universe)} tickers...")
    
    buy_signals = []
    sell_signals = []
    
    for i, ticker in enumerate(full_universe):
        if i % 50 == 0: print(f"Processing {i}/{len(full_universe)}...")
        
        result = analyze_ticker(ticker)
        if result:
            if result['type'] == 'BUY': buy_signals.append(result)
            else: sell_signals.append(result)
        
        time.sleep(0.05) # Rate limit protection
    
    if not buy_signals and not sell_signals:
        print("No signals found today.")
    else:
        msg = "🔔 **DE-MARK HIGH VALUE ALERTS** 🔔\n\n"
        if buy_signals:
            msg += "🟢 **BUY SIGNALS (TD 9)**\n"
            for s in buy_signals:
                icon = "✅" if s['perfected'] else "⚠️"
                msg += f"• **{s['ticker']}**: ${s['price']:.2f} {icon}\n"
        
        if sell_signals:
            msg += "🔴 **SELL SIGNALS (TD 9)**\n"
            for s in sell_signals:
                icon = "✅" if s['perfected'] else "⚠️"
                msg += f"• **{s['ticker']}**: ${s['price']:.2f} {icon}\n"
        
        msg += "\n_(✅ = Perfected | ⚠️ = Unperfected)_"
        print("Sending Alert...")
        send_telegram_alert(msg)
