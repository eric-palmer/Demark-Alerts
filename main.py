import yfinance as yf
import pandas as pd
import pandas_datareader.data as web 
import requests
import os
import time
import numpy as np

# --- CONFIGURATION ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']

STRATEGIC_TICKERS = [
    # -- Meme / PolitiFi --
    'DJT', 'DOGE-USD', 'SHIB-USD', 'PEPE-USD',
    
    # -- Crypto: Coins --
    'BTC-USD', 'ETH-USD', 'SOL-USD',
    
    # -- Crypto: Miners & Infrastructure --
    'BTDR', 'MARA', 'RIOT', 'HUT', 'CLSK', 'IREN', 'CIFR', 'BTBT',
    'WYFI', 'CORZ', 'CRWV', 'APLD', 'NBIS', 'WULF', 'HIVE', 'BITF',
    'WGMI', 'MNRS', 'OWNB', 'BMNR', 'SBET', 'FWDI', 'BKKT',
    
    # -- Crypto: ETFs & Proxies --
    'IBIT', 'ETHA', 'BITQ', 'BSOL', 'GSOL', 'SOLT',
    'MSTR', 'COIN', 'HOOD', 'GLXY', 'STKE', 'DFDV', 'NODE', 'GEMI', 'BLSH',
    'CRCL',
    
    # -- Commodities (Futures & Proxies) --
    'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F', 'PA=F',
    'GLD', 'SLV', 'PALL', 'PPLT', 'NIKL', 'LIT', 'ILIT', 'REMX',
    
    # -- Energy / Grid / Uranium --
    'VOLT', 'GRID', 'EQT', 'TAC', 'BE', 'OKLO', 'SMR', 'NEE', 
    'URA', 'SRUUF', 'CCJ', 'KAMJY', 'UNL',
    
    # -- Tech / AI / Mag 7 --
    'NVDA', 'SMH', 'SMHX', 'TSM', 'AVGO', 'QCOM', 'MU', 'AMD', 'TER', 
    'NOW', 'AXON', 'SNOW', 'PLTR', 'GOOG', 'MSFT', 'META', 'AMZN', 'AAPL',
    'TSLA', 'NFLX', 'SPOT', 'SHOP', 'UBER', 'DASH', 'NET', 'DXCM', 'ETSY',
    'SQ', 'FIG', 'MAGS', 'MTUM', 'IVES',
    
    # -- Innovation / Fundstrat --
    'ARKK', 'ARKF', 'ARKG', 'GRNY', 'GRNI', 'GRNJ', 'XBI', 'XHB',
    
    # -- Sectors --
    'XLK', 'XLI', 'XLU', 'XLRE', 'XLB', 'XLV', 'XLF', 'XLE', 'XLP', 'XLY', 'XLC',
    
    # -- International --
    'BABA', 'JD', 'BIDU', 'PDD', 'XIACY', 'BYDDY', 'LKNCY', 'TCEHY',
    'MCHI', 'INDA', 'EWZ', 'EWJ', 'EWG', 'EWU', 'EWY', 'EWW', 'EWT', 'EWC', 'EEM',
    'AMX', 'PBR', 'VALE', 'NSRGY', 'DEO',
    
    # -- Financials / Bio / Other --
    'BLK', 'STT', 'ARES', 'SOFI', 'PYPL', 'IBKR', 'WU',
    'RXRX', 'SDGR', 'TEM', 'ABSI', 'DNA', 'TWST', 'GLW', 
    'KHC', 'LULU', 'YETI', 'DLR', 'EQIX', 'ORCL', 'LSF'
]

# --- ROBUST TELEGRAM SENDER ---
def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    
    # Chunking to prevent failures on long messages
    max_len = 4000
    for i in range(0, len(message), max_len):
        chunk = message[i:i+max_len]
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})
            time.sleep(1) # Rate limit protection
        except Exception as e:
            print(f"Telegram Fail: {e}")

def format_price(price):
    if price < 0.01: return f"${price:.6f}"
    if price < 1.00: return f"${price:.4f}"
    return f"${price:.2f}"

# --- DATA ENGINE (CRASH PROOF) ---
def safe_download(ticker, period="2y"):
    try:
        # User-Agent rotation to bypass 403 Forbidden
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
        
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, session=session)
        
        # Validation: Must have data, must have volume (avoid ghosts)
        if df.empty or len(df) < 50: return None
        
        # Handle MultiIndex columns if Yahoo returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if 'Close' not in df.columns: return None
        return df
    except:
        return None

def get_shared_macro_data():
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=730)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD', 'DGS10', 'DGS2', 'T5YIE'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        term_premia_proxy = fred['DGS10'] - fred['DGS2']
        spy = safe_download('SPY')
        if spy is None: return None
        
        return {'net_liq': net_liq, 'term_premia': term_premia_proxy, 'inflation': fred['T5YIE'], 'fed_assets': fred['WALCL'] / 1000, 'spy': spy['Close']}
    except: return None

# --- INDICATORS ---
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_squeeze(df):
    try:
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        upper_bb = sma + (std * 2); lower_bb = sma - (std * 2)
        
        tr = pd.DataFrame()
        tr['h-l'] = df['High'] - df['Low']
        tr['h-pc'] = abs(df['High'] - df['Close'].shift(1))
        tr['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
        atr = tr.max(axis=1).rolling(20).mean()
        
        upper_kc = sma + (atr * 1.5); lower_kc = sma - (atr * 1.5)
        
        is_squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        if not is_squeeze.iloc[-1]: return None
        
        # Bias
        y = df['Close'].iloc[-20:].values; x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        bias = "BULLISH 🟢" if slope > 0 else "BEARISH 🔴"
        return {'bias': bias, 'move': atr.iloc[-1] * 2}
    except: return None

def calc_fib(df):
    try:
        lookback = 126 # 6 months
        if len(df) < lookback: return None
        
        high = df['High'].iloc[-lookback:].max()
        low = df['Low'].iloc[-lookback:].min()
        price = df['Close'].iloc[-1]
        
        fibs = {
            0.382: high - (high-low)*0.382,
            0.618: high - (high-low)*0.618
        }
        
        # Check proximity (1%)
        for level, val in fibs.items():
            if abs(price - val)/price < 0.01:
                action = "BOUNCE" if price > val else "REJECTION"
                return {'level': f"{level*100}%", 'action': action, 'price': val}
        return None
    except: return None

def calc_demark(df):
    # Standard TD Sequential
    df = df.copy()
    df['Close_4'] = df['Close'].shift(4)
    df['Buy_Setup'] = 0; df['Sell_Setup'] = 0
    df['Buy_Countdown'] = 0; df['Sell_Countdown'] = 0
    
    buy_seq = 0; sell_seq = 0; buy_cd = 0; sell_cd = 0; active_buy = False; active_sell = False
    
    closes = df['Close'].values; closes_4 = df['Close_4'].values
    lows = df['Low'].values; highs = df['High'].values
    
    for i in range(4, len(df)):
        # Setup
        if closes[i] < closes_4[i]: buy_seq += 1; sell_seq = 0
        elif closes[i] > closes_4[i]: sell_seq += 1; buy_seq = 0
        else: buy_seq = 0; sell_seq = 0
        df.iloc[i, df.columns.get_loc('Buy_Setup')] = buy_seq
        df.iloc[i, df.columns.get_loc('Sell_Setup')] = sell_seq
        
        if buy_seq == 9: active_buy = True; buy_cd = 0; active_sell = False
        if sell_seq == 9: active_sell = True; sell_cd = 0; active_buy = False
        
        # Countdown
        if active_buy and closes[i] <= lows[i-2]:
            buy_cd += 1; df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_cd
            if buy_cd == 13: active_buy = False
        if active_sell and closes[i] >= highs[i-2]:
            sell_cd += 1; df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_cd
            if sell_cd == 13: active_sell = False
            
    return df

def analyze_ticker(ticker):
    try:
        df = safe_download(ticker)
        if df is None: return None
        
        # Weekly Data
        df_weekly = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        if len(df_weekly) < 20: return None
        
        # Calculate Indicators
        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        df_weekly = calc_demark(df_weekly)
        
        # Last Values
        last_d = df.iloc[-1]; last_w = df_weekly.iloc[-1]
        price = last_d['Close']
        
        # Squeeze & Fib
        d_sq = calc_squeeze(df); w_sq = calc_squeeze(df_weekly)
        fib = calc_fib(df)
        
        # Signals
        dm_sig = None; dm_perf = False
        
        # Daily DeMark
        if last_d['Buy_Countdown'] == 13: dm_sig = {'type': 'BUY 13', 'tf': 'Daily', 'target': price*1.15, 'stop': min(df['Low'].iloc[-13:])}
        elif last_d['Sell_Countdown'] == 13: dm_sig = {'type': 'SELL 13', 'tf': 'Daily', 'target': price*0.85, 'stop': max(df['High'].iloc[-13:])}
        elif last_d['Buy_Setup'] == 9: dm_sig = {'type': 'BUY 9', 'tf': 'Daily', 'target': price*1.05, 'stop': min(df['Low'].iloc[-9:])}
        elif last_d['Sell_Setup'] == 9: dm_sig = {'type': 'SELL 9', 'tf': 'Daily', 'target': price*0.95, 'stop': max(df['High'].iloc[-9:])}
        
        # Weekly Overwrite
        if last_w['Buy_Countdown'] == 13: dm_sig = {'type': 'BUY 13', 'tf': 'Weekly', 'target': price*1.30, 'stop': min(df_weekly['Low'].iloc[-13:])}
        elif last_w['Sell_Countdown'] == 13: dm_sig = {'type': 'SELL 13', 'tf': 'Weekly', 'target': price*0.70, 'stop': max(df_weekly['High'].iloc[-13:])}
        elif last_w['Buy_Setup'] == 9: dm_sig = {'type': 'BUY 9', 'tf': 'Weekly', 'target': price*1.10, 'stop': min(df_weekly['Low'].iloc[-9:])}
        elif last_w['Sell_Setup'] == 9: dm_sig = {'type': 'SELL 9', 'tf': 'Weekly', 'target': price*0.90, 'stop': max(df_weekly['High'].iloc[-9:])}
        
        # Perfection Logic
        if dm_sig:
            if '13' in dm_sig['type']: dm_perf = True
            elif 'BUY 9' in dm_sig['type']: dm_perf = (df['Low'].iloc[-1] < df['Low'].iloc[-3] and df['Low'].iloc[-1] < df['Low'].iloc[-4])
            elif 'SELL 9' in dm_sig['type']: dm_perf = (df['High'].iloc[-1] > df['High'].iloc[-3] and df['High'].iloc[-1] > df['High'].iloc[-4])
            dm_sig['perfected'] = dm_perf

        # RSI Logic
        rsi_sig = None
        if last_d['RSI'] < 30: rsi_sig = {'type': 'OVERSOLD', 'val': last_d['RSI']}
        elif last_d['RSI'] > 70: rsi_sig = {'type': 'OVERBOUGHT', 'val': last_d['RSI']}
        
        # Squeeze Logic
        sq_sig = None
        if d_sq: sq_sig = {'tf': 'Daily', 'bias': d_sq['bias'], 'move': d_sq['move']}
        elif w_sq: sq_sig = {'tf': 'Weekly', 'bias': w_sq['bias'], 'move': w_sq['move']}
        
        return {
            'ticker': ticker, 'price': price,
            'demark': dm_sig, 'rsi': rsi_sig, 'squeeze': sq_sig, 'fib': fib,
            'rsi_val': last_d['RSI'], 'count_buy': last_d['Buy_Setup'], 'count_sell': last_d['Sell_Setup']
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("1. Generating Macro...")
    macro = get_shared_macro_data()
    msg = "🌍 **GLOBAL MACRO REPORT** 🌍\n\n"
    
    if macro:
        # Radar
        liq_momo = macro['net_liq'].pct_change(63).iloc[-1]
        spy_momo = macro['spy'].pct_change(126).iloc[-1]
        regime = "RISK ON 🟢" if (liq_momo > 0 and spy_momo > 0) else "RISK OFF 🔴"
        msg += f"📊 **MARKET RADAR**: {regime}\n   └ Liq: {liq_momo*100:.2f}% | Growth: {spy_momo*100:.2f}%\n\n"
        
        # Howell
        roc_med = macro['net_liq'].pct_change(63).iloc[-1]
        accel = macro['net_liq'].pct_change(63).diff(20).iloc[-1]
        if roc_med > 0:
            phase = "REBOUND (Early Cycle)" if accel > 0 else "SPECULATION (Late Cycle)"
        else:
            phase = "TURBULENCE (Contraction)" if accel < 0 else "CALM (Bottoming)"
        msg += f"🏛️ **CAPITAL WARS**: {phase}\n\n"
    else:
        msg += "⚠️ Macro Data Unavailable\n\n"
        
    send_telegram_alert(msg)
    
    # --- PORTFOLIO ---
    print("2. Portfolio Analysis...")
    p_msg = "💼 **PORTFOLIO HEALTH** 💼\n"
    for t in CURRENT_PORTFOLIO:
        res = analyze_ticker(t)
        if res:
            p = format_price(res['price'])
            count = f"Buy {res['count_buy']}" if res['count_buy'] > 0 else f"Sell {res['count_sell']}"
            p_msg += f"🔹 **{t}**: {p} | RSI: {res['rsi_val']:.0f}\n   └ Count: {count}\n"
            if res['demark']: p_msg += f"   🚨 SIGNAL: {res['demark']['type']}\n"
            if res['squeeze']: p_msg += f"   ⚠️ SQUEEZE: {res['squeeze']['tf']} ({res['squeeze']['bias']})\n"
            if res['fib']: p_msg += f"   🕸️ FIB: {res['fib']['action']} @ {res['fib']['level']}\n"
            p_msg += "\n"
    send_telegram_alert(p_msg)
    
    # --- MARKET SCAN ---
    print("3. Scanning Universe...")
    # Dynamic list building
    universe = list(set(STRATEGIC_TICKERS + get_top_200_cryptos() + get_top_futures() + get_sp500_tickers() + get_nasdaq_tickers()))
    
    power = []; perfected = []; unperf = []; rsi_List = []; sq_list = []
    
    for t in universe:
        res = analyze_ticker(t)
        if not res: continue
        
        d = res['demark']
        # Power Logic
        score = 0
        if d and d['perfected']: score += 1
        if res['rsi']: score += 1
        if res['squeeze']: score += 1
        if res['fib']: score += 1
        
        if score >= 2 and d and d['perfected']: power.append(res)
        
        # Categorize
        if d:
            if d['perfected']: perfected.append(res)
            else: unperf.append(res)
        if res['rsi']: rsi_List.append(res)
        if res['squeeze']: sq_list.append(res)
        
        time.sleep(0.01) # Be nice to Yahoo
        
    # --- ALERTS ---
    a_msg = "🔔 **MARKET ALERTS** 🔔\n\n"
    
    if power:
        a_msg += "🔥 **POWER RANKINGS (High Conviction)** 🔥\n"
        for s in power[:10]:
            d = s['demark']
            a_msg += f"🚀 **{s['ticker']}**: {format_price(s['price'])}\n   └ {d['type']} ({d['tf']}) ✅\n   └ Target: {format_price(d['target'])}\n"
            if s['fib']: a_msg += f"   └ Fib: {s['fib']['action']}\n"
            a_msg += "──────────\n"
            
    if perfected:
        a_msg += "\n✅ **PERFECTED DEMARK**\n"
        perfected.sort(key=lambda x: '13' in x['demark']['type'], reverse=True)
        for s in perfected[:15]:
            if s in power: continue
            d = s['demark']
            a_msg += f"🟢 **{s['ticker']}**: {d['type']} ({d['tf']}) @ {format_price(s['price'])}\n"

    if unperf:
        a_msg += "\n⚠️ **WATCHLIST (Unperfected)**\n"
        unperf.sort(key=lambda x: '13' in x['demark']['type'], reverse=True)
        for s in unperf[:10]:
            d = s['demark']
            a_msg += f"⚪ **{s['ticker']}**: {d['type']} ({d['tf']})\n"
            
    if sq_list:
        a_msg += "\n💥 **VOLATILITY SQUEEZES**\n"
        sq_list.sort(key=lambda x: x['squeeze']['tf'] == 'Weekly', reverse=True)
        for s in sq_list[:10]:
            if s in power: continue
            sq = s['squeeze']
            a_msg += f"⚠️ **{s['ticker']}**: {sq['tf']} ({sq['bias']})\n"

    send_telegram_alert(a_msg)
    print("Done.")
