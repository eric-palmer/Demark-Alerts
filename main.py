import yfinance as yf
import pandas as pd
import pandas_datareader.data as web 
import requests
import os
import sys
import time
import datetime
import numpy as np

print(f"🚀 INITIALIZING INSTITUTIONAL SCANNER | Python: {sys.version}")

# --- CONFIGURATION ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']

STRATEGIC_TICKERS = [
    # -- Meme / PolitiFi --
    'DJT', 'DOGE-USD', 'SHIB-USD', 'PEPE-USD',
    
    # -- Crypto (Yahoo Tickers) --
    'BTC-USD', 'ETH-USD', 'SOL-USD',
    
    # -- Miners & Proxies --
    'BTDR', 'MARA', 'RIOT', 'HUT', 'CLSK', 'IREN', 'CIFR', 'BTBT',
    'WYFI', 'CORZ', 'CRWV', 'APLD', 'NBIS', 'WULF', 'HIVE', 'BITF',
    'MSTR', 'COIN', 'HOOD', 'GLXY', 'STKE', 'DFDV', 'NODE', 'GEMI', 'BLSH',
    
    # -- ETFs --
    'IBIT', 'ETHA', 'BITQ', 'BSOL', 'GSOL', 'SOLT',
    
    # -- Commodities --
    'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F', 'PA=F',
    'GLD', 'SLV', 'PALL', 'PPLT', 'NIKL', 'LIT', 'ILIT', 'REMX',
    
    # -- Energy --
    'VOLT', 'GRID', 'EQT', 'TAC', 'BE', 'OKLO', 'SMR', 'NEE', 
    'URA', 'SRUUF', 'CCJ', 'KAMJY', 'UNL',
    
    # -- Tech / Mag 7 --
    'NVDA', 'SMH', 'SMHX', 'TSM', 'AVGO', 'QCOM', 'MU', 'AMD', 'TER', 
    'NOW', 'AXON', 'SNOW', 'PLTR', 'GOOG', 'MSFT', 'META', 'AMZN', 'AAPL',
    'TSLA', 'NFLX', 'SPOT', 'SHOP', 'UBER', 'DASH', 'NET', 'DXCM', 'ETSY',
    'SQ', 'MAGS', 'MTUM', 'IVES',
    
    # -- Sectors --
    'XLK', 'XLI', 'XLU', 'XLRE', 'XLB', 'XLV', 'XLF', 'XLE', 'XLP', 'XLY', 'XLC',
    
    # -- Financials / Other --
    'BLK', 'STT', 'ARES', 'SOFI', 'PYPL', 'IBKR', 'WU',
    'RXRX', 'SDGR', 'TEM', 'ABSI', 'DNA', 'TWST', 'GLW', 
    'KHC', 'LULU', 'YETI', 'DLR', 'EQIX', 'ORCL', 'LSF'
]

# --- HELPER FUNCTIONS ---
def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    
    max_len = 4000
    for i in range(0, len(message), max_len):
        chunk = message[i:i+max_len]
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})
            time.sleep(1)
        except Exception as e: 
            print(f"Telegram Error: {e}")

def format_price(price):
    if price is None: return "N/A"
    if price < 0.01: return f"${price:.6f}"
    if price < 1.00: return f"${price:.4f}"
    return f"${price:.2f}"

# --- DATA FETCHERS (ANTI-BLOCK METHOD) ---
def safe_download(ticker):
    try:
        # METHOD CHANGE: Use Ticker().history() instead of download()
        # This mimics a single user session better than the bulk downloader
        dat = yf.Ticker(ticker)
        df = dat.history(period="2y", auto_adjust=True)
        
        if df.empty or len(df) < 50: 
            # Try one fallback with standard download if history fails
            df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        
        if df.empty: return None, "Empty Data"
        
        # Clean columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df, None
    except Exception as e: 
        return None, str(e)

def get_top_futures():
    return ['ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F', 'FTSE=F', 'CL=F', 'NG=F', 'RB=F', 'HO=F', 'BZ=F', 'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'ZT=F', 'ZF=F', 'ZN=F', 'ZB=F', 'DX-Y.NYB', 'ZC=F', 'ZS=F', 'CC=F', 'KC=F', 'SB=F']

def get_shared_macro_data():
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=730)
        # FRED is notoriously flaky on cloud IPs. We wrap it tightly.
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD', 'DGS10', 'DGS2', 'T5YIE'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        term_premia = fred['DGS10'] - fred['DGS2']
        
        # Use new safe download for SPY
        spy, err = safe_download('SPY')
        if spy is None: return None
        
        return {'net_liq': net_liq, 'term_premia': term_premia, 'inflation': fred['T5YIE'], 'spy': spy['Close']}
    except:
        return None

# --- INDICATORS ---
def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except: return pd.Series([50]*len(series))

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
        
        y = df['Close'].iloc[-20:].values; x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        bias = "BULLISH 🟢" if slope > 0 else "BEARISH 🔴"
        return {'bias': bias, 'move': atr.iloc[-1] * 2}
    except: return None

def calc_fib(df):
    try:
        lookback = 126
        if len(df) < lookback: return None
        high = df['High'].iloc[-lookback:].max(); low = df['Low'].iloc[-lookback:].min()
        price = df['Close'].iloc[-1]
        fibs = {0.382: high - (high-low)*0.382, 0.618: high - (high-low)*0.618, 0.5: high - (high-low)*0.5}
        for level, val in fibs.items():
            if abs(price - val)/price < 0.015:
                action = "SUPPORT" if price > val else "RESISTANCE"
                return {'level': f"{level*100}%", 'action': action, 'price': val}
        return None
    except: return None

def calc_demark(df):
    try:
        df = df.copy()
        df['Close_4'] = df['Close'].shift(4)
        df['Buy_Setup'] = 0; df['Sell_Setup'] = 0; df['Buy_Countdown'] = 0; df['Sell_Countdown'] = 0
        buy_seq = 0; sell_seq = 0; buy_cd = 0; sell_cd = 0; active_buy = False; active_sell = False
        
        closes = df['Close'].values; closes_4 = df['Close_4'].values
        lows = df['Low'].values; highs = df['High'].values
        
        for i in range(4, len(df)):
            if closes[i] < closes_4[i]: buy_seq += 1; sell_seq = 0
            elif closes[i] > closes_4[i]: sell_seq += 1; buy_seq = 0
            else: buy_seq = 0; sell_seq = 0
            df.iloc[i, df.columns.get_loc('Buy_Setup')] = buy_seq
            df.iloc[i, df.columns.get_loc('Sell_Setup')] = sell_seq
            
            if buy_seq == 9: active_buy = True; buy_cd = 0; active_sell = False
            if sell_seq == 9: active_sell = True; sell_cd = 0; active_buy = False
            
            if active_buy and closes[i] <= lows[i-2]:
                buy_cd += 1; df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_cd
                if buy_cd == 13: active_buy = False
            if active_sell and closes[i] >= highs[i-2]:
                sell_cd += 1; df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_cd
                if sell_cd == 13: active_sell = False
        return df
    except: return df

def analyze_ticker(ticker, is_portfolio=False):
    try:
        df, err = safe_download(ticker)
        
        # ERROR HANDLING: Pass the error message back
        if df is None: return {'error': err}
        
        if not is_portfolio and (df['Volume'].iloc[-5:].sum() == 0 or df['Close'].iloc[-1] < 0.00000001): return None
        
        df_weekly = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        if len(df_weekly) < 20: return None
        
        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        df_weekly = calc_demark(df_weekly)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        d_sq = calc_squeeze(df); w_sq = calc_squeeze(df_weekly); fib = calc_fib(df)
        last_d = df.iloc[-1]; last_w = df_weekly.iloc[-1]; price = last_d['Close']
        
        # Targets
        atr = df['High'].sub(df['Low']).rolling(14).mean().iloc[-1]
        if np.isnan(atr): atr = price * 0.02
        p_target = price + (atr * 2) if price > df['Close'].rolling(50).mean().iloc[-1] else price - (atr * 2)
        p_stop = price - (atr * 1.5) if price > df['Close'].rolling(50).mean().iloc[-1] else price + (atr * 1.5)
        
        # Signals
        dm_sig = None; dm_perf = False
        if last_d['Buy_Countdown'] == 13: dm_sig = {'type': 'BUY 13', 'tf': 'Daily'}
        elif last_d['Sell_Countdown'] == 13: dm_sig = {'type': 'SELL 13', 'tf': 'Daily'}
        elif last_d['Buy_Setup'] == 9: dm_sig = {'type': 'BUY 9', 'tf': 'Daily'}
        elif last_d['Sell_Setup'] == 9: dm_sig = {'type': 'SELL 9', 'tf': 'Daily'}
        elif last_w['Buy_Countdown'] == 13: dm_sig = {'type': 'BUY 13', 'tf': 'Weekly'}
        elif last_w['Sell_Countdown'] == 13: dm_sig = {'type': 'SELL 13', 'tf': 'Weekly'}
        
        if dm_sig:
            if '13' in dm_sig['type']: dm_perf = True
            elif 'BUY' in dm_sig['type']: dm_perf = (df['Low'].iloc[-1] < df['Low'].iloc[-3] and df['Low'].iloc[-1] < df['Low'].iloc[-4])
            elif 'SELL' in dm_sig['type']: dm_perf = (df['High'].iloc[-1] > df['High'].iloc[-3] and df['High'].iloc[-1] > df['High'].iloc[-4])
            dm_sig['perfected'] = dm_perf

        rsi_sig = None
        if last_d['RSI'] < 30: rsi_sig = {'type': 'OVERSOLD', 'val': last_d['RSI']}
        elif last_d['RSI'] > 70: rsi_sig = {'type': 'OVERBOUGHT', 'val': last_d['RSI']}
        
        sq_sig = None
        if d_sq: sq_sig = {'tf': 'Daily', 'bias': d_sq['bias'], 'move': d_sq['move']}
        elif w_sq: sq_sig = {'tf': 'Weekly', 'bias': w_sq['bias'], 'move': w_sq['move']}
        
        trend = "BULLISH" if price > df['Close'].rolling(200).mean().iloc[-1] else "BEARISH"
        macd_val = "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"
        
        verdict = "HOLD"
        timing = "N/A"
        if dm_sig and "BUY" in dm_sig['type']: verdict = "BUY (DeMark)"; timing = "1-4 Weeks"
        elif dm_sig and "SELL" in dm_sig['type']: verdict = "SELL (DeMark)"; timing = "1-4 Weeks"
        elif trend == "BULLISH" and macd_val == "Bullish": verdict = "BUY (Trend)"; timing = "Trend"
        elif trend == "BEARISH" and macd_val == "Bearish": verdict = "SELL (Trend)"; timing = "Trend"

        return {
            'ticker': ticker, 'price': price,
            'demark': dm_sig, 'rsi': rsi_sig, 'squeeze': sq_sig, 'fib': fib,
            'trend': trend, 'verdict': verdict, 'target': p_target, 'stop': p_stop,
            'rsi_val': last_d['RSI'], 'macd': macd_val, 'timing': timing,
            'count': f"Buy {last_d['Buy_Setup']}" if last_d['Buy_Setup'] > 0 else f"Sell {last_d['Sell_Setup']}"
        }
    except: return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("1. Generating Macro...")
    macro = get_shared_macro_data()
    msg = "🌍 **GLOBAL MACRO REPORT** 🌍\n\n"
    
    if macro:
        liq_momo = macro['net_liq'].pct_change(63).iloc[-1]
        spy_momo = macro['spy'].pct_change(126).iloc[-1]
        regime = "RISK ON 🟢" if (liq_momo > 0 and spy_momo > 0) else "RISK OFF 🔴"
        msg += f"📊 **MARKET RADAR**: {regime}\n   └ Liq: {liq_momo*100:.2f}% | Growth: {spy_momo*100:.2f}%\n\n"
    else:
        # Reduced verbosity on failure
        msg += "⚠️ Macro Data Unavailable (Feed Blocked)\n\n"
        
    send_telegram_alert(msg)
    
    # --- PORTFOLIO ---
    print("2. Portfolio Analysis...")
    p_msg = "💼 **CURRENT PORTFOLIO** 💼\n"
    for t in CURRENT_PORTFOLIO:
        res = analyze_ticker(t, is_portfolio=True)
        if res and 'error' not in res:
            p = format_price(res['price']); tgt = format_price(res['target']); stp = format_price(res['stop'])
            p_msg += f"🔹 **{t}**: {res['verdict']} @ {p}\n   └ 🎯 Target: {tgt} | 🛑 Stop: {stp}\n   └ ⏳ Timing: {res['timing']}\n   └ 📊 Techs: {res['trend']} | RSI: {res['rsi_val']:.0f} | {res['count']}\n"
            if res['demark']: p_msg += f"   🚨 DEMARK: {res['demark']['type']} ({'Perf' if res['demark'].get('perfected') else 'Unperf'})\n"
            if res['fib']: p_msg += f"   🕸️ FIB: {res['fib']['action']} @ {res['fib']['level']}\n"
            if res['squeeze']: p_msg += f"   💥 SQUEEZE: {res['squeeze']['tf']} ({res['squeeze']['bias']})\n"
            p_msg += "\n"
        else:
             # PRINT THE EXACT ERROR FOR DEBUGGING
             err_msg = res['error'] if res else "Unknown Error"
             p_msg += f"🔹 **{t}**: ⚠️ DATA ERROR: {err_msg}\n\n"
             
    send_telegram_alert(p_msg)
    
    # --- MARKET SCAN ---
    print("3. Scanning Universe...")
    universe = list(set(STRATEGIC_TICKERS + get_top_futures()))
    power = []; perfected = []; unperf = []; sq_list = []; fib_list = []
    
    for t in universe:
        res = analyze_ticker(t)
        if not res or 'error' in res: continue
        
        d = res['demark']
        score = 0
        if d and d.get('perfected'): score += 1
        if res['rsi']: score += 1
        if res['squeeze']: score += 1
        if res['fib']: score += 1
        
        if score >= 2 and d and d.get('perfected'): power.append(res)
        if d:
            if d.get('perfected'): perfected.append(res)
            else: unperf.append(res)
        if res['squeeze']: sq_list.append(res)
        if res['fib']: fib_list.append(res)
        time.sleep(0.1)
        
    # --- ALERTS ---
    a_msg = "🔔 **MARKET ALERTS** 🔔\n\n"
    if power:
        a_msg += "🔥 **POWER RANKINGS (High Conviction)** 🔥\n"
        for s in power[:10]:
            d = s['demark']
            a_msg += f"🚀 **{s['ticker']}**: {format_price(s['price'])}\n   └ {d['type']} ({d['tf']}) ✅\n   └ Target: {format_price(s['target'])}\n──────────\n"
            
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
            if s in power: continue
            d = s['demark']
            a_msg += f"⚪ **{s['ticker']}**: {d['type']} ({d['tf']})\n"
            
    if sq_list:
        a_msg += "\n💥 **VOLATILITY SQUEEZES**\n"
        sq_list.sort(key=lambda x: x['squeeze']['tf'] == 'Weekly', reverse=True)
        for s in sq_list[:10]:
            if s in power: continue
            sq = s['squeeze']
            a_msg += f"⚠️ **{s['ticker']}**: {sq['tf']} ({sq['bias']})\n   └ Move: +/- {format_price(sq['move'])}\n"

    if fib_list:
        a_msg += "\n🕸️ **FIBONACCI LEVELS**\n"
        for s in fib_list[:10]:
            if s in power: continue
            f = s['fib']
            a_msg += f"🎯 **{s['ticker']}**: {f['action']} @ {f['level']} ({format_price(s['price'])})\n"

    send_telegram_alert(a_msg)
    print("Done.")
