import yfinance as yf
import pandas as pd
import pandas_datareader.data as web 
import requests
import os
import time
import io
import datetime
import numpy as np
import sys

# --- CONFIGURATION (INSTITUTIONAL WATCHLIST) ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']

STRATEGIC_TICKERS = [
    # -- Meme / PolitiFi --
    'DJT', 'PENGU-USD', 'FARTCOIN-USD', 'DOGE-USD', 'SHIB-USD', 'PEPE-USD',
    
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

# --- HELPER FUNCTIONS ---
def send_telegram_alert(message, header=""):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    
    full_msg = f"{header}\n\n{message}" if header else message
    
    # Chunking loop
    max_len = 4000
    for i in range(0, len(full_msg), max_len):
        chunk = full_msg[i:i+max_len]
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})
            time.sleep(1)
        except Exception as e:
            print(f"Telegram Error: {e}")

def format_price(price):
    if price is None: return "N/A"
    if price < 0.0001: return f"${price:.8f}"
    if price < 1.00: return f"${price:.4f}"
    return f"${price:.2f}"

# --- DATA FETCHERS (ANTI-BLOCK ARCHITECTURE) ---
def safe_download(ticker, period="2y"):
    """
    Uses the 'History' method which mimics a user session.
    Much harder for Yahoo to block than the bulk downloader.
    """
    try:
        dat = yf.Ticker(ticker)
        df = dat.history(period=period, auto_adjust=True)
        
        if df.empty or len(df) < 20: 
            # Fallback to standard download if history fails
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        
        if df.empty: return None
        
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df
    except Exception as e: 
        print(f"Fetch Error ({ticker}): {e}")
        return None

def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get(url, headers=headers)
        return [t.replace('.', '-') for t in pd.read_html(io.StringIO(r.text))[0]['Symbol'].tolist()]
    except: return []

def get_nasdaq_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get(url, headers=headers)
        return pd.read_html(io.StringIO(r.text))[0]['Ticker'].tolist()
    except: return []

def get_top_200_cryptos():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 200, 'page': 1}
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        return [f"{c['symbol'].upper()}-USD" for c in data]
    except: 
        print("⚠️ CoinGecko Failed - Using Static List")
        return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD']

def get_top_futures():
    return [
        'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F', 'FTSE=F',
        'CL=F', 'NG=F', 'RB=F', 'HO=F', 'BZ=F',
        'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F',
        'ZT=F', 'ZF=F', 'ZN=F', 'ZB=F',
        '6E=F', '6B=F', '6J=F', '6A=F', 'DX-Y.NYB',
        'ZC=F', 'ZS=F', 'ZW=F', 'ZL=F', 'ZM=F',
        'CC=F', 'KC=F', 'SB=F', 'CT=F', 'LE=F', 'HE=F'
    ]

# ==========================================================
#  SHARED MACRO DATA
# ==========================================================
def get_shared_macro_data():
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=730)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD', 'DGS10', 'DGS2', 'T5YIE'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        term_premia_proxy = fred['DGS10'] - fred['DGS2']
        
        # USE SAFE DOWNLOAD HERE to prevent blocking
        spy = safe_download('SPY')
        if spy is None: return None
        spy_close = spy['Close']
        
        return {'net_liq': net_liq, 'term_premia': term_premia_proxy, 'inflation': fred['T5YIE'], 'fed_assets': fred['WALCL'] / 1000, 'spy': spy_close}
    except Exception as e:
        print(f"Macro Data Error: {e}"); return None

# ==========================================================
#  MACRO ENGINES
# ==========================================================
def get_market_radar_regime(data):
    if not data: return "⚠️ Data Error", "NEUTRAL"
    liq_momo = data['net_liq'].pct_change(63).iloc[-1] * 100
    growth_momo = data['spy'].pct_change(126).iloc[-1] * 100
    
    if liq_momo > 0 and growth_momo > 0: regime = "RISK ON"
    elif liq_momo > 0 and growth_momo < 0: regime = "ACCUMULATE"
    elif liq_momo < 0 and growth_momo < 0: regime = "SLOW DOWN"
    else: regime = "RISK OFF"
    
    icon = "🟢" if regime == "RISK ON" else "🔵" if regime == "ACCUMULATE" else "🟠" if regime == "SLOW DOWN" else "🔴"
    return f"{icon} **{regime}**\n   └ Liq: {liq_momo:.2f}% | Growth: {growth_momo:.2f}%", regime

def get_michael_howell_update(data):
    if not data: return "⚠️ Data Error", "NEUTRAL"
    roc_med = data['net_liq'].pct_change(63).iloc[-1]
    acceleration = data['net_liq'].pct_change(63).diff(20).iloc[-1]
    tp_trend = "RISING (Bearish)" if data['term_premia'].iloc[-1] > data['term_premia'].iloc[-20] else "FALLING (Bullish)"
    
    fed_trend = data['fed_assets'].pct_change(63).iloc[-1]
    treasury_qe = (roc_med > -0.01 and fed_trend < 0)

    if roc_med > 0:
        if acceleration > 0: phase = "REBOUND (Early Cycle)"; action = "Overweight: Tech / Crypto / High Beta"
        else: phase = "SPECULATION (Late Cycle)"; action = "Overweight: Energy / Commodities / Bonds"
    else:
        if acceleration < 0: phase = "TURBULENCE (Contraction)"; action = "Overweight: Cash / Gold"
        else: phase = "CALM (Bottoming)"; action = "Accumulate: Credit / Quality"

    msg = f"🏛️ **CAPITAL WARS**\n   └ **Phase:** {phase}\n   └ **Term Premia:** {tp_trend}\n   └ **Action:** {action}"
    if treasury_qe: msg += "\n   └ 🚨 **Signal:** 'Treasury QE' Active (Baton Pass)"
    return msg, phase

def get_bitcoin_layer_update(data):
    if not data: return "⚠️ Data Error"
    try:
        # Use safe_download
        btc = safe_download('BTC-USD')
        if btc is None: return "⚠️ BTC Data Missing"
        
        combined = pd.DataFrame({'BTC': btc['Close'], 'LIQ': data['net_liq']}).dropna()
        correlation = combined['BTC'].rolling(90).corr(combined['LIQ']).iloc[-1]
        
        signal = "🟡 NEUTRAL"
        if correlation > 0.65: signal = "🟢 HIGH CONVICTION (Macro Driven)"
        elif correlation < 0.3: signal = "⚪ DECOUPLED (Idiosyncratic)"
        return f"🟠 **THE BITCOIN LAYER**\n   └ **Signal:** {signal} (Corr: {correlation:.2f})"
    except: return "⚠️ Bitcoin Layer Error"

def get_btc_macro_update(data):
    if not data: return "⚠️ Data Error"
    inf_trend = data['inflation'].pct_change(20).iloc[-1]
    tp_current = data['term_premia'].iloc[-1]
    tp_prev = data['term_premia'].iloc[-20]
    
    btc_signal = "⚪ NEUTRAL"
    if inf_trend > 0 and tp_current < tp_prev: btc_signal = "🟢 BULLISH (Debasement + Liquidity)"
    elif inf_trend < 0: btc_signal = "🔴 BEARISH (Disinflation)"
    elif tp_current > tp_prev: btc_signal = "🟡 CAUTION (Rising Rates)"
    return f"🪙 **BITCOIN MACRO SENSITIVITY**\n   └ **Signal:** {btc_signal}\n   └ **Inflation Swaps:** {'Rising' if inf_trend > 0 else 'Falling'}\n   └ **Term Premia:** {'Rising' if tp_current > tp_prev else 'Falling'}"

def get_onchain_update():
    try:
        btc = safe_download('BTC-USD')
        if btc is None: return "⚠️ BTC Data Missing"
        close = btc['Close']
        sma = close.rolling(20).mean(); std = close.rolling(20).std()
        bbw = ((sma + std*2) - (sma - std*2)) / sma
        bbw_rank = bbw.rolling(365).rank(pct=True).iloc[-1]
        squeeze_sig = "⚠️ **SQUEEZE** (Explosive Move)" if bbw_rank < 0.10 else "⚪ NORMAL"
        mayer = close.iloc[-1] / close.rolling(200).mean().iloc[-1]
        mayer_sig = "🟢 UNDERVALUED" if mayer < 0.8 else ("🔴 OVERHEATED" if mayer > 2.4 else "⚪ FAIR")
        return f"🔗 **ON-CHAIN & VOLATILITY**\n   └ **Squeeze:** {squeeze_sig}\n   └ **Mayer Multiple:** {mayer:.2f} ({mayer_sig})"
    except: return "⚠️ On-Chain Error"

# ==========================================================
#  ENGINE 6: TECHNICAL INDICATORS
# ==========================================================
def calculate_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except: return pd.Series([50]*len(series))

def check_squeeze(df):
    try:
        sma = df['Close'].rolling(20).mean(); std = df['Close'].rolling(20).std()
        upper_bb = sma + (std * 2); lower_bb = sma - (std * 2)
        tr = pd.DataFrame()
        tr['h-l'] = df['High'] - df['Low']
        tr['h-pc'] = abs(df['High'] - df['Close'].shift(1))
        tr['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
        atr = tr.max(axis=1).rolling(20).mean()
        upper_kc = sma + (atr * 1.5); lower_kc = sma - (atr * 1.5)
        
        if (lower_bb.iloc[-1] > lower_kc.iloc[-1]) and (upper_bb.iloc[-1] < upper_kc.iloc[-1]):
            y = df['Close'].iloc[-20:].values; x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return {'status': True, 'bias': "BULLISH 🟢" if slope > 0 else "BEARISH 🔴", 'move': atr.iloc[-1] * 2}
        return {'status': False}
    except: return {'status': False}

def calculate_demark(df):
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
        # USE SAFE DOWNLOAD
        df = safe_download(ticker, period="3y")
        
        if df is None: return None
        if not is_portfolio:
            if len(df) < 100: return None
            if df['Volume'].iloc[-5:].mean() == 0: return None 
            if df['Close'].iloc[-1] < 0.00000001: return None 
        
        df_weekly = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        if len(df_weekly) < 20: return None

        # INDICATORS
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # ADX
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = df['Low'].diff().clip(lower=0)
        tr = df['High'] - df['Low']
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
        dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
        df['ADX'] = dx.rolling(14).mean()

        for frame in [df, df_weekly]:
            frame['RSI'] = calculate_rsi(frame['Close'])
            frame = calculate_demark(frame)
        
        if not is_portfolio:
            if pd.isna(df['RSI'].iloc[-1]) or df['RSI'].iloc[-1] <= 1 or df['RSI'].iloc[-1] >= 99: return None

        d_sq = check_squeeze(df); w_sq = check_squeeze(df_weekly)
        last_d = df.iloc[-1]; last_w = df_weekly.iloc[-1]
        price = last_d['Close']
        
        # --- SIGNALS ---
        dm_sig = None; dm_data = None
        
        # Daily
        if last_d['Buy_Countdown'] == 13: dm_data = {'type': 'BUY 13', 'target': price*1.15, 'stop': min(df['Low'].iloc[-13:]), 'time': 'Reversal (Weeks)', 'tf': 'Daily'}
        elif last_d['Sell_Countdown'] == 13: dm_data = {'type': 'SELL 13', 'target': price*0.85, 'stop': max(df['High'].iloc[-13:]), 'time': 'Reversal (Weeks)', 'tf': 'Daily'}
        elif last_d['Buy_Setup'] == 9: dm_data = {'type': 'BUY 9', 'target': price*1.05, 'stop': min(df['Low'].iloc[-9:]), 'time': 'Bounce (1-4 Days)', 'tf': 'Daily'}
        elif last_d['Sell_Setup'] == 9: dm_data = {'type': 'SELL 9', 'target': price*0.95, 'stop': max(df['High'].iloc[-9:]), 'time': 'Pullback (1-4 Days)', 'tf': 'Daily'}
        
        # Weekly
        if last_w['Buy_Countdown'] == 13: dm_data = {'type': 'BUY 13', 'target': price*1.30, 'stop': min(df_weekly['Low'].iloc[-13:]), 'time': 'Major Bottom', 'tf': 'Weekly'}
        elif last_w['Sell_Countdown'] == 13: dm_data = {'type': 'SELL 13', 'target': price*0.70, 'stop': max(df_weekly['High'].iloc[-13:]), 'time': 'Major Top', 'tf': 'Weekly'}
        elif last_w['Buy_Setup'] == 9: dm_data = {'type': 'BUY 9', 'target': price*1.10, 'stop': min(df_weekly['Low'].iloc[-9:]), 'time': 'Trend Exhaustion', 'tf': 'Weekly'}
        elif last_w['Sell_Setup'] == 9: dm_data = {'type': 'SELL 9', 'target': price*0.90, 'stop': max(df_weekly['High'].iloc[-9:]), 'time': 'Trend Exhaustion', 'tf': 'Weekly'}
        
        d_perf = False
        if dm_data:
            if '13' in dm_data['type']: d_perf = True
            elif 'BUY 9' in dm_data['type']: d_perf = (df['Low'].iloc[-1] < df['Low'].iloc[-3] and df['Low'].iloc[-1] < df['Low'].iloc[-4])
            elif 'SELL 9' in dm_data['type']: d_perf = (df['High'].iloc[-1] > df['High'].iloc[-3] and df['High'].iloc[-1] > df['High'].iloc[-4])
            dm_data['perfected'] = d_perf
            dm_sig = dm_data
        
        rsi_sig = None
        if last_d['RSI'] < 30: rsi_sig = {'type': 'OVERSOLD', 'val': last_d['RSI'], 'target': df['Close'].rolling(20).mean().iloc[-1], 'stop': min(df['Low'].iloc[-5:]), 'time': 'Snapback (1-3 Days)'}
        elif last_d['RSI'] > 70: rsi_sig = {'type': 'OVERBOUGHT', 'val': last_d['RSI'], 'target': df['Close'].rolling(20).mean().iloc[-1], 'stop': max(df['High'].iloc[-5:]), 'time': 'Snapback (1-3 Days)'}
        
        sq_sig = None
        if d_sq['status']: sq_sig = {'tf': 'Daily', 'move': d_sq['move'], 'bias': d_sq['bias'], 'time': 'Imminent'}
        elif w_sq['status']: sq_sig = {'tf': 'Weekly', 'move': w_sq['move'], 'bias': w_sq['bias'], 'time': 'Building'}
        
        # Portfolio Context
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        trend = "BULLISH" if price > sma200 else "BEARISH"
        macd_val = df['MACD'].iloc[-1]
        macd_sig = "Bullish" if macd_val > df['Signal_Line'].iloc[-1] else "Bearish"
        
        # Cross logic
        tech_alert = None
        if (df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] < df['Signal_Line'].iloc[-2]): 
            tech_alert = {'type': 'BULLISH MACD CROSS', 'target': price + (atr.iloc[-1]*2), 'stop': price - (atr.iloc[-1]*1.5), 'time': 'Trend Change'}
        elif (df['MACD'].iloc[-1] < df['Signal_Line'].iloc[-1] and df['MACD'].iloc[-2] > df['Signal_Line'].iloc[-2]):
            tech_alert = {'type': 'BEARISH MACD CROSS', 'target': price - (atr.iloc[-1]*2), 'stop': price + (atr.iloc[-1]*1.5), 'time': 'Trend Change'}

        # Overall Verdict
        verdict = "HOLD"
        if dm_sig and "BUY" in dm_sig['type']: verdict = "BUY (Signal)"
        elif dm_sig and "SELL" in dm_sig['type']: verdict = "SELL (Signal)"
        elif trend == "BULLISH" and macd_sig == "Bullish": verdict = "BUY (Trend)"
        elif trend == "BEARISH" and macd_sig == "Bearish": verdict = "SELL (Trend)"

        count_str = f"Buy {last_d['Buy_Setup']}" if last_d['Buy_Setup'] > 0 else f"Sell {last_d['Sell_Setup']}"
        
        return {
            'ticker': ticker, 'price': price,
            'demark': dm_sig, 'rsi': rsi_sig, 'squeeze': sq_sig, 'perfected': d_perf, 'tech': tech_alert,
            'verdict': verdict, 'trend': trend, 'count': count_str, 
            'rsi_val': last_d['RSI'], 'macd_sig': macd_sig, 'adx': f"{last_d['ADX']:.0f}"
        }
    except Exception as e:
        # print(f"Err {ticker}: {e}") # Debug only
        return None

# ==========================================================
#  MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print(f"🚀 INITIALIZING INSTITUTIONAL SCANNER | Python: {sys.version}")
    
    print("1. Generating Macro Report...")
    macro_data = get_shared_macro_data()
    radar_txt, radar_regime = get_market_radar_regime(macro_data)
    howell_txt, howell_phase = get_michael_howell_update(macro_data)
    btc_layer = get_bitcoin_layer_update(macro_data)
    btc_macro = get_btc_macro_update(macro_data)
    onchain = get_onchain_update()
    
    macro_msg = f"🌍 **GLOBAL MACRO INSIGHTS** 🌍\n\n📊 **MARKET RADAR**\n{radar_txt}\n\n{howell_txt}\n\n{btc_layer}\n\n{btc_macro}\n\n{onchain}\n───────────────"
    send_telegram_alert(macro_msg)
    
    print("2. Analyzing Portfolio...")
    port_msg = "💼 **CURRENT PORTFOLIO INTELLIGENCE** 💼\n"
    
    # Calculate Overall Health
    port_data = []
    bull_count = 0; bear_count = 0
    for ticker in CURRENT_PORTFOLIO:
        res = analyze_ticker(ticker, is_portfolio=True)
        if res:
            port_data.append(res)
            if "BUY" in res['verdict'] or "BULLISH" in res['verdict']: bull_count += 1
            if "SELL" in res['verdict'] or "BEARISH" in res['verdict']: bear_count += 1
        else:
            port_msg += f"⚠️ **{ticker}**: Data Fetch Failed\n"
    
    overall = "🟢 BULLISH" if bull_count > bear_count else "🔴 BEARISH" if bear_count > bull_count else "🟡 NEUTRAL"
    port_msg += f"**Overall Technicals:** {overall} ({bull_count} Bull / {bear_count} Bear)\n───────────────\n"
    
    for res in port_data:
        p = format_price(res['price'])
        icon = "🟢" if "BUY" in res['verdict'] else "🔴" if "SELL" in res['verdict'] else "🟡"
        
        port_msg += f"{icon} **{res['ticker']}**: {res['verdict']} @ {p}\n"
        port_msg += f"   └ Trend: {res['trend']} | MACD: {res['macd_sig']} | RSI: {res['rsi_val']:.0f}\n"
        port_msg += f"   └ DeMark: {res['count']}\n"
        
        if res['demark']: port_msg += f"   🚨 SIGNAL: {res['demark']['type']} ({'Perf' if res['perfected'] else 'Unperf'})\n"
        if res['squeeze']: port_msg += f"   ⚠️ SQUEEZE: {res['squeeze']['tf']} ({res['squeeze']['bias']})\n"
        
        if "SLV" in res['ticker'] and "SPECULATION" in howell_phase: port_msg += "   ✅ **MACRO:** Aligned (Commodities Overweight)\n"
        port_msg += "\n"
        
    send_telegram_alert(port_msg)
    
    print("3. Scanning Tickers...")
    # Combine all lists + dynamic fetchers
    full_universe = list(set(STRATEGIC_TICKERS + get_top_200_cryptos() + get_top_futures() + get_sp500_tickers() + get_nasdaq_tickers()))
    print(f"Scanning {len(full_universe)} tickers...")
    
    power_list = []; perfected_list = []; unperfected_list = []; rsi_list = []; squeeze_list = []; tech_list = []
    
    for i, ticker in enumerate(full_universe):
        if i % 100 == 0: print(f"Processing {i}...")
        res = analyze_ticker(ticker)
        if res:
            d = res['demark']
            
            # Power Rankings Logic
            confluence = 0
            if d and res['perfected']: confluence += 1
            if res['rsi']: confluence += 1
            if res['squeeze']: confluence += 1
            if confluence >= 2 and d and res['perfected']: power_list.append(res)
            
            if d:
                if res['perfected']: perfected_list.append(res)
                else: unperfected_list.append(res)
            
            if res['rsi']: rsi_list.append(res)
            if res['squeeze']: squeeze_list.append(res)
            if res['tech']: tech_list.append(res)
            
        time.sleep(0.01) # Tiny sleep to let CPU breathe
        
    # --- REPORT GENERATION ---
    msg = "🔔 **INSTITUTIONAL SCANNER RESULTS** 🔔\n"
    
    if power_list:
        msg += "\n🔥 **POWER RANKINGS (Perfected + Confluence)** 🔥\n"
        for s in power_list[:15]:
            p = format_price(s['price'])
            d = s['demark']
            msg += f"🚀 **{s['ticker']}**: {p}\n   └ DeMark: {d['type']} ({d['tf']}) ✅\n   └ 🎯 Target: {format_price(d['target'])} | 🛑 Stop: {format_price(d['stop'])}\n"
            if s['rsi']: msg += f"   └ RSI: {s['rsi']['type']} ({s['rsi']['val']:.0f})\n"
            if s['squeeze']: msg += f"   └ Squeeze: {s['squeeze']['tf']} Active ({s['squeeze']['bias']})\n"
            msg += "───────────────\n"

    if perfected_list:
        msg += "\n✅ **PERFECTED DEMARK SIGNALS**\n"
        perfected_list.sort(key=lambda x: '13' in x['demark']['type'], reverse=True)
        for s in perfected_list[:15]:
            if s in power_list: continue 
            d = s['demark']; p = format_price(s['price'])
            icon = "🟢" if "BUY" in d['type'] else "🔴"
            msg += f"{icon} **{s['ticker']}**: {d['type']} ({d['tf']}) @ {p}\n   └ 🎯 Target: {format_price(d['target'])} | 🛑 Stop: {format_price(d['stop'])}\n   └ ⏳ Timing: {d['time']}\n"

    if tech_list:
        msg += "\n🌊 **MOMENTUM & TREND (MACD Cross)**\n"
        for s in tech_list[:10]:
            if s in power_list: continue
            t = s['tech']; p = format_price(s['price'])
            icon = "🟢" if "BULLISH" in t['type'] else "🔴"
            msg += f"{icon} **{s['ticker']}**: {t['type']} @ {p}\n   └ 🎯 Target: {format_price(t['target'])} | 🛑 Stop: {format_price(t['stop'])}\n   └ ⏳ Timing: {t['time']}\n"

    if unperfected_list:
        msg += "\n⚠️ **UNPERFECTED SIGNALS (Watchlist Only)**\n"
        unperfected_list.sort(key=lambda x: '13' in x['demark']['type'], reverse=True)
        for s in unperfected_list[:10]:
            d = s['demark']
            msg += f"⚪ **{s['ticker']}**: {d['type']} ({d['tf']}) - Unperfected\n"

    if rsi_list:
        msg += "\n2️⃣ **RSI EXTREMES (<30 or >70)**\n"
        rsi_list.sort(key=lambda x: abs(50 - x['rsi']['val']), reverse=True)
        for s in rsi_list[:10]:
            if s in power_list: continue
            r = s['rsi']; p = format_price(s['price'])
            icon = "🟢" if r['type'] == "OVERSOLD" else "🔴"
            msg += f"{icon} **{s['ticker']}**: {r['type']} ({r['val']:.0f}) @ {p}\n   └ 🎯 Reversion: {format_price(r['target'])}\n"

    if squeeze_list:
        msg += "\n3️⃣ **VOLATILITY SQUEEZES**\n"
        squeeze_list.sort(key=lambda x: x['squeeze']['tf'] == 'Weekly', reverse=True)
        for s in squeeze_list[:10]:
            if s in power_list: continue
            sq = s['squeeze']; p = format_price(s['price'])
            msg += f"⚠️ **{s['ticker']}**: {sq['tf']} Squeeze ({sq['bias']}) @ {p}\n   └ Exp. Move: +/- {format_price(sq['move'])}\n"

    if not (power_list or perfected_list or unperfected_list):
        msg = "No DeMark signals found today."

    send_telegram_alert(msg)
    print("Done.")
