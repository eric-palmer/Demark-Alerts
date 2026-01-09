import yfinance as yf
import pandas as pd
import pandas_datareader.data as web 
import requests
import os
import time
import io
import datetime
import numpy as np

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
    if len(full_msg) > 4000:
        for i in range(0, len(full_msg), 4000):
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": full_msg[i:i+4000], "parse_mode": "Markdown"})
    else:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                      json={"chat_id": chat_id, "text": full_msg, "parse_mode": "Markdown"})

def format_price(price):
    if price < 0.01: return f"${price:.6f}"
    if price < 1.00: return f"${price:.4f}"
    return f"${price:.2f}"

# --- DATA FETCHERS ---
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        return [t.replace('.', '-') for t in pd.read_html(io.StringIO(r.text))[0]['Symbol'].tolist()]
    except: return []

def get_nasdaq_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        return pd.read_html(io.StringIO(r.text))[0]['Ticker'].tolist()
    except: return []

def get_top_200_cryptos():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 250, 'page': 1}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return [f"{c['symbol'].upper()}-USD" for c in data][:200]
    except: return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD']

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
        spy = yf.download('SPY', start=start, progress=False)['Close']
        if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
        
        return {'net_liq': net_liq, 'term_premia': term_premia_proxy, 'inflation': fred['T5YIE'], 'fed_assets': fred['WALCL'] / 1000, 'spy': spy}
    except Exception as e:
        print(f"Data Fetch Error: {e}"); return None

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

    msg = f"🏛️ **CAPITAL WARS (Michael Howell)**\n   └ **Phase:** {phase}\n   └ **Term Premia:** {tp_trend}\n   └ **Action:** {action}"
    if treasury_qe: msg += "\n   └ 🚨 **Signal:** 'Treasury QE' Active (Baton Pass)"
    return msg, phase

def get_bitcoin_layer_update(data):
    if not data: return "⚠️ Data Error"
    try:
        btc = yf.download('BTC-USD', period="2y", progress=False)['Close']
        if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
        combined = pd.DataFrame({'BTC': btc, 'LIQ': data['net_liq']}).dropna()
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
        btc = yf.download('BTC-USD', period="2y", progress=False)['Close']
        if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
        sma = btc.rolling(20).mean(); std = btc.rolling(20).std()
        bbw = ((sma + std*2) - (sma - std*2)) / sma
        bbw_rank = bbw.rolling(365).rank(pct=True).iloc[-1]
        squeeze_sig = "⚠️ **SQUEEZE** (Explosive Move)" if bbw_rank < 0.10 else "⚪ NORMAL"
        mayer = btc.iloc[-1] / btc.rolling(200).mean().iloc[-1]
        mayer_sig = "🟢 UNDERVALUED" if mayer < 0.8 else ("🔴 OVERHEATED" if mayer > 2.4 else "⚪ FAIR")
        return f"🔗 **ON-CHAIN & VOLATILITY**\n   └ **Squeeze:** {squeeze_sig}\n   └ **Mayer Multiple:** {mayer:.2f} ({mayer_sig})"
    except: return "⚠️ On-Chain Error"

# ==========================================================
#  ENGINE 6: TECHNICAL INDICATORS (RSI, Squeeze, MACD, ADX)
# ==========================================================
def calculate_technical_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ADX (14) - Trend Strength
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr = pd.DataFrame()
    tr['h-l'] = df['High'] - df['Low']
    tr['h-pc'] = abs(df['High'] - df['Close'].shift(1))
    tr['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
    tr = tr.max(axis=1)
    atr = tr.rolling(14).mean()
    
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
    df['ADX'] = dx.rolling(14).mean()
    
    # Bollinger/Keltner Squeeze
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    upper_bb = sma + (std * 2); lower_bb = sma - (std * 2)
    upper_kc = sma + (atr * 1.5); lower_kc = sma - (atr * 1.5)
    
    is_squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    df['Squeeze'] = is_squeeze
    df['ATR'] = atr
    
    return df

def calculate_demark(df):
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

def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if 'Close' not in df.columns or df['Close'].iloc[-1] == 0: return None

        # Resample Weekly
        df_weekly = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        # Calculate All Indicators
        df = calculate_technical_indicators(df)
        df = calculate_demark(df)
        df_weekly = calculate_technical_indicators(df_weekly)
        df_weekly = calculate_demark(df_weekly)
        
        last_d = df.iloc[-1]; last_w = df_weekly.iloc[-1]
        price = last_d['Close']
        
        # --- 1. PORTFOLIO & GENERAL STATUS ---
        # Determine "Hold" State based on Trend & Momentum
        sma50 = df['Close'].rolling(50).mean().iloc[-1]
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        trend = "BULLISH" if price > sma200 else "BEARISH"
        
        # Targets for "Hold" State (ATR Bands)
        atr_d = last_d['ATR']
        hold_target = price + (atr_d * 2) # Resistance
        hold_stop = price - (atr_d * 1.5) # Support
        
        # --- 2. SIGNALS (DeMark) ---
        dm_sig = None; dm_data = None
        
        # Perfection Check
        d_perf = False
        if '9' in str(last_d['Buy_Setup']): d_perf = (df['Low'].iloc[-1] < df['Low'].iloc[-3] and df['Low'].iloc[-1] < df['Low'].iloc[-4])
        elif '9' in str(last_d['Sell_Setup']): d_perf = (df['High'].iloc[-1] > df['High'].iloc[-3] and df['High'].iloc[-1] > df['High'].iloc[-4])
        elif last_d['Buy_Countdown'] == 13 or last_d['Sell_Countdown'] == 13: d_perf = True
        
        # Signal Logic
        if last_d['Buy_Countdown'] == 13: dm_data = {'type': 'BUY 13', 'target': price*1.15, 'stop': min(df['Low'].iloc[-13:]), 'time': 'Reversal (Weeks)', 'tf': 'Daily'}
        elif last_d['Sell_Countdown'] == 13: dm_data = {'type': 'SELL 13', 'target': price*0.85, 'stop': max(df['High'].iloc[-13:]), 'time': 'Reversal (Weeks)', 'tf': 'Daily'}
        elif last_d['Buy_Setup'] == 9: dm_data = {'type': 'BUY 9', 'target': price*1.05, 'stop': min(df['Low'].iloc[-9:]), 'time': 'Bounce (1-4 Days)', 'tf': 'Daily'}
        elif last_d['Sell_Setup'] == 9: dm_data = {'type': 'SELL 9', 'target': price*0.95, 'stop': max(df['High'].iloc[-9:]), 'time': 'Pullback (1-4 Days)', 'tf': 'Daily'}
        
        # Weekly Overwrite
        if last_w['Buy_Countdown'] == 13: dm_data = {'type': 'BUY 13', 'target': price*1.30, 'stop': min(df_weekly['Low'].iloc[-13:]), 'time': 'Major Bottom', 'tf': 'Weekly'}
        elif last_w['Sell_Countdown'] == 13: dm_data = {'type': 'SELL 13', 'target': price*0.70, 'stop': max(df_weekly['High'].iloc[-13:]), 'time': 'Major Top', 'tf': 'Weekly'}
        elif last_w['Buy_Setup'] == 9: dm_data = {'type': 'BUY 9', 'target': price*1.10, 'stop': min(df_weekly['Low'].iloc[-9:]), 'time': 'Trend Exhaustion', 'tf': 'Weekly'}
        elif last_w['Sell_Setup'] == 9: dm_data = {'type': 'SELL 9', 'target': price*0.90, 'stop': max(df_weekly['High'].iloc[-9:]), 'time': 'Trend Exhaustion', 'tf': 'Weekly'}
        
        if dm_data:
            dm_data['perfected'] = d_perf
            dm_sig = dm_data
            
        # --- 3. RSI ---
        rsi_sig = None
        if last_d['RSI'] < 30: rsi_sig = {'type': 'OVERSOLD', 'val': last_d['RSI'], 'target': df['Close'].rolling(20).mean().iloc[-1], 'stop': min(df['Low'].iloc[-5:]), 'time': 'Snapback (1-3 Days)'}
        elif last_d['RSI'] > 70: rsi_sig = {'type': 'OVERBOUGHT', 'val': last_d['RSI'], 'target': df['Close'].rolling(20).mean().iloc[-1], 'stop': max(df['High'].iloc[-5:]), 'time': 'Snapback (1-3 Days)'}
        
        # --- 4. SQUEEZE ---
        sq_sig = None
        if last_d['Squeeze']:
            # Bias via Linear Reg
            y = df['Close'].iloc[-20:].values; x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            bias = "BULLISH 🟢" if slope > 0 else "BEARISH 🔴"
            sq_sig = {'tf': 'Daily', 'move': last_d['ATR']*2, 'bias': bias, 'time': 'Imminent'}
        elif last_w['Squeeze']:
            y = df_weekly['Close'].iloc[-20:].values; x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            bias = "BULLISH 🟢" if slope > 0 else "BEARISH 🔴"
            sq_sig = {'tf': 'Weekly', 'move': last_w['ATR']*2, 'bias': bias, 'time': 'Building'}
            
        # --- 5. ADVANCED TECHNICALS (MACD/ADX) ---
        # MACD Crossover
        macd_sig = "NEUTRAL"
        if last_d['MACD'] > last_d['Signal_Line'] and df['MACD'].iloc[-2] < df['Signal_Line'].iloc[-2]: macd_sig = "BULLISH CROSS 🟢"
        elif last_d['MACD'] < last_d['Signal_Line'] and df['MACD'].iloc[-2] > df['Signal_Line'].iloc[-2]: macd_sig = "BEARISH CROSS 🔴"
        
        # ADX Regime
        adx_val = last_d['ADX']
        adx_regime = "TRENDING 📈" if adx_val > 25 else "CHOPPY 〰️"
        
        tech_sig = None
        if "CROSS" in macd_sig:
            tech_sig = {'type': f"MACD {macd_sig}", 'adx': f"{adx_val:.0f} ({adx_regime})", 'target': hold_target, 'stop': hold_stop, 'time': 'Trend Change'}

        # --- PORTFOLIO DECISION LOGIC ---
        # Prioritize signals -> then trend
        rating = "HOLD"
        p_target = hold_target
        p_stop = hold_stop
        p_time = "Indefinite"
        
        if dm_sig:
            rating = "BUY" if "BUY" in dm_sig['type'] else "SELL"
            p_target = dm_sig['target']; p_stop = dm_sig['stop']; p_time = dm_sig['time']
        elif sq_sig:
            rating = "PREPARE"
            p_target = price + sq_sig['move']; p_stop = price - sq_sig['move']; p_time = sq_sig['time']
        elif "BULLISH" in trend:
            rating = "HOLD (Trend)"
        else:
            rating = "HOLD (Caution)"

        return {
            'ticker': ticker, 'price': price,
            'demark': dm_sig, 'rsi': rsi_sig, 'squeeze': sq_sig, 'tech': tech_sig, 'perfected': d_perf,
            'trend': trend, 'rating': rating, 'p_target': p_target, 'p_stop': p_stop, 'p_time': p_time,
            'rsi_val': last_d['RSI'], 'adx_val': adx_val, 'macd_sig': macd_sig
        }
    except: return None

# ==========================================================
#  MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
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
    for ticker in CURRENT_PORTFOLIO:
        res = analyze_ticker(ticker)
        if res:
            p = format_price(res['price'])
            t = format_price(res['p_target'])
            s = format_price(res['p_stop'])
            
            icon = "🟢" if "BUY" in res['rating'] or "Trend" in res['rating'] else "🔴" if "SELL" in res['rating'] else "🟡"
            
            port_msg += f"{icon} **{ticker}**: {res['rating']} @ {p}\n"
            port_msg += f"   └ 🎯 Target: {t} | 🛑 Stop: {s}\n"
            port_msg += f"   └ ⏳ Timing: {res['p_time']}\n"
            port_msg += f"   └ 📊 Techs: RSI {res['rsi_val']:.0f} | MACD {res['macd_sig']}\n"
            if res['squeeze']: port_msg += f"   ⚠️ SQUEEZE: {res['squeeze']['tf']} ({res['squeeze']['bias']})\n"
            
            # Context
            if "SLV" in ticker and "SPECULATION" in howell_phase: port_msg += "   ✅ **MACRO:** Aligned (Commodities Overweight)\n"
            port_msg += "───────────────\n"
    send_telegram_alert(port_msg)
    
    print("3. Scanning Tickers...")
    full_universe = list(set(STRATEGIC_TICKERS + get_top_200_cryptos() + get_top_futures() + get_sp500_tickers() + get_nasdaq_tickers()))
    print(f"Scanning {len(full_universe)} tickers...")
    
    power_list = []; perfected_list = []; unperfected_list = []; rsi_list = []; squeeze_list = []; tech_list = []
    
    for i, ticker in enumerate(full_universe):
        if i % 100 == 0: print(f"Processing {i}/{len(full_universe)}...")
        res = analyze_ticker(ticker)
        if res:
            d = res['demark']
            
            # 1. Power Rankings (Strict: Perfected + Confluence)
            confluence = 0
            if d and res['perfected']: confluence += 1
            if res['rsi']: confluence += 1
            if res['squeeze']: confluence += 1
            if res['tech']: confluence += 1 # MACD/ADX
            if confluence >= 2 and d and d['perfected']: 
                power_list.append(res)
            
            # 2. Categorization
            if d:
                if res['perfected']: perfected_list.append(res)
                else: unperfected_list.append(res)
            
            if res['rsi']: rsi_list.append(res)
            if res['squeeze']: squeeze_list.append(res)
            if res['tech']: tech_list.append(res)
            
        time.sleep(0.01)
        
    # --- REPORT GENERATION ---
    msg = "🔔 **INSTITUTIONAL SCANNER RESULTS** 🔔\n"
    
    # 1. POWER RANKINGS
    if power_list:
        msg += "\n🔥 **POWER RANKINGS (Perfected + Confluence)** 🔥\n"
        for s in power_list[:15]:
            p = format_price(s['price'])
            d = s['demark']
            msg += f"🚀 **{s['ticker']}**: {p}\n"
            if d: msg += f"   └ DeMark: {d['type']} ({d['tf']}) ✅\n"
            if d: msg += f"   └ 🎯 Target: {format_price(d['target'])} | 🛑 Stop: {format_price(d['stop'])}\n"
            if s['rsi']: msg += f"   └ RSI: {s['rsi']['type']} ({s['rsi']['val']:.0f})\n"
            if s['squeeze']: msg += f"   └ Squeeze: {s['squeeze']['tf']} Active ({s['squeeze']['bias']})\n"
            if s['tech']: msg += f"   └ Tech: {s['tech']['type']}\n"
            msg += "───────────────\n"

    # 2. PERFECTED DEMARK
    if perfected_list:
        msg += "\n✅ **PERFECTED DEMARK SIGNALS**\n"
        perfected_list.sort(key=lambda x: '13' in x['demark']['type'], reverse=True)
        for s in perfected_list[:15]:
            if s in power_list: continue 
            d = s['demark']
            p = format_price(s['price'])
            icon = "🟢" if "BUY" in d['type'] else "🔴"
            msg += f"{icon} **{s['ticker']}**: {d['type']} ({d['tf']}) @ {p}\n"
            msg += f"   └ 🎯 Target: {format_price(d['target'])} | 🛑 Stop: {format_price(d['stop'])}\n"
            msg += f"   └ ⏳ Timing: {d['time']}\n"

    # 3. MOMENTUM & TREND (New Section)
    if tech_list:
        msg += "\n🌊 **MOMENTUM & TREND (MACD/ADX)**\n"
        for s in tech_list[:10]:
            if s in power_list: continue
            t = s['tech']
            icon = "🟢" if "BULLISH" in t['type'] else "🔴"
            msg += f"{icon} **{s['ticker']}**: {t['type']}\n"
            msg += f"   └ ADX: {t['adx']}\n"

    # 4. UNPERFECTED DEMARK
    if unperfected_list:
        msg += "\n⚠️ **UNPERFECTED SIGNALS (Watchlist Only)**\n"
        unperfected_list.sort(key=lambda x: '13' in x['demark']['type'], reverse=True)
        for s in unperfected_list[:10]:
            d = s['demark']
            msg += f"⚪ **{s['ticker']}**: {d['type']} ({d['tf']}) - Unperfected\n"

    # 5. RSI SIGNALS
    if rsi_list:
        msg += "\n2️⃣ **RSI EXTREMES (<30 or >70)**\n"
        rsi_list.sort(key=lambda x: abs(50 - x['rsi']['val']), reverse=True)
        for s in rsi_list[:10]:
            if s in power_list: continue
            r = s['rsi']; p = format_price(s['price'])
            icon = "🟢" if r['type'] == "OVERSOLD" else "🔴"
            msg += f"{icon} **{s['ticker']}**: {r['type']} ({r['val']:.0f}) @ {p}\n"
            msg += f"   └ 🎯 Reversion: {format_price(r['target'])}\n"
            msg += f"   └ ⏳ Timing: {r['time']}\n"

    # 6. SQUEEZE SIGNALS
    if squeeze_list:
        msg += "\n3️⃣ **VOLATILITY SQUEEZES**\n"
        squeeze_list.sort(key=lambda x: x['squeeze']['tf'] == 'Weekly', reverse=True)
        for s in squeeze_list[:10]:
            if s in power_list: continue
            sq = s['squeeze']; p = format_price(s['price'])
            msg += f"⚠️ **{s['ticker']}**: {sq['tf']} Squeeze ({sq['bias']}) @ {p}\n"
            msg += f"   └ Exp. Move: +/- {format_price(sq['move'])}\n"
            msg += f"   └ ⏳ Timing: {sq['time']}\n"

    if not (power_list or perfected_list or unperfected_list):
        msg = "No DeMark signals found today."

    send_telegram_alert(msg)
    print("Done.")
