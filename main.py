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

# --- HELPER: TELEGRAM SENDER ---
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
#  ENGINE 1: MARKET RADAR (Regime)
# ==========================================================
def get_market_radar_regime(data):
    if not data: return "⚠️ Data Error"
    liq_momo = data['net_liq'].pct_change(63).iloc[-1] * 100
    growth_momo = data['spy'].pct_change(126).iloc[-1] * 100
    
    if liq_momo > 0 and growth_momo > 0: return f"🟢 **RISK ON** (Reflation)\n   └ Liq: {liq_momo:.2f}% | Growth: {growth_momo:.2f}%"
    elif liq_momo > 0 and growth_momo < 0: return f"🔵 **ACCUMULATE** (Recovery)\n   └ Liq: {liq_momo:.2f}% | Growth: {growth_momo:.2f}%"
    elif liq_momo < 0 and growth_momo < 0: return f"🟠 **SLOW DOWN** (Deflation)\n   └ Liq: {liq_momo:.2f}% | Growth: {growth_momo:.2f}%"
    else: return f"🔴 **RISK OFF** (Turbulence)\n   └ Liq: {liq_momo:.2f}% | Growth: {growth_momo:.2f}%"

# ==========================================================
#  ENGINE 2: MICHAEL HOWELL (Capital Wars)
# ==========================================================
def get_michael_howell_update(data):
    if not data: return "⚠️ Data Error"
    roc_med = data['net_liq'].pct_change(63).iloc[-1]
    acceleration = data['net_liq'].pct_change(63).diff(20).iloc[-1]
    
    tp_current = data['term_premia'].iloc[-1]
    tp_prev = data['term_premia'].iloc[-20]
    tp_trend = "RISING (Bearish)" if tp_current > tp_prev else "FALLING (Bullish)"
    
    fed_trend = data['fed_assets'].pct_change(63).iloc[-1]
    treasury_qe = (roc_med > -0.01 and fed_trend < 0)

    if roc_med > 0:
        if acceleration > 0: phase = "REBOUND (Early Cycle)"; action = "Overweight: Tech / Crypto / High Beta"
        else: phase = "SPECULATION (Late Cycle)"; action = "Overweight: Energy / Commodities / 5Y Bonds"
    else:
        if acceleration < 0: phase = "TURBULENCE (Contraction)"; action = "Overweight: Cash / Gold"
        else: phase = "CALM (Bottoming)"; action = "Accumulate: Credit / Quality"

    msg = f"🏛️ **CAPITAL WARS (Michael Howell)**\n   └ **Phase:** {phase}\n   └ **Liquidity:** {'Expanding' if roc_med > 0 else 'Contracting'} ({roc_med*100:.2f}%)\n   └ **Term Premia:** {tp_trend} (Slope: {tp_current:.2f}bps)\n   └ **Action:** {action}\n"
    if treasury_qe: msg += "   └ 🚨 **Signal:** 'Treasury QE' Active (Baton Pass)"
    return msg

# ==========================================================
#  ENGINE 3: BITCOIN LAYER (Correlation)
# ==========================================================
def get_bitcoin_layer_update(data):
    if not data: return "⚠️ Data Error"
    try:
        btc = yf.download('BTC-USD', period="2y", progress=False)['Close']
        if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
        combined = pd.DataFrame({'BTC': btc, 'LIQ': data['net_liq']}).dropna()
        correlation = combined['BTC'].rolling(90).corr(combined['LIQ']).iloc[-1]
        liq_trend = data['net_liq'].pct_change(63).iloc[-1]
        
        signal = "🟡 NEUTRAL"
        if correlation > 0.65: signal = "🟢 HIGH CONVICTION LONG" if liq_trend > 0 else "🔴 HIGH CONVICTION SHORT"
        elif correlation < 0.3: signal = "⚪ DECOUPLED (Idiosyncratic)"
        return f"🟠 **THE BITCOIN LAYER**\n   └ **Signal:** {signal}\n   └ **Correlation:** {correlation:.2f} (90-Day)"
    except: return "⚠️ Bitcoin Layer Error"

# ==========================================================
#  ENGINE 4: BITCOIN MACRO (Capital Flows)
# ==========================================================
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

# ==========================================================
#  ENGINE 5: ON-CHAIN & SQUEEZE INTELLIGENCE
# ==========================================================
def get_onchain_update():
    try:
        btc = yf.download('BTC-USD', period="2y", progress=False)['Close']
        if isinstance(btc, pd.DataFrame): btc = btc.iloc[:, 0]
        sma = btc.rolling(20).mean(); std = btc.rolling(20).std()
        bbw = ((sma + std*2) - (sma - std*2)) / sma
        bbw_rank = bbw.rolling(365).rank(pct=True).iloc[-1]
        squeeze_sig = "⚠️ **SQUEEZE** (Explosive Move Imminent)" if bbw_rank < 0.10 else "⚪ NORMAL"
        mayer = btc.iloc[-1] / btc.rolling(200).mean().iloc[-1]
        mayer_sig = "🟢 UNDERVALUED" if mayer < 0.8 else ("🔴 OVERHEATED" if mayer > 2.4 else "⚪ FAIR")
        return f"🔗 **ON-CHAIN & VOLATILITY**\n   └ **Squeeze:** {squeeze_sig}\n   └ **Mayer Multiple:** {mayer:.2f} ({mayer_sig})"
    except: return "⚠️ On-Chain Error"

# ==========================================================
#  ENGINE 6: MULTI-TIMEFRAME ANALYSIS (DeMark + RSI + Squeeze)
# ==========================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def check_squeeze(df):
    sma = df['Close'].rolling(20).mean(); std = df['Close'].rolling(20).std()
    upper_bb = sma + (std * 2); lower_bb = sma - (std * 2)
    tr = pd.DataFrame()
    tr['h-l'] = df['High'] - df['Low']
    tr['h-pc'] = abs(df['High'] - df['Close'].shift(1))
    tr['l-pc'] = abs(df['Low'] - df['Close'].shift(1))
    atr = tr.max(axis=1).rolling(20).mean()
    upper_kc = sma + (atr * 1.5); lower_kc = sma - (atr * 1.5)
    is_squeeze = (lower_bb.iloc[-1] > lower_kc.iloc[-1]) and (upper_bb.iloc[-1] < upper_kc.iloc[-1])
    
    if is_squeeze:
        y = df['Close'].iloc[-20:].values; x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return {'status': True, 'bias': "BULLISH 🟢" if slope > 0 else "BEARISH 🔴", 'move': atr.iloc[-1] * 2}
    return {'status': False}

def calculate_demark(df):
    df['Close_4'] = df['Close'].shift(4)
    df['Buy_Setup'] = 0; df['Sell_Setup'] = 0
    df['Buy_Countdown'] = 0; df['Sell_Countdown'] = 0
    df['Buy_13_Perfected'] = False; df['Sell_13_Perfected'] = False
    buy_seq = 0; sell_seq = 0; buy_cd = 0; sell_cd = 0
    active_buy = False; active_sell = False; buy_idxs = []; sell_idxs = []

    closes = df['Close'].values; closes_4 = df['Close_4'].values
    lows = df['Low'].values; highs = df['High'].values
    
    for i in range(4, len(df)):
        if closes[i] < closes_4[i]: buy_seq += 1; sell_seq = 0
        elif closes[i] > closes_4[i]: sell_seq += 1; buy_seq = 0
        else: buy_seq = 0; sell_seq = 0
        
        df.iloc[i, df.columns.get_loc('Buy_Setup')] = buy_seq
        df.iloc[i, df.columns.get_loc('Sell_Setup')] = sell_seq
        
        if buy_seq == 9: active_buy = True; buy_cd = 0; buy_idxs = []; active_sell = False
        if sell_seq == 9: active_sell = True; sell_cd = 0; sell_idxs = []; active_buy = False

        if active_buy:
            if closes[i] <= lows[i-2]:
                buy_cd += 1; buy_idxs.append(i)
                df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_cd
                if buy_cd == 13:
                    if len(buy_idxs) >= 8 and lows[i] <= closes[buy_idxs[7]]: df.iloc[i, df.columns.get_loc('Buy_13_Perfected')] = True
                    active_buy = False
        if active_sell:
            if closes[i] >= highs[i-2]:
                sell_cd += 1; sell_idxs.append(i)
                df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_cd
                if sell_cd == 13:
                    if len(sell_idxs) >= 8 and highs[i] >= closes[sell_idxs[7]]: df.iloc[i, df.columns.get_loc('Sell_13_Perfected')] = True
                    active_sell = False
    return df

def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if 'Close' not in df.columns: return None

        # --- MULTI-TIMEFRAME ANALYSIS ---
        df_weekly = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        # Calculate Indicators
        for frame in [df, df_weekly]:
            frame['RSI'] = calculate_rsi(frame['Close'])
        
        d_squeeze = check_squeeze(df)
        w_squeeze = check_squeeze(df_weekly)
        
        df = calculate_demark(df)
        df_weekly = calculate_demark(df_weekly)
        
        last_d = df.iloc[-1]; last_w = df_weekly.iloc[-1]
        price = last_d['Close']
        
        # --- SIGNAL LOGIC ---
        d_sig = None; w_sig = None; d_det = ""
        
        # Daily Logic
        if last_d['Buy_Countdown'] == 13: d_sig = "BUY 13"; d_det = f"ACCUMULATE | Stop: ${min(df['Low'].iloc[-13:]):.2f}"
        elif last_d['Sell_Countdown'] == 13: d_sig = "SELL 13"; d_det = f"DISTRIBUTE | Stop: ${max(df['High'].iloc[-13:]):.2f}"
        elif last_d['Buy_Setup'] == 9: d_sig = "BUY 9"; d_det = f"SCALP LONG | Stop: ${min(df['Low'].iloc[-9:]):.2f}"
        elif last_d['Sell_Setup'] == 9: d_sig = "SELL 9"; d_det = f"SCALP SHORT | Stop: ${max(df['High'].iloc[-9:]):.2f}"
        
        # Weekly Logic
        if last_w['Buy_Countdown'] == 13: w_sig = "BUY 13 (Major Bottom)"
        elif last_w['Sell_Countdown'] == 13: w_sig = "SELL 13 (Major Top)"
        elif last_w['Buy_Setup'] == 9: w_sig = "BUY 9 (Trend Exh)"
        elif last_w['Sell_Setup'] == 9: w_sig = "SELL 9 (Trend Exh)"
        
        # Perfection
        d_perf = last_d.get('Buy_13_Perfected') or last_d.get('Sell_13_Perfected') or False
        if d_sig and '9' in d_sig:
            if 'BUY' in d_sig:
                l9, l8, l7, l6 = df['Low'].iloc[-1], df['Low'].iloc[-2], df['Low'].iloc[-3], df['Low'].iloc[-4]
                d_perf = (l9 < l7 and l9 < l6) or (l8 < l7 and l8 < l6)
            else:
                h9, h8, h7, h6 = df['High'].iloc[-1], df['High'].iloc[-2], df['High'].iloc[-3], df['High'].iloc[-4]
                d_perf = (h9 > h7 and h9 > h6) or (h8 > h7 and h8 > h6)

        return {
            'ticker': ticker, 'price': price,
            'daily_sig': d_sig, 'weekly_sig': w_sig, 'daily_detail': d_det, 'perfected': d_perf,
            'daily_rsi': last_d['RSI'], 'weekly_rsi': last_w['RSI'],
            'daily_squeeze': d_squeeze, 'weekly_squeeze': w_squeeze
        }
    except: return None

# ==========================================================
#  MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("1. Generating Macro Report...")
    macro_data = get_shared_macro_data()
    radar = get_market_radar_regime(macro_data)
    howell = get_michael_howell_update(macro_data)
    btc_layer = get_bitcoin_layer_update(macro_data)
    btc_macro = get_btc_macro_update(macro_data)
    onchain = get_onchain_update()
    
    macro_msg = f"🌍 **GLOBAL MACRO INSIGHTS** 🌍\n\n📊 **MARKET RADAR**\n{radar}\n\n{howell}\n\n{btc_layer}\n\n{btc_macro}\n\n{onchain}\n───────────────"
    send_telegram_alert(macro_msg)
    
    print("2. Scanning Tickers...")
    full_universe = list(set(STRATEGIC_TICKERS + get_top_200_cryptos() + get_top_futures() + get_sp500_tickers() + get_nasdaq_tickers()))
    print(f"Scanning {len(full_universe)} tickers...")
    
    power_ratings = []
    
    for i, ticker in enumerate(full_universe):
        if i % 100 == 0: print(f"Processing {i}/{len(full_universe)}...")
        res = analyze_ticker(ticker)
        if res:
            # Only keep tickers with at least ONE significant signal
            if res['daily_sig'] or res['weekly_sig'] or res['daily_squeeze']['status'] or res['weekly_squeeze']['status']:
                power_ratings.append(res)
        time.sleep(0.01)
    
    # --- REPORT GENERATION ---
    msg = "🔔 **POWER RATING: HIGH CONVICTION** 🔔\n"
    
    # 1. DUAL CONVICTION (Daily + Weekly)
    dual = [s for s in power_ratings if s['daily_sig'] and s['weekly_sig']]
    if dual:
        msg += "\n🔥 **DUAL CONVICTION (All-In)** 🔥\n"
        for s in dual:
            icon = "🟢" if "BUY" in s['daily_sig'] else "🔴"
            msg += f"{icon} **{s['ticker']}**: ${s['price']:.2f}\n"
            msg += f"   ⚡ Daily: {s['daily_sig']} (RSI: {s['daily_rsi']:.0f}) | {s['daily_detail']}\n"
            msg += f"   📅 Weekly: {s['weekly_sig']} (RSI: {s['weekly_rsi']:.0f})\n"
            if s['daily_squeeze']['status']: msg += f"   ⚠️ SQUEEZE: {s['daily_squeeze']['bias']} (Exp Move: {s['daily_squeeze']['move']:.2f})\n"
            msg += "───────────────\n"

    # 2. POSITION SETUPS (Weekly Focused)
    weekly_only = [s for s in power_ratings if s['weekly_sig'] and not s['daily_sig']]
    if weekly_only:
        msg += "\n📅 **POSITION SETUPS (Weekly Conviction)**\n"
        for s in weekly_only:
            icon = "🟢" if "BUY" in s['weekly_sig'] else "🔴"
            msg += f"{icon} **{s['ticker']}**: {s['weekly_sig']}\n"
            msg += f"   └ RSI: {s['weekly_rsi']:.0f} | Price: ${s['price']:.2f}\n"
            if s['weekly_squeeze']['status']: msg += f"   ⚠️ WEEKLY SQUEEZE DETECTED\n"

    # 3. TACTICAL SETUPS (Daily Focused)
    daily_only = [s for s in power_ratings if s['daily_sig'] and not s['weekly_sig']]
    daily_only.sort(key=lambda x: (x['daily_squeeze']['status'], '13' in str(x['daily_sig'])), reverse=True)
    
    if daily_only:
        msg += "\n⚡ **TACTICAL SETUPS (Daily Swing)**\n"
        for s in daily_only[:20]: # Top 20 to save space
            icon = "🟢" if "BUY" in s['daily_sig'] else "🔴"
            msg += f"{icon} **{s['ticker']}**: {s['daily_sig']} ({'⭐' if s['perfected'] else ''})\n"
            msg += f"   └ {s['daily_detail']}\n"
            if s['daily_squeeze']['status']: msg += f"   ⚠️ SQUEEZE: {s['daily_squeeze']['bias']}\n"

    if not dual and not weekly_only and not daily_only:
        msg = "No significant setups found today."

    send_telegram_alert(msg)
    print("Done.")
