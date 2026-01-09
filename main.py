import yfinance as yf
import pandas as pd
import pandas_datareader.data as web 
import requests
import os
import time
import io
import datetime
import numpy as np

# --- CONFIGURATION (CUSTOM WATCHLIST) ---
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
        'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F', 'FTSE=F', # Indices
        'CL=F', 'NG=F', 'RB=F', 'HO=F', 'BZ=F', # Energy
        'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', # Metals
        'ZT=F', 'ZF=F', 'ZN=F', 'ZB=F', # Rates
        '6E=F', '6B=F', '6J=F', '6A=F', 'DX-Y.NYB', # Currencies
        'ZC=F', 'ZS=F', 'ZW=F', 'ZL=F', 'ZM=F', # Grains
        'CC=F', 'KC=F', 'SB=F', 'CT=F', 'LE=F', 'HE=F' # Softs/Meat
    ]

# ==========================================================
#  ENGINE 1: MICHAEL HOWELL (Capital Wars)
# ==========================================================
def get_michael_howell_update():
    """
    Fixed Logic: Ensures no blank output.
    Tracks: Net Liquidity Cycle, Treasury QE, and Term Premia.
    """
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=730)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        
        # Momentum (63-day) & Acceleration (20-day delta of momentum)
        roc_med = net_liq.pct_change(63).iloc[-1]
        roc_series = net_liq.pct_change(63)
        acceleration = roc_series.diff(20).iloc[-1]
        
        # Treasury QE (Net Liq Up vs Fed Assets Down)
        fed_assets_trend = (fred['WALCL'] / 1000).pct_change(63).iloc[-1]
        treasury_qe = (roc_med > -0.01 and fed_assets_trend < 0)

        # Cycle Logic (Exhaustive)
        if roc_med > 0 and acceleration >= 0:
            phase = "REBOUND (Early Cycle)"
            action = "Overweight: Tech / Crypto / High Beta"
        elif roc_med > 0 and acceleration < 0:
            phase = "SPECULATION (Late Cycle / Peaking)"
            action = "Overweight: Energy / Commodities\n   👉 Buy: 5Y Treasuries (Falling Term Premia)"
        elif roc_med <= 0 and acceleration < 0:
            phase = "TURBULENCE (Contraction)"
            action = "Overweight: Cash / Gold / Volatility\n   ⚠️ Avoid: Credit & High Beta"
        elif roc_med <= 0 and acceleration >= 0:
            phase = "CALM (Bottoming)"
            action = "Accumulate: Corporate Credit / Quality Stocks"
        else:
            phase = "NEUTRAL (Transition)"
            action = "Hold Quality / Hedged"

        tp_signal = "Falling (Bullish Bonds) 📉" if roc_med < 0 else "Rising (Bearish Bonds) 📈"

        msg = f"🏛️ **CAPITAL WARS (Michael Howell)**\n"
        msg += f"   └ **Phase:** {phase}\n"
        msg += f"   └ **Trend:** {'Expanding' if roc_med > 0 else 'Contracting'} ({roc_med*100:.2f}%)\n"
        msg += f"   └ **Term Premia:** {tp_signal}\n"
        msg += f"   └ **Action:** {action}\n"
        
        if treasury_qe:
            msg += "   └ 🚨 **Signal:** 'Treasury QE' Active (Baton Pass)"
        
        return msg
    except Exception as e: return f"⚠️ Howell Engine Error: {e}"

# ==========================================================
#  ENGINE 2: BTC MACRO SENSITIVITY
# ==========================================================
def get_btc_macro_sensitivity():
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=400)
        fred = web.DataReader(['T5YIE', 'DGS10', 'DGS2'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        
        inflation_exp = fred['T5YIE'].iloc[-1]
        inflation_trend = fred['T5YIE'].pct_change(20).iloc[-1]
        
        # Term Premia Proxy (Steepener vs Flattener)
        tp_proxy = fred['DGS10'] - fred['DGS2']
        tp_current = tp_proxy.iloc[-1]
        tp_prev = tp_proxy.iloc[-20]
        tp_trend = "RISING (Bearish)" if tp_current > tp_prev else "FALLING (Bullish)"
        
        btc_signal = "⚪ NEUTRAL"
        if inflation_trend > 0 and tp_current < tp_prev:
            btc_signal = "🟢 BULLISH (Ideal: Inflation Up / TP Down)"
        elif inflation_trend < 0:
            btc_signal = "🔴 BEARISH (Disinflation)"
            
        return f"🪙 **BITCOIN MACRO SENSITIVITY**\n   └ **Signal:** {btc_signal}\n   └ **Inflation Swaps:** {inflation_exp:.2f}% ({'Up' if inflation_trend > 0 else 'Down'})\n   └ **Term Premia:** {tp_trend}"
    except Exception as e: return f"⚠️ BTC Macro Error: {e}"

# ==========================================================
#  ENGINE 3: THE BITCOIN LAYER
# ==========================================================
def get_bitcoin_layer_update():
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=400)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        
        velocity = net_liq.diff().diff().rolling(window=10).mean().iloc[-1]
        signal = "🟢 HIGH VELOCITY (Impulse Up)" if velocity > 0 else "🟡 LOW VELOCITY (Stalling)"
        
        return f"🟠 **THE BITCOIN LAYER**\n   └ **Velocity:** {signal}\n   └ **Metric:** 2nd Derivative of Net Liq"
    except: return "⚠️ Bitcoin Layer Error"

# ==========================================================
#  ENGINE 4: REAL VISION (Everything Code)
# ==========================================================
def get_real_vision_update():
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=400)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        spy = yf.download('SPY', start=start, progress=False)['Close']
        if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
        
        liq_momo = net_liq.pct_change(63).iloc[-1]
        growth_momo = spy.pct_change(63).iloc[-1]
        
        regime = "NEUTRAL"
        if liq_momo > 0 and growth_momo <= 0: regime = "🍌 **BANANA ZONE** (Accumulate)"
        elif liq_momo > 0 and growth_momo > 0: regime = "🟢 **SPRING** (Risk On)"
        elif liq_momo < 0 and growth_momo < 0: regime = "🍂 **FALL** (Slowdown)"
        else: regime = "❄️ **WINTER** (Risk Off)"
            
        return f"🧠 **REAL VISION**\n   └ **Regime:** {regime}"
    except: return "⚠️ Real Vision Error"

# ==========================================================
#  ENGINE 5: MULTI-TIMEFRAME DEMARK
# ==========================================================
def calculate_demark(df):
    df['Close_4'] = df['Close'].shift(4)
    df['Buy_Setup'] = 0; df['Sell_Setup'] = 0
    df['Buy_Countdown'] = 0; df['Sell_Countdown'] = 0
    df['Buy_13_Perfected'] = False; df['Sell_13_Perfected'] = False

    buy_seq = 0; sell_seq = 0
    buy_cd = 0; sell_cd = 0
    active_buy = False; active_sell = False
    buy_idxs = []; sell_idxs = []

    closes = df['Close'].values; closes_4 = df['Close_4'].values
    lows = df['Low'].values; highs = df['High'].values
    
    for i in range(4, len(df)):
        if closes[i] < closes_4[i]: buy_seq += 1
        else: buy_seq = 0
        df.iloc[i, df.columns.get_loc('Buy_Setup')] = buy_seq
        
        if closes[i] > closes_4[i]: sell_seq += 1
        else: sell_seq = 0
        df.iloc[i, df.columns.get_loc('Sell_Setup')] = sell_seq
        
        if buy_seq == 9:
            active_buy = True; buy_cd = 0; buy_idxs = []
            active_sell = False
        if sell_seq == 9:
            active_sell = True; sell_cd = 0; sell_idxs = []
            active_buy = False

        if active_buy:
            if closes[i] <= lows[i-2]:
                buy_cd += 1; buy_idxs.append(i)
                df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_cd
                if buy_cd == 13:
                    if len(buy_idxs) >= 8 and lows[i] <= closes[buy_idxs[7]]:
                        df.iloc[i, df.columns.get_loc('Buy_13_Perfected')] = True
                    active_buy = False

        if active_sell:
            if closes[i] >= highs[i-2]:
                sell_cd += 1; sell_idxs.append(i)
                df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_cd
                if sell_cd == 13:
                    if len(sell_idxs) >= 8 and highs[i] >= closes[sell_idxs[7]]:
                        df.iloc[i, df.columns.get_loc('Sell_13_Perfected')] = True
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

        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df = calculate_demark(df)
        
        df_weekly = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        df_weekly = calculate_demark(df_weekly)
        
        last_d = df.iloc[-1]
        last_w = df_weekly.iloc[-1]
        price = last_d['Close']
        
        daily_sig = None; weekly_sig = None
        
        # Daily
        if last_d['Buy_Countdown'] == 13: daily_sig = "BUY 13"
        elif last_d['Sell_Countdown'] == 13: daily_sig = "SELL 13"
        elif last_d['Buy_Setup'] == 9: daily_sig = "BUY 9"
        elif last_d['Sell_Setup'] == 9: daily_sig = "SELL 9"
        
        # Weekly
        if last_w['Buy_Countdown'] == 13: weekly_sig = "BUY 13"
        elif last_w['Sell_Countdown'] == 13: weekly_sig = "SELL 13"
        elif last_w['Buy_Setup'] == 9: weekly_sig = "BUY 9"
        elif last_w['Sell_Setup'] == 9: weekly_sig = "SELL 9"
        
        if not daily_sig and not weekly_sig: return None
        
        # Perfection Check
        d_perf = last_d.get('Buy_13_Perfected') or last_d.get('Sell_13_Perfected') or False
        if daily_sig and '9' in daily_sig:
            if 'BUY' in daily_sig:
                l9, l8, l7, l6 = df['Low'].iloc[-1], df['Low'].iloc[-2], df['Low'].iloc[-3], df['Low'].iloc[-4]
                d_perf = (l9 < l7 and l9 < l6) or (l8 < l7 and l8 < l6)
            else:
                h9, h8, h7, h6 = df['High'].iloc[-1], df['High'].iloc[-2], df['High'].iloc[-3], df['High'].iloc[-4]
                d_perf = (h9 > h7 and h9 > h6) or (h8 > h7 and h8 > h6)

        return {
            'ticker': ticker, 'price': price,
            'daily': daily_sig, 'weekly': weekly_sig, 'perfected': d_perf
        }
    except: return None

# ==========================================================
#  MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("1. Generating Macro Report...")
    howell = get_michael_howell_update()
    btc_macro = get_btc_macro_sensitivity()
    bitcoin = get_bitcoin_layer_update()
    rv = get_real_vision_update()
    
    macro_msg = f"🌍 **GLOBAL MACRO INSIGHTS** 🌍\n\n{howell}\n\n{btc_macro}\n\n{bitcoin}\n\n{rv}\n───────────────"
    send_telegram_alert(macro_msg)
    
    print("2. Fetching Ticker Universe...")
    top_crypto = get_top_200_cryptos()
    top_futures = get_top_futures()
    sp500 = get_sp500_tickers()
    nasdaq = get_nasdaq_tickers()
    
    # FIXED: Added top_futures to the list
    full_universe = list(set(STRATEGIC_TICKERS + top_crypto + top_futures + sp500 + nasdaq))
    print(f"Scanning {len(full_universe)} tickers...")
    
    dual_signals = []
    weekly_signals = []
    daily_signals = []
    
    for i, ticker in enumerate(full_universe):
        if i % 100 == 0: print(f"Processing {i}/{len(full_universe)}...")
        res = analyze_ticker(ticker)
        if res:
            if res['daily'] and res['weekly']: dual_signals.append(res)
            elif res['weekly']: weekly_signals.append(res)
            elif res['daily']: daily_signals.append(res)
        time.sleep(0.01)
        
    if not dual_signals and not weekly_signals and not daily_signals:
        print("No signals.")
    else:
        msg = "🔔 **DEMARK DUAL-TIMEFRAME ALERTS** 🔔\n"
        
        if dual_signals:
            msg += "\n🚨 **INSTITUTIONAL CONVICTION (Daily + Weekly)** 🚨\n"
            for s in dual_signals:
                icon = "🟢" if "BUY" in s['daily'] else "🔴"
                msg += f"{icon} **{s['ticker']}**: ${s['price']:.2f}\n   └ 📅 W: {s['weekly']} | ⚡ D: {s['daily']} ({'⭐' if s['perfected'] else ''})\n"

        if weekly_signals:
            msg += "\n📅 **WEEKLY SIGNALS (Major Trend)**\n"
            for s in weekly_signals:
                icon = "🟢" if "BUY" in s['weekly'] else "🔴"
                msg += f"{icon} **{s['ticker']}**: {s['weekly']} @ ${s['price']:.2f}\n"

        if daily_signals:
            msg += "\n⚡ **DAILY SIGNALS (Tactical)**\n"
            # Sort: 13s first
            daily_signals.sort(key=lambda x: '13' in str(x['daily']), reverse=True)
            for s in daily_signals:
                icon = "🟢" if "BUY" in s['daily'] else "🔴"
                perf = "⭐" if s['perfected'] else ""
                msg += f"{icon} **{s['ticker']}**: {s['daily']} {perf}\n"

        send_telegram_alert(msg)
    
    print("Done.")
