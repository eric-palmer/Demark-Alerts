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

# ==========================================================
#  ENGINE 1: BTC MACRO SENSITIVITY (Capital Flows)
# ==========================================================
def get_btc_macro_sensitivity():
    """
    Capital Flows Framework:
    Bitcoin reacts to:
    1. Inflation Swaps (Proxy: 5Y Breakeven Inflation 'T5YIE')
    2. Term Premia (Proxy: Yield Curve Slope 10Y-2Y or ACM Term Premia)
    3. Net Liquidity Velocity
    """
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=400)
        
        # FRED Data: 5Y Breakeven (Inflation) & 10Y Yield
        fred = web.DataReader(['T5YIE', 'DGS10', 'DGS2'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        
        # 1. Inflation Expectations (The "Debasement" Trade)
        inflation_exp = fred['T5YIE'].iloc[-1]
        inflation_trend = fred['T5YIE'].pct_change(20).iloc[-1] # Monthly trend
        
        # 2. Term Premia Proxy (Steepener vs Flattener)
        # Rising Term Premia (Steepening) often hurts risk assets initially unless driven by inflation
        term_premia_proxy = fred['DGS10'].iloc[-1] - fred['DGS2'].iloc[-1]
        tp_trend = "RISING (Bearish Duration)" if term_premia_proxy > fred['DGS10'].iloc[-20] - fred['DGS2'].iloc[-20] else "FALLING (Bullish Duration)"
        
        # 3. Bitcoin Signal
        # BTC loves: Rising Inflation Exp + Stable/Falling Real Rates (Liquidity)
        btc_signal = ""
        if inflation_trend > 0:
            btc_signal = "🟢 **BULLISH** (Inflation Expectations Rising)"
        elif inflation_trend < 0:
            btc_signal = "🔴 **BEARISH** (Disinflation / Cooling)"
        else:
            btc_signal = "⚪ **NEUTRAL**"
            
        return f"🪙 **BITCOIN MACRO SENSITIVITY**\n   └ **Signal:** {btc_signal}\n   └ **Inflation Swaps (5Y):** {inflation_exp:.2f}% ({'Up' if inflation_trend > 0 else 'Down'})\n   └ **Term Premia:** {tp_trend}"
    
    except Exception as e: return f"⚠️ BTC Macro Error: {e}"

# ==========================================================
#  ENGINE 2: MICHAEL HOWELL (Dynamic Cycle)
# ==========================================================
def get_michael_howell_update():
    """
    Dynamic Liquidity Cycle (No Hardcoding).
    Based on Rate of Change (RoC) of Global Liquidity Proxy.
    """
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=500)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        
        # Momentum (63-day RoC)
        roc_med = net_liq.pct_change(63).iloc[-1]
        
        # Acceleration (Is the RoC increasing or decreasing?)
        roc_prev = net_liq.pct_change(63).iloc[-20]
        acceleration = roc_med - roc_prev
        
        # Dynamic Phase Determination
        phase = ""
        action = ""
        
        if roc_med > 0 and acceleration > 0:
            phase = "REBOUND (Accelerating)"
            action = "Overweight: Tech / Crypto / High Beta"
        elif roc_med > 0 and acceleration < 0:
            phase = "SPECULATION (Peaking / Slowing)"
            action = "Overweight: Commodities / Energy / 5Y Bonds"
        elif roc_med < 0 and acceleration < 0:
            phase = "TURBULENCE (Contraction)"
            action = "Overweight: Cash / Gold / Volatility"
        elif roc_med < 0 and acceleration > 0:
            phase = "CALM (Bottoming)"
            action = "Accumulate: Credit / Quality Equities"
            
        return f"🏛️ **CAPITAL WARS (Michael Howell)**\n   └ **Phase:** {phase}\n   └ **Trend:** {'Expanding' if roc_med > 0 else 'Contracting'} ({roc_med*100:.2f}%)\n   └ **Action:** {action}"
    except: return "⚠️ Howell Engine Error"

# ==========================================================
#  ENGINE 3: THE BITCOIN LAYER (Liquidity Velocity)
# ==========================================================
def get_bitcoin_layer_update():
    """
    Focus: Velocity (Acceleration) of Liquidity.
    BTC follows the 2nd Derivative.
    """
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=400)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        
        # Velocity Calculation (2nd Derivative)
        velocity = net_liq.diff().diff().rolling(window=10).mean().iloc[-1]
        
        signal = ""
        if velocity > 0:
            signal = "🟢 HIGH VELOCITY (Impulse Up)"
        else:
            signal = "🟡 LOW VELOCITY (Stalling/Drag)"
            
        return f"🟠 **THE BITCOIN LAYER**\n   └ **Velocity:** {signal}\n   └ **Metric:** 2nd Derivative of Net Liq"
    except: return "⚠️ Bitcoin Layer Error"

# ==========================================================
#  ENGINE 4: REAL VISION (The Everything Code)
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
        
        regime = ""
        if liq_momo > 0 and growth_momo <= 0:
            regime = "🍌 **BANANA ZONE** (Liq UP / Growth DOWN)"
        elif liq_momo > 0 and growth_momo > 0:
            regime = "🟢 **SPRING/SUMMER** (Risk On)"
        elif liq_momo < 0 and growth_momo < 0:
            regime = "🍂 **FALL** (Slowdown)"
        else:
            regime = "❄️ **WINTER** (Risk Off)"
            
        return f"🧠 **REAL VISION**\n   └ **Regime:** {regime}"
    except: return "⚠️ Real Vision Error"

# ==========================================================
#  ENGINE 5: DEMARK STOCK SCANNER
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
        last = df.iloc[-1]
        price = last['Close']
        sma = last['SMA_200']
        
        signal = None
        
        if last['Buy_Countdown'] == 13:
            signal = {
                'ticker': ticker, 'type': 'BUY', 'algo': 'COUNTDOWN 13',
                'price': price, 'perfected': last['Buy_13_Perfected'],
                'action': 'ACCUMULATE', 'timing': 'Weeks',
                'stop': min(df['Low'].iloc[-13:]), 'target': price * 1.15,
                'trend': 'Bullish' if price > sma else 'Bearish'
            }
        elif last['Sell_Countdown'] == 13:
            signal = {
                'ticker': ticker, 'type': 'SELL', 'algo': 'COUNTDOWN 13',
                'price': price, 'perfected': last['Sell_13_Perfected'],
                'action': 'DISTRIBUTE', 'timing': 'Weeks',
                'stop': max(df['High'].iloc[-13:]), 'target': price * 0.85,
                'trend': 'Bullish' if price > sma else 'Bearish'
            }
        elif last['Buy_Setup'] == 9:
            stop = min(df['Low'].iloc[-9:])
            risk = max(price - stop, price * 0.01)
            l9, l8, l7, l6 = df['Low'].iloc[-1], df['Low'].iloc[-2], df['Low'].iloc[-3], df['Low'].iloc[-4]
            perf = (l9 < l7 and l9 < l6) or (l8 < l7 and l8 < l6)
            
            signal = {
                'ticker': ticker, 'type': 'BUY', 'algo': 'SETUP 9',
                'price': price, 'perfected': perf,
                'action': 'BOUNCE', 'timing': '1-4 Days',
                'stop': stop, 'target': price + (risk * 2),
                'trend': 'Bullish' if price > sma else 'Bearish'
            }
        elif last['Sell_Setup'] == 9:
            stop = max(df['High'].iloc[-9:])
            risk = max(stop - price, price * 0.01)
            h9, h8, h7, h6 = df['High'].iloc[-1], df['High'].iloc[-2], df['High'].iloc[-3], df['High'].iloc[-4]
            perf = (h9 > h7 and h9 > h6) or (h8 > h7 and h8 > h6)
            
            signal = {
                'ticker': ticker, 'type': 'SELL', 'algo': 'SETUP 9',
                'price': price, 'perfected': perf,
                'action': 'PULLBACK', 'timing': '1-4 Days',
                'stop': stop, 'target': price - (risk * 2),
                'trend': 'Bullish' if price > sma else 'Bearish'
            }
        return signal
    except: return None

# ==========================================================
#  MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("1. Generating Macro Report...")
    btc_macro = get_btc_macro_sensitivity()
    howell = get_michael_howell_update()
    bitcoin = get_bitcoin_layer_update()
    rv = get_real_vision_update()
    
    macro_msg = f"🌍 **GLOBAL MACRO INSIGHTS** 🌍\n\n{howell}\n\n{btc_macro}\n\n{bitcoin}\n\n{rv}\n───────────────"
    send_telegram_alert(macro_msg)
    
    print("2. Fetching Ticker Universe...")
    top_crypto = get_top_200_cryptos()
    sp500 = get_sp500_tickers()
    nasdaq = get_nasdaq_tickers()
    
    full_universe = list(set(STRATEGIC_TICKERS + top_crypto + sp500 + nasdaq))
    print(f"Scanning {len(full_universe)} tickers...")
    
    signals = []
    for i, ticker in enumerate(full_universe):
        if i % 100 == 0: print(f"Processing {i}/{len(full_universe)}...")
        res = analyze_ticker(ticker)
        if res: signals.append(res)
        time.sleep(0.01)
        
    if signals:
        signals.sort(key=lambda x: (x['algo'] == 'SETUP 9', x['type']))
        
        stock_msg = "🔔 **DEMARK SIGNALS (INSTITUTIONAL)** 🔔\n"
        for s in signals:
            icon = "🟢" if "BUY" in s['type'] else "🔴"
            perf = "⭐" if s['perfected'] else "⚠️"
            stock_msg += f"{icon} **{s['ticker']}** [{s['algo']}] {perf}\n"
            stock_msg += f"   ⚡ {s['action']} ({s['trend']})\n"
            stock_msg += f"   🎯 ${s['target']:.2f} | 🛑 ${s['stop']:.2f}\n"
            stock_msg += f"   ⏳ {s['timing']}\n"
            stock_msg += "───────────────\n"
            
        send_telegram_alert(stock_msg)
    else:
        print("No stock signals found.")
    
    print("Done.")
