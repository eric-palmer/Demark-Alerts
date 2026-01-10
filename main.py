import yfinance as yf
import pandas as pd
import requests
import os
import sys
import time
import io
import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION (INSTITUTIONAL WATCHLIST) ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']

STRATEGIC_TICKERS = [
    # -- Meme / PolitiFi --
    'DJT', 'DOGE-USD', 'SHIB-USD', 'PEPE-USD',
    
    # -- Crypto: Coins --
    'BTC-USD', 'ETH-USD', 'SOL-USD',
    
    # -- Crypto: Miners --
    'BTDR', 'MARA', 'RIOT', 'HUT', 'CLSK', 'IREN', 'CIFR', 'BTBT',
    'WYFI', 'CORZ', 'CRWV', 'APLD', 'NBIS', 'WULF', 'HIVE', 'BITF',
    
    # -- Crypto: ETFs --
    'IBIT', 'ETHA', 'BITQ', 'BSOL', 'GSOL', 'SOLT',
    'MSTR', 'COIN', 'HOOD', 'GLXY', 'STKE', 'DFDV', 'NODE', 'GEMI', 'BLSH',
    'CRCL',
    
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

CRYPTO_MAP = {
    'BTC-USD': 'bitcoin',
    'ETH-USD': 'ethereum',
    'SOL-USD': 'solana',
    'DOGE-USD': 'dogecoin',
    'SHIB-USD': 'shiba-inu',
    'PEPE-USD': 'pepe',
}

# --- HELPER FUNCTIONS ---
def send_telegram_alert(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    
    max_len = 4000
    for i in range(0, len(message), max_len):
        chunk = message[i:i+max_len]
        try:
            response = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"})
            if not response.ok:
                print(f"Telegram error: {response.json()}")
            time.sleep(1)
        except Exception as e:
            print(f"Telegram send error: {e}")

def format_price(price):
    if price is None: return "N/A"
    if price < 0.01: return f"${price:.6f}"
    if price < 1.00: return f"${price:.4f}"
    return f"${price:.2f}"

# --- DATA FETCHERS (CRASH PROOF + RETRY) ---
def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br'
    })
    return session

def safe_download(ticker, retries=5):
    session = get_session()
    for i in range(retries):
        try:
            df = yf.download(ticker, period="2y", progress=False, auto_adjust=True, session=session)
            if df.empty or len(df) < 50: 
                time.sleep(2); continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            print(f"yfinance error for {ticker}: {e}")
            time.sleep(2)
    
    # Fallback for crypto
    if ticker in CRYPTO_MAP:
        id_ = CRYPTO_MAP[ticker]
        try:
            url_ohlc = f"https://api.coingecko.com/api/v3/coins/{id_}/ohlc?vs_currency=usd&days=730"
            resp_ohlc = session.get(url_ohlc, timeout=10)
            if resp_ohlc.status_code != 200:
                raise ValueError(f"CoinGecko OHLC error: {resp_ohlc.text}")
            data_ohlc = resp_ohlc.json()
            df = pd.DataFrame(data_ohlc, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            url_vol = f"https://api.coingecko.com/api/v3/coins/{id_}/market_chart?vs_currency=usd&days=730&interval=daily"
            resp_vol = session.get(url_vol, timeout=10)
            if resp_vol.status_code != 200:
                raise ValueError(f"CoinGecko volume error: {resp_vol.text}")
            data_vol = resp_vol.json()
            timestamps_vol = [v[0] for v in data_vol['total_volumes']]
            volumes = [v[1] for v in data_vol['total_volumes']]
            df_vol = pd.DataFrame({'Volume': volumes}, index=pd.to_datetime(timestamps_vol, unit='ms'))
            
            df = df.join(df_vol, how='left')
            if len(df) < 50:
                return None
            return df
        except Exception as e:
            print(f"CoinGecko fallback error for {ticker}: {e}")
    
    return None

def get_top_futures():
    return ['ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F', 'FTSE=F', 'CL=F', 'NG=F', 'RB=F', 'HO=F', 'BZ=F', 'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'ZT=F', 'ZF=F', 'ZN=F', 'ZB=F', '6E=F', '6B=F', '6J=F', '6A=F', 'DX-Y.NYB', 'ZC=F', 'ZS=F', 'ZW=F', 'ZL=F', 'ZM=F', 'CC=F', 'KC=F', 'SB=F', 'CT=F', 'LE=F', 'HE=F']

def fetch_fred_series(series_id, start_date, session, retries=5):
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {
        'id': series_id,
        'cosd': start_date.strftime('%Y-%m-%d'),
        'coed': datetime.datetime.now().strftime('%Y-%m-%d')
    }
    for attempt in range(retries):
        try:
            response = session.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), parse_dates=['DATE'], index_col='DATE')
            df = df[series_id].to_frame(name=series_id)
            df = df.replace('.', np.nan).astype(float)
            return df
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            time.sleep(2)
    return pd.DataFrame()

def get_shared_macro_data():
    try:
        start_date = datetime.datetime.now() - datetime.timedelta(days=730)
        series_list = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'DGS10', 'DGS2', 'T5YIE']
        session = get_session()
        
        fred_dfs = []
        for series in series_list:
            df = fetch_fred_series(series, start_date, session)
            if not df.empty:
                fred_dfs.append(df)
        
        if not fred_dfs:
            raise ValueError("All FRED series failed to fetch")
        
        fred = pd.concat(fred_dfs, axis=1)
        fred = fred.resample('D').ffill().dropna()
        
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        term_premia = fred['DGS10'] - fred['DGS2']
        spy = safe_download('SPY')
        if spy is None: raise ValueError("SPY data unavailable")
        
        return {'net_liq': net_liq, 'term_premia': term_premia, 'inflation': fred['T5YIE'], 'fed_assets': fred['WALCL'] / 1000, 'spy': spy['Close']}
    except Exception as e:
        print(f"Macro fetch error: {e}")
        return None

# --- INDICATOR LIBRARY ---
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
        sma = df['Close'].rolling(20).mean(); std = df['Close'].rolling(20).std()
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
        fibs = {0.382: high - (high-low)*0.382, 0.618: high - (high-low)*0.618}
        
        for level, val in fibs.items():
            if abs(price - val)/price < 0.01:
                action = "BOUNCE" if price > val else "REJECTION"
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

def calc_shannon(df):
    try:
        # 5-Day EMA (Momentum)
        df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        # 10, 20, 50 SMA (Trend)
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        # AVWAP (Approx Year-to-Date using full dataframe as proxy for anchor)
        df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VP'] = df['Typical'] * df['Volume']
        df['Total_VP'] = df['VP'].cumsum()
        df['Total_Vol'] = df['Volume'].cumsum()
        df['AVWAP'] = df['Total_VP'] / df['Total_Vol']
        
        last = df.iloc[-1]
        near_term = "BULLISH" if last['Close'] > last['EMA5'] else "BEARISH"
        
        # Breakout Signal: 10 crosses 20 while > 50
        prev = df.iloc[-2]
        breakout = (prev['SMA10'] < prev['SMA20']) and (last['SMA10'] > last['SMA20']) and (last['Close'] > last['SMA50'])
        
        return {'near_term': near_term, 'avwap': last['AVWAP'], 'breakout': breakout}
    except: return None

def calc_macd(df, fast=12, slow=26, signal=9):
    try:
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    except:
        return pd.Series([0]*len(df)), pd.Series([0]*len(df)), pd.Series([0]*len(df))

def calc_stochastic(df, k_period=14, d_period=3):
    try:
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        k_line = 100 * (df['Close'] - low_min) / (high_max - low_min)
        d_line = k_line.rolling(window=d_period).mean()
        return k_line, d_line
    except:
        return pd.Series([50]*len(df)), pd.Series([50]*len(df))

def calc_adx(df, period=14):
    try:
        high_diff = df['High'] - df['High'].shift(1)
        low_diff = df['Low'].shift(1) - df['Low']
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    except:
        return pd.Series([0]*len(df)), pd.Series([0]*len(df)), pd.Series([0]*len(df))

# --- ANALYSIS ENGINE ---
def analyze_ticker(ticker, is_portfolio=False):
    try:
        df = safe_download(ticker)
        
        # Filter Ghost Data
        if df is None: 
            print(f"Data download failed for {ticker}")
            return None
        if not is_portfolio and (df['Volume'].iloc[-5:].sum() == 0 or df['Close'].iloc[-1] < 0.00000001): return None
        
        df_weekly = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        if len(df_weekly) < 20: return None
        
        # Calc All Indicators
        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        df_weekly = calc_demark(df_weekly)
        
        d_sq = calc_squeeze(df); w_sq = calc_squeeze(df_weekly)
        fib = calc_fib(df)
        shannon = calc_shannon(df)
        
        macd_line, signal_line, macd_hist = calc_macd(df)
        sto_k, sto_d = calc_stochastic(df)
        adx, plus_di, minus_di = calc_adx(df)
        
        last_d = df.iloc[-1]; last_w = df_weekly.iloc[-1]
        price = last_d['Close']
        
        # Targets (ATR)
        tr = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        p_target = price + (atr*2)
        p_stop = price - (atr*1.5)
        
        # --- DEMARK LOGIC ---
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

        # --- RSI LOGIC ---
        rsi_sig = None
        if last_d['RSI'] < 30: rsi_sig = {'type': 'OVERSOLD', 'val': last_d['RSI']}
        elif last_d['RSI'] > 70: rsi_sig = {'type': 'OVERBOUGHT', 'val': last_d['RSI']}
        
        # --- SQUEEZE LOGIC ---
        sq_sig = None
        if d_sq: sq_sig = {'tf': 'Daily', 'bias': d_sq['bias'], 'move': d_sq['move']}
        elif w_sq: sq_sig = {'tf': 'Weekly', 'bias': w_sq['bias'], 'move': w_sq['move']}
        
        # --- MACD LOGIC ---
        macd_sig = None
        if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
            macd_sig = {'type': 'BULLISH CROSS'}
        elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
            macd_sig = {'type': 'BEARISH CROSS'}
        
        # --- STOCHASTIC LOGIC ---
        sto_sig = None
        if sto_k.iloc[-1] > sto_d.iloc[-1] and sto_k.iloc[-2] <= sto_d.iloc[-2] and sto_k.iloc[-1] < 20:
            sto_sig = {'type': 'BULLISH CROSS OVERSOLD'}
        elif sto_k.iloc[-1] < sto_d.iloc[-1] and sto_k.iloc[-2] >= sto_d.iloc[-2] and sto_k.iloc[-1] > 80:
            sto_sig = {'type': 'BEARISH CROSS OVERBOUGHT'}
        
        # --- ADX LOGIC ---
        adx_sig = None
        if adx.iloc[-1] > 25:
            dir_ = 'UPTREND' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'DOWNTREND'
            adx_sig = {'strength': 'STRONG', 'direction': dir_}
        
        # --- PORTFOLIO SUMMARY ---
        trend = "BULLISH" if price > df['Close'].rolling(200).mean().iloc[-1] else "BEARISH"
        
        verdict = "HOLD"
        if dm_sig and "BUY" in dm_sig['type']: verdict = "BUY (Signal)"
        elif dm_sig and "SELL" in dm_sig['type']: verdict = "SELL (Signal)"
        elif trend == "BULLISH" and shannon['near_term'] == "BULLISH": verdict = "BUY (Trend)"
        elif trend == "BEARISH" and shannon['near_term'] == "BEARISH": verdict = "SELL (Trend)"
        
        return {
            'ticker': ticker, 'price': price,
            'demark': dm_sig, 'rsi': rsi_sig, 'squeeze': sq_sig, 'fib': fib, 'shannon': shannon,
            'macd': macd_sig, 'stochastic': sto_sig, 'adx': adx_sig,
            'trend': trend, 'verdict': verdict, 'target': p_target, 'stop': p_stop,
            'rsi_val': last_d['RSI'],
            'count': f"Buy {last_d['Buy_Setup']}" if last_d['Buy_Setup'] > 0 else f"Sell {last_d['Sell_Setup']}"
        }
    except Exception as e:
        print(f"Analysis error for {ticker}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("1. Generating Macro...")
    macro = get_shared_macro_data()
    msg = "🌍 **GLOBAL MACRO REPORT** 🌍\n\n"
    
    if macro:
        liq_momo = macro['net_liq'].pct_change(63).iloc[-1]
        regime = "RISK ON 🟢" if liq_momo > 0 else "RISK OFF 🔴"
        msg += f"📊 **MARKET RADAR**: {regime}\n   └ Liq Trend: {liq_momo*100:.2f}%\n"
        
        roc_med = macro['net_liq'].pct_change(63).iloc[-1]
        accel = macro['net_liq'].pct_change(63).diff(20).iloc[-1]
        phase = "REBOUND" if roc_med > 0 and accel > 0 else "TURBULENCE"
        msg += f"🏛️ **CAPITAL WARS**: {phase}\n\n"
    else:
        msg += "⚠️ Macro Data Unavailable\n\n"
        
    send_telegram_alert(msg)
    
    # --- PORTFOLIO ---
    print("2. Portfolio Analysis...")
    p_msg = "💼 **CURRENT PORTFOLIO** 💼\n"
    for t in CURRENT_PORTFOLIO:
        res = analyze_ticker(t, is_portfolio=True)
        if res:
            p = format_price(res['price']); tgt = format_price(res['target']); stp = format_price(res['stop'])
            p_msg += f"🔹 **{t}**: {res['verdict']} @ {p}\n   └ 🎯 Target: {tgt} | 🛑 Stop: {stp}\n   └ ⏳ Timing: 1-4 Weeks (Swing)\n   └ 📊 Techs: {res['trend']} | RSI: {res['rsi_val']:.0f} | 5-EMA: {res['shannon']['near_term']}\n   └ DeMark: {res['count']}\n"
            if res['demark']: p_msg += f"   🚨 SIGNAL: {res['demark']['type']} ({'Perf' if res['demark'].get('perfected') else 'Unperf'})\n"
            if res['squeeze']: p_msg += f"   ⚠️ SQUEEZE: {res['squeeze']['tf']} ({res['squeeze']['bias']})\n"
            if res['macd']: p_msg += f"   📈 MACD: {res['macd']['type']}\n"
            if res['stochastic']: p_msg += f"   📉 Stochastic: {res['stochastic']['type']}\n"
            if res['adx']: p_msg += f"   💪 ADX: {res['adx']['strength']} {res['adx']['direction']}\n"
            p_msg += "\n"
    send_telegram_alert(p_msg)
    
    # --- MARKET SCAN ---
    print("3. Scanning Universe...")
    universe = list(set(STRATEGIC_TICKERS + get_top_futures()))
    power = []; perfected = []; unperf = []; sq_list = []; fib_list = []; shannon_list = []
    macd_list = []; sto_list = []; adx_list = []
    
    def process_ticker(t):
        return analyze_ticker(t)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(process_ticker, t): t for t in universe}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if not res: continue
            
            d = res['demark']
            score = 0
            if d and d.get('perfected'): score += 2  # Higher weight for perfected DeMark
            if res['rsi']: score += 1
            if res['squeeze']: score += 1
            if res['fib']: score += 1
            if res['shannon']['breakout']: score += 1
            if res['macd']: score += 1
            if res['stochastic']: score += 1
            if res['adx']: score += 1
            
            if score >= 3: power.append(res)  # Raised threshold for more conviction
            
            if d:
                if d.get('perfected'): perfected.append(res)
                else: unperf.append(res)
            if res['squeeze']: sq_list.append(res)
            if res['fib']: fib_list.append(res)
            if res['shannon']['breakout']: shannon_list.append(res)
            if res['macd']: macd_list.append(res)
            if res['stochastic']: sto_list.append(res)
            if res['adx']: adx_list.append(res)
    
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
            a_msg += f"🟢 **{s['ticker']}**: {d['type']} ({d['tf']}) @ {format_price(s['price'])}\n   └ Target: {format_price(s['target'])}\n"

    if shannon_list:
        a_msg += "\n🌊 **ALPHATRENDS (Momentum Breakouts)**\n"
        for s in shannon_list[:10]:
            if s in power: continue
            a_msg += f"🔵 **{s['ticker']}**: 10/20 SMA Cross (Bullish)\n   └ Above 50 SMA | 5-EMA Support\n"

    if macd_list:
        a_msg += "\n📈 **MACD CROSSOVERS**\n"
        for s in macd_list[:10]:
            if s in power: continue
            m = s['macd']
            a_msg += f"⚡ **{s['ticker']}**: {m['type']} @ {format_price(s['price'])}\n"

    if sto_list:
        a_msg += "\n📉 **STOCHASTIC SIGNALS**\n"
        for s in sto_list[:10]:
            if s in power: continue
            st = s['stochastic']
            a_msg += f"🌀 **{s['ticker']}**: {st['type']} @ {format_price(s['price'])}\n"

    if adx_list:
        a_msg += "\n💪 **STRONG TRENDS (ADX)**\n"
        for s in adx_list[:10]:
            if s in power: continue
            a = s['adx']
            a_msg += f"📊 **{s['ticker']}**: {a['strength']} {a['direction']} @ {format_price(s['price'])}\n"

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