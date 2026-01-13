# indicators.py - Verified Institutional Math (Pro)
import pandas as pd
import numpy as np

def sanitize(series):
    return series.fillna(0)

def calc_fibs(df):
    """
    Auto-Fibonacci Retracements (6-Month Lookback).
    Returns levels relative to current price.
    """
    try:
        # Lookback ~126 trading days (6 months)
        lookback = 126 if len(df) > 126 else len(df)
        recent = df.iloc[-lookback:]
        
        high = recent['High'].max()
        low = recent['Low'].min()
        diff = high - low
        
        # Standard Fibs
        fib_levels = {
            '0.0% (High)': high,
            '23.6%': high - (diff * 0.236),
            '38.2%': high - (diff * 0.382),
            '50.0%': high - (diff * 0.5),
            '61.8% (Golden)': high - (diff * 0.618),
            '78.6%': high - (diff * 0.786),
            '100.0% (Low)': low
        }
        
        # Find nearest levels to current price
        price = df['Close'].iloc[-1]
        nearest = min(fib_levels.items(), key=lambda x: abs(x[1] - price))
        
        return {'levels': fib_levels, 'nearest_name': nearest[0], 'nearest_val': nearest[1]}
    except: return {'levels': {}, 'nearest_name': 'None', 'nearest_val': 0}

def calc_vol_term(df):
    """
    Volatility Term Structure (Short vs Long Term Vol).
    HV10 < HV100 = Cheap Gamma (Buy Options).
    HV10 > HV100 = Expensive Gamma (Sell Options).
    """
    try:
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        
        hv10 = log_ret.rolling(10).std() * np.sqrt(252) * 100
        hv50 = log_ret.rolling(50).std() * np.sqrt(252) * 100
        
        curr_hv10 = hv10.iloc[-1]
        curr_hv50 = hv50.iloc[-1]
        
        if curr_hv10 < curr_hv50 * 0.8:
            status = "Compression (Options Cheap)"
            strategy = "DEBIT (Long)"
        elif curr_hv10 > curr_hv50 * 1.2:
            status = "Expansion (Options Expensive)"
            strategy = "CREDIT (Short)"
        else:
            status = "Normal"
            strategy = "Spread"
            
        return {'hv10': curr_hv10, 'hv50': curr_hv50, 'status': status, 'strat': strategy}
    except: return {'hv10': 0, 'hv50': 0, 'status': 'Error', 'strat': 'Shares'}

# --- STANDARD INDICATORS (Unchanged) ---
def calc_ma_trend(df):
    try:
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        ema8 = df['Close'].ewm(span=8, adjust=False).mean()
        ema21 = df['Close'].ewm(span=21, adjust=False).mean()
        return {'sma50': sma50, 'sma200': sma200, 'ema8': ema8, 'ema21': ema21}
    except:
        z = pd.Series([0]*len(df), index=df.index)
        return {'sma50': z, 'sma200': z, 'ema8': z, 'ema21': z}

def calc_trend_stack(df):
    try:
        c = df['Close']
        ema8 = c.ewm(span=8, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        sma50 = c.rolling(50).mean()
        
        last_c = c.iloc[-1]; last_8 = ema8.iloc[-1]
        last_21 = ema21.iloc[-1]; last_50 = sma50.iloc[-1]
        
        if last_c > last_8 > last_21 > last_50: status = "Strong Uptrend (Price > 8 > 21 > 50)"
        elif last_c < last_8 < last_21 < last_50: status = "Strong Downtrend (Price < 8 < 21 < 50)"
        elif last_c > last_8: status = "Positive Momentum"
        else: status = "No Trend (Wait)"
        return {'status': status}
    except: return {'status': "Data Error"}

def calc_macd(df):
    try:
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {'macd': macd, 'signal': signal}
    except:
        z = pd.Series([0]*len(df), index=df.index)
        return {'macd': z, 'signal': z}

def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return sanitize((100 - (100 / (1 + rs))).fillna(50))
    except: return pd.Series([50]*len(series), index=series.index)

def calc_squeeze(df):
    try:
        if len(df) < 20: return None
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        upper_bb = sma + (std * 2); lower_bb = sma - (std * 2)
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        upper_kc = sma + (atr * 1.5); lower_kc = sma - (atr * 1.5)
        in_squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        if not in_squeeze.iloc[-1]: return None
        y = df['Close'].iloc[-20:].values; x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return {'bias': "BULLISH" if slope > 0 else "BEARISH", 'tf': 'Daily'}
    except: return None

def calc_demark_detailed(df):
    try:
        df = df.copy()
        c = df['Close'].values
        l = df['Low'].values
        h = df['High'].values
        
        # 1. SETUP (Vectorized)
        buy_setup = np.zeros(len(c), dtype=int)
        sell_setup = np.zeros(len(c), dtype=int)
        
        for i in range(4, len(c)):
            if c[i] < c[i-4]: buy_setup[i] = buy_setup[i-1] + 1
            else: buy_setup[i] = 0
            if c[i] > c[i-4]: sell_setup[i] = sell_setup[i-1] + 1
            else: sell_setup[i] = 0
            
        # 2. COUNTDOWN (Loop)
        buy_countdown = 0; sell_countdown = 0
        buy_active = False; sell_active = False
        start_idx = max(0, len(c) - 100)
        
        for i in range(start_idx, len(c)):
            if buy_setup[i] == 9: buy_active = True; buy_countdown = 0
            if sell_setup[i] == 9: sell_active = True; sell_countdown = 0
            
            if buy_active and i >= 2 and c[i] <= l[i-2]:
                buy_countdown += 1
                if buy_countdown == 13: buy_active = False
            
            if sell_active and i >= 2 and c[i] >= h[i-2]:
                sell_countdown += 1
                if sell_countdown == 13: sell_active = False
        
        # 3. PERFECTION
        last = len(c) - 1; perf = False
        if buy_setup[last] >= 9:
            if (l[last] < l[last-2] and l[last] < l[last-3]) or (l[last-1] < l[last-2] and l[last-1] < l[last-3]): perf = True
        elif sell_setup[last] >= 9:
            if (h[last] > h[last-2] and h[last] > h[last-3]) or (h[last-1] > h[last-2] and h[last-1] > h[last-3]): perf = True

        bs_curr = buy_setup[-1]; ss_curr = sell_setup[-1]
        
        if bs_curr > 0: return {'type': 'Buy', 'count': bs_curr, 'countdown': buy_countdown, 'perf': perf}
        elif ss_curr > 0: return {'type': 'Sell', 'count': ss_curr, 'countdown': sell_countdown, 'perf': perf}
        
        return {'type': 'Neutral', 'count': 0, 'countdown': 0, 'perf': False}
    except: return {'type': 'Error', 'count': 0, 'countdown': 0, 'perf': False}

def calc_shannon(df):
    try:
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        last = df.iloc[-1]; prev = df.iloc[-2]
        breakout = (prev['SMA10'] <= prev['SMA20'] and last['SMA10'] > last['SMA20'] and last['Close'] > last['SMA50'])
        return {'breakout': breakout}
    except: return {'breakout': False}

def calc_adx(df, period=14):
    try:
        up = df['High'].diff(); down = -df['Low'].diff()
        p_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
        m_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        tr = tr.replace(0, 0.0001)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        p_di = 100 * (p_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        m_di = 100 * (m_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        dx = 100 * abs(p_di - m_di) / (p_di + m_di)
        return sanitize(dx.rolling(window=period).mean())
    except: return pd.Series([0]*len(df), index=df.index)

def calc_rvol(df):
    try:
        vol = df['Volume']; avg = vol.rolling(20).mean()
        if avg.iloc[-1] == 0: return 1.0
        return vol.iloc[-1] / avg.iloc[-1]
    except: return 1.0

def calc_donchian(df):
    try:
        h20 = df['High'].rolling(20).max().iloc[-1]
        l10 = df['Low'].rolling(10).min().iloc[-1]
        return {'high': h20, 'low': l10}
    except: return {'high': 0, 'low': 0}
