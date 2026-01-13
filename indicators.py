# indicators.py - Verified Institutional Math (True DeMark)
import pandas as pd
import numpy as np

def sanitize(series):
    return series.fillna(0)

def calc_ma_trend(df):
    """Trend Stack"""
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
    """Shannon Stack Status"""
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

def calc_demark_detailed(df):
    """
    True Institutional DeMark Logic:
    1. Calculates Setup (9 consecutive closes)
    2. Calculates Countdown (13 non-consecutive closes AFTER Setup 9 completes)
    3. Checks Perfection (Bar 8 or 9 vs Bar 6 and 7)
    """
    try:
        c = df['Close'].values
        h = df['High'].values
        l = df['Low'].values
        
        # 1. SETUP (9)
        buy_setup = np.zeros(len(c), dtype=int)
        sell_setup = np.zeros(len(c), dtype=int)
        
        # Vectorized Setup Count
        # Compare Close to Close 4 bars ago
        for i in range(4, len(c)):
            if c[i] < c[i-4]:
                buy_setup[i] = buy_setup[i-1] + 1
            else:
                buy_setup[i] = 0
                
            if c[i] > c[i-4]:
                sell_setup[i] = sell_setup[i-1] + 1
            else:
                sell_setup[i] = 0
                
        # 2. COUNTDOWN (13) - Starts only after a completed 9
        # This requires a state machine loop
        buy_countdown = 0
        sell_countdown = 0
        buy_setup_active = False
        sell_setup_active = False
        
        # We only care about the *current* status for the report, 
        # but we need history to calculate it.
        # Scan last 100 bars for efficiency
        lookback = 100 if len(c) > 100 else len(c)
        start_idx = len(c) - lookback
        
        for i in range(start_idx, len(c)):
            # Activate Setup?
            if buy_setup[i] == 9: buy_setup_active = True; buy_countdown = 0
            if sell_setup[i] == 9: sell_setup_active = True; sell_countdown = 0
            
            # Count 13 (Price <= Low 2 bars ago for Buy)
            if buy_setup_active and i >= 2:
                if c[i] <= l[i-2]: buy_countdown += 1
                if buy_countdown == 13: buy_setup_active = False # Reset after 13
                
            # Count 13 (Price >= High 2 bars ago for Sell)
            if sell_setup_active and i >= 2:
                if c[i] >= h[i-2]: sell_countdown += 1
                if sell_countdown == 13: sell_setup_active = False # Reset
        
        # 3. PERFECTION CHECK (Current Bar)
        last_idx = len(c) - 1
        perf = False
        
        # Buy Perfection: Low of 8 or 9 < Low of 6 and 7
        if buy_setup[last_idx] >= 9:
            # Check indices relative to current
            # If current is 9 (idx), then 8 is idx-1, etc.
            l9 = l[last_idx]; l8 = l[last_idx-1]
            l7 = l[last_idx-2]; l6 = l[last_idx-3]
            if (l9 < l7 and l9 < l6) or (l8 < l7 and l8 < l6):
                perf = True
                
        # Sell Perfection: High of 8 or 9 > High of 6 and 7
        if sell_setup[last_idx] >= 9:
            h9 = h[last_idx]; h8 = h[last_idx-1]
            h7 = h[last_idx-2]; h6 = h[last_idx-3]
            if (h9 > h7 and h9 > h6) or (h8 > h7 and h8 > h6):
                perf = True

        # Current Status Return
        bs_curr = buy_setup[-1]
        ss_curr = sell_setup[-1]
        
        # Decide which to report (whichever is active or higher)
        if bs_curr > 0: 
            return {'type': 'Buy', 'count': bs_curr, 'countdown': buy_countdown, 'perf': perf}
        elif ss_curr > 0:
            return {'type': 'Sell', 'count': ss_curr, 'countdown': sell_countdown, 'perf': perf}
        else:
            # If currently 0, but we have a countdown running
            if buy_countdown > 0: return {'type': 'Buy', 'count': 0, 'countdown': buy_countdown, 'perf': False}
            if sell_countdown > 0: return {'type': 'Sell', 'count': 0, 'countdown': sell_countdown, 'perf': False}
            
        return {'type': 'Neutral', 'count': 0, 'countdown': 0, 'perf': False}
        
    except: return {'type': 'Error', 'count': 0, 'countdown': 0, 'perf': False}

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
        return sanitize(dx.ewm(alpha=1/period, min_periods=period).mean())
    except: return pd.Series([0]*len(df), index=df.index)

def calc_hv(df):
    try:
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        return sanitize(log_returns.rolling(20).std() * np.sqrt(252) * 100)
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
