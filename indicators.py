# indicators.py - Verified Institutional Math (Fixed)
import pandas as pd
import numpy as np

def sanitize(series):
    return series.fillna(0)

def calc_ma_trend(df):
    """Calculates Moving Averages for Trend Context"""
    try:
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        # Exponential (for Shannon)
        ema8 = df['Close'].ewm(span=8, adjust=False).mean()
        ema21 = df['Close'].ewm(span=21, adjust=False).mean()
        return {'sma50': sma50, 'sma200': sma200, 'ema8': ema8, 'ema21': ema21}
    except:
        z = pd.Series([0]*len(df), index=df.index)
        return {'sma50': z, 'sma200': z, 'ema8': z, 'ema21': z}

def calc_trend_stack(df):
    """Brian Shannon's Trend Stack"""
    try:
        c = df['Close']
        ema8 = c.ewm(span=8, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        sma50 = c.rolling(50).mean()
        
        last_c = c.iloc[-1]; last_8 = ema8.iloc[-1]
        last_21 = ema21.iloc[-1]; last_50 = sma50.iloc[-1]
        
        if last_c > last_8 > last_21 > last_50: status = "Strong Uptrend (Price > 8 > 21 > 50)"
        elif last_c < last_8 < last_21 < last_50: status = "Strong Downtrend (Price < 8 < 21 < 50)"
        elif last_c > last_8: status = "Positive Momentum (Holding 8 EMA)"
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

def calc_demark(df):
    """
    Standard DeMark Setup (9) Logic
    Matches your diagnostic script exactly.
    """
    try:
        df = df.copy()
        c = df['Close'].values
        # Initialize arrays
        buy_setup = np.zeros(len(c), dtype=int)
        sell_setup = np.zeros(len(c), dtype=int)
        
        # Loop (The proven method from your test)
        for i in range(4, len(c)):
            if c[i] < c[i-4]:
                buy_setup[i] = buy_setup[i-1] + 1
            else:
                buy_setup[i] = 0
                
            if c[i] > c[i-4]:
                sell_setup[i] = sell_setup[i-1] + 1
            else:
                sell_setup[i] = 0
        
        # Perfection Check (Current Bar)
        last = len(c) - 1
        perf = False
        l = df['Low'].values
        h = df['High'].values
        
        if buy_setup[last] >= 9:
            if (l[last] < l[last-2] and l[last] < l[last-3]) or (l[last-1] < l[last-2] and l[last-1] < l[last-3]): 
                perf = True
        elif sell_setup[last] >= 9:
            if (h[last] > h[last-2] and h[last] > h[last-3]) or (h[last-1] > h[last-2] and h[last-1] > h[last-3]): 
                perf = True
                
        df['Buy_Setup'] = buy_setup
        df['Sell_Setup'] = sell_setup
        df['Perfected'] = perf
        return df
    except:
        df['Buy_Setup'] = 0; df['Sell_Setup'] = 0; df['Perfected'] = False
        return df

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
        # Use Numpy where but wrap in Series to keep index
        p_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
        m_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
        
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        tr = tr.replace(0, 0.0001)
        
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        p_di = 100 * (p_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        m_di = 100 * (m_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        
        dx = 100 * abs(p_di - m_di) / (p_di + m_di)
        # Using simple mean for smoother ADX curve like TradingView
        return sanitize(dx.rolling(window=period).mean()) 
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
