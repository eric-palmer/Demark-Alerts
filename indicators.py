# indicators.py - Institutional Math (Shannon/Newton Upgrades)
import pandas as pd
import numpy as np

def sanitize(series):
    return series.fillna(0)

def calc_trend_stack(df):
    """
    Brian Shannon's Trend Stack:
    Checks alignment of Price, 8 EMA, 21 EMA, and 50 SMA.
    """
    try:
        c = df['Close']
        ema8 = c.ewm(span=8, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        
        # Determine State
        last_c = c.iloc[-1]
        last_8 = ema8.iloc[-1]
        last_21 = ema21.iloc[-1]
        last_50 = sma50.iloc[-1]
        
        if last_c > last_8 > last_21 > last_50:
            status = "🚀 FULL BULL (Stack Aligned)"
        elif last_c < last_8 < last_21 < last_50:
            status = "🛑 FULL BEAR (Stack Aligned)"
        elif last_c > last_8 and last_8 > last_21:
            status = "🟢 Bullish Momentum (Price > 8 > 21)"
        elif last_c < last_8:
            status = "⚠️ Weakness (Price < 8 EMA)"
        else:
            status = "⚪ Neutral/Choppy"
            
        return {
            'status': status, 
            'ema8': last_8, 'ema21': last_21, 
            'sma50': last_50, 'sma200': sma200.iloc[-1]
        }
    except: return {'status': "Error", 'ema8': 0, 'ema21': 0, 'sma50': 0, 'sma200': 0}

def calc_rvol(df):
    """Relative Volume (Institutions Voting)"""
    try:
        vol = df['Volume']
        avg_vol = vol.rolling(20).mean()
        rvol = vol.iloc[-1] / avg_vol.iloc[-1]
        return rvol
    except: return 1.0

def calc_donchian(df):
    """Dynamic Targets based on Price Structure"""
    try:
        # Target: Recent High (Liquidity Pool)
        high_20 = df['High'].rolling(20).max().iloc[-1]
        # Stop: Recent Low (Structure Break)
        low_10 = df['Low'].rolling(10).min().iloc[-1]
        return {'high_20': high_20, 'low_10': low_10}
    except: return {'high_20': 0, 'low_10': 0}

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
    try:
        df = df.copy()
        c = df['Close'].values
        c_4 = np.roll(c, 4); c_4[:4] = c[:4] 
        buy_setup = (c < c_4); sell_setup = (c > c_4)
        def count(condition):
            s = pd.Series(condition)
            return s.groupby((s != s.shift()).cumsum()).cumsum() * s
        df['Buy_Setup'] = count(buy_setup)
        df['Sell_Setup'] = count(sell_setup)
        
        # Countdown 13 Logic (Simplified)
        # Real 13 requires a complex "countdown" phase after a 9. 
        # For this bot, we check if the Setup count extends to 13 (a variation).
        
        last = len(df) - 1; perf = False
        if last > 10:
            if df['Buy_Setup'].iloc[-1] == 9:
                lows = df['Low'].values
                if (lows[last] < lows[last-2] and lows[last] < lows[last-3]) or (lows[last-1] < lows[last-2] and lows[last-1] < lows[last-3]): perf = True
            elif df['Sell_Setup'].iloc[-1] == 9:
                highs = df['High'].values
                if (highs[last] > highs[last-2] and highs[last] > highs[last-3]) or (highs[last-1] > highs[last-2] and highs[last-1] > highs[last-3]): perf = True
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
        p_dm = np.where((up > down) & (up > 0), up, 0)
        m_dm = np.where((down > up) & (down > 0), down, 0)
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        tr = tr.replace(0, 0.0001) # Fix division by zero
        atr = tr.ewm(alpha=1/period).mean()
        p_di = 100 * (pd.Series(p_dm).ewm(alpha=1/period).mean() / atr)
        m_di = 100 * (pd.Series(m_dm).ewm(alpha=1/period).mean() / atr)
        sum_di = (p_di + m_di).replace(0, 1)
        dx = 100 * abs(p_di - m_di) / sum_di
        return sanitize(dx.ewm(alpha=1/period).mean())
    except: return pd.Series([0]*len(df), index=df.index)
