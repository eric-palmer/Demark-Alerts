# indicators.py - Verified Institutional Math
import pandas as pd
import numpy as np

def sanitize(series):
    return series.fillna(0)

def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Wilder's Smoothing (Standard)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return sanitize(rsi.fillna(50))
    except: return pd.Series([50]*len(series), index=series.index)

def calc_squeeze(df):
    try:
        if len(df) < 20: return None
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        upper_bb = sma + (std * 2)
        lower_bb = sma - (std * 2)
        
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        
        upper_kc = sma + (atr * 1.5)
        lower_kc = sma - (atr * 1.5)
        
        in_squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        if not in_squeeze.iloc[-1]: return None
        
        # Momentum Slope (Linear Reg)
        y = df['Close'].iloc[-20:].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return {'bias': "BULLISH" if slope > 0 else "BEARISH", 'tf': 'Daily'}
    except: return None

def calc_demark(df):
    """
    DeMark Sequential - Strict Institutional Rules
    Setup: 9 consecutive closes >/< close 4 bars ago.
    Perfection: 
      - Buy: Low of bar 8 OR 9 < Low of bars 6 AND 7.
      - Sell: High of bar 8 OR 9 > High of bars 6 AND 7.
    """
    try:
        df = df.copy()
        c = df['Close'].values
        # Shift 4 bars back (Numpy Roll)
        c_4 = np.roll(c, 4); c_4[:4] = c[:4] 
        
        buy_setup = (c < c_4)
        sell_setup = (c > c_4)
        
        def count(condition):
            s = pd.Series(condition)
            return s.groupby((s != s.shift()).cumsum()).cumsum() * s
            
        df['Buy_Setup'] = count(buy_setup)
        df['Sell_Setup'] = count(sell_setup)
        
        # Perfection Check (Look back safely)
        last = len(df) - 1
        perf = False
        
        if last > 10:
            # Indices: last=9, last-1=8, last-2=7, last-3=6
            if df['Buy_Setup'].iloc[-1] == 9:
                lows = df['Low'].values
                # Check Bar 9 vs 6&7 OR Bar 8 vs 6&7
                low_9_valid = (lows[last] < lows[last-2] and lows[last] < lows[last-3])
                low_8_valid = (lows[last-1] < lows[last-2] and lows[last-1] < lows[last-3])
                if low_9_valid or low_8_valid:
                    perf = True
                    
            elif df['Sell_Setup'].iloc[-1] == 9:
                highs = df['High'].values
                high_9_valid = (highs[last] > highs[last-2] and highs[last] > highs[last-3])
                high_8_valid = (highs[last-1] > highs[last-2] and highs[last-1] > highs[last-3])
                if high_9_valid or high_8_valid:
                    perf = True
                
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
        # Breakout: 10 crosses 20, while Price > 50
        breakout = (prev['SMA10'] <= prev['SMA20'] and 
                   last['SMA10'] > last['SMA20'] and 
                   last['Close'] > last['SMA50'])
        return {'breakout': breakout}
    except: return {'breakout': False}

def calc_adx(df, period=14):
    try:
        up = df['High'].diff(); down = -df['Low'].diff()
        p_dm = np.where((up > down) & (up > 0), up, 0)
        m_dm = np.where((down > up) & (down > 0), down, 0)
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period).mean()
        p_di = 100 * (pd.Series(p_dm).ewm(alpha=1/period).mean() / atr)
        m_di = 100 * (pd.Series(m_dm).ewm(alpha=1/period).mean() / atr)
        sum_di = (p_di + m_di).replace(0, 1)
        dx = 100 * abs(p_di - m_di) / sum_di
        return sanitize(dx.ewm(alpha=1/period).mean())
    except: return pd.Series([0]*len(df), index=df.index)
