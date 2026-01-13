# indicators.py - Verified Institutional Math (Multi-Timeframe)
import pandas as pd
import numpy as np

def sanitize(series):
    """Replaces NaNs with 0"""
    return series.fillna(0)

def calc_ma_trend(df):
    """Calculates 50d and 200d SMA for trend context"""
    try:
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        return {'sma50': sma50, 'sma200': sma200}
    except:
        return {'sma50': pd.Series([0]*len(df)), 'sma200': pd.Series([0]*len(df))}

def calc_macd(df):
    """Standard MACD (12, 26, 9) for Medium Term Momentum"""
    try:
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return {'macd': macd, 'signal': signal, 'hist': hist}
    except:
        z = pd.Series([0]*len(df))
        return {'macd': z, 'signal': z, 'hist': z}

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
        
        y = df['Close'].iloc[-20:].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return {'bias': "BULLISH" if slope > 0 else "BEARISH", 'tf': 'Daily'}
    except: return None

def calc_demark(df):
    try:
        df = df.copy()
        c = df['Close'].values
        c_4 = np.roll(c, 4); c_4[:4] = c[:4] 
        buy_cond = c < c_4; sell_cond = c > c_4
        
        def count_consecutive(condition_array, index):
            s = pd.Series(condition_array, index=index)
            return s.groupby((s != s.shift()).cumsum()).cumsum() * s
            
        df['Buy_Setup'] = count_consecutive(buy_cond, df.index)
        df['Sell_Setup'] = count_consecutive(sell_cond, df.index)
        
        last = len(df) - 1; perf = False
        if last > 10:
            if df['Buy_Setup'].iloc[-1] == 9:
                lows = df['Low'].values
                if (lows[last] < lows[last-2] and lows[last] < lows[last-3]) or \
                   (lows[last-1] < lows[last-2] and lows[last-1] < lows[last-3]): perf = True
            elif df['Sell_Setup'].iloc[-1] == 9:
                highs = df['High'].values
                if (highs[last] > highs[last-2] and highs[last] > highs[last-3]) or \
                   (highs[last-1] > highs[last-2] and highs[last-1] > highs[last-3]): perf = True
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
        p_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
        m_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        tr = tr.replace(0, 0.0001)
        atr = tr.ewm(alpha=1/period).mean()
        p_di = 100 * (p_dm.ewm(alpha=1/period).mean() / atr)
        m_di = 100 * (m_dm.ewm(alpha=1/period).mean() / atr)
        sum_di = (p_di + m_di).replace(0, 1)
        dx = 100 * abs(p_di - m_di) / sum_di
        return sanitize(dx.ewm(alpha=1/period).mean())
    except: return pd.Series([0]*len(df), index=df.index)
