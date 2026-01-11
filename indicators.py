# indicators.py - Robust Calculations
import pandas as pd
import numpy as np

def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)
    except:
        return pd.Series([50]*len(series), index=series.index)

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
        
        # Momentum Slope
        y = df['Close'].iloc[-20:].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        
        return {'bias': "BULLISH" if slope > 0 else "BEARISH", 'tf': 'Daily'}
    except:
        return None

def calc_demark(df):
    try:
        df = df.copy()
        c = df['Close'].values
        # Shift 4 bars back (Robust Numpy Roll)
        c_4 = np.roll(c, 4)
        c_4[:4] = c[:4] # Prevent comparison with wrapped values
        
        # 1. Setup Logic
        buy_setup = (c < c_4)
        sell_setup = (c > c_4)
        
        # 2. Vectorized Counting
        def count_consecutive(condition):
            # Cumsum resets on False
            s = pd.Series(condition)
            return s.groupby((s != s.shift()).cumsum()).cumsum() * s
            
        df['Buy_Setup'] = count_consecutive(buy_setup)
        df['Sell_Setup'] = count_consecutive(sell_setup)
        
        # 3. Perfection (Check High/Low of bar 8/9)
        # Simplified: Just check if bar 9 low < bar 6/7 low
        last = len(df) - 1
        perf = False
        
        if df['Buy_Setup'].iloc[-1] == 9:
            # Low of 9 <= Low of 6 and 7
            if df['Low'].iloc[-1] < df['Low'].iloc[-3] and df['Low'].iloc[-1] < df['Low'].iloc[-4]:
                perf = True
        elif df['Sell_Setup'].iloc[-1] == 9:
            if df['High'].iloc[-1] > df['High'].iloc[-3] and df['High'].iloc[-1] > df['High'].iloc[-4]:
                perf = True
                
        df['Perfected'] = perf
        return df
    except Exception as e:
        print(f"DM Error: {e}") # Print error to log instead of failing silent
        df['Buy_Setup'] = 0
        df['Sell_Setup'] = 0
        df['Perfected'] = False
        return df

def calc_shannon(df):
    try:
        df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        breakout = (prev['SMA10'] <= prev['SMA20'] and 
                   last['SMA10'] > last['SMA20'] and 
                   last['Close'] > last['SMA50'])
                   
        return {'breakout': breakout}
    except:
        return {'breakout': False}

def calc_macd(df):
    e12 = df['Close'].ewm(span=12, adjust=False).mean()
    e26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    return macd, sig, macd - sig

def calc_stoch(df):
    l14 = df['Low'].rolling(14).min()
    h14 = df['High'].rolling(14).max()
    k = 100 * (df['Close'] - l14) / (h14 - l14)
    return k, k.rolling(3).mean()

def calc_adx(df, period=14):
    try:
        up = df['High'].diff()
        down = -df['Low'].diff()
        p_dm = np.where((up > down) & (up > 0), up, 0)
        m_dm = np.where((down > up) & (down > 0), down, 0)
        
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        p_di = 100 * (pd.Series(p_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        m_di = 100 * (pd.Series(m_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = 100 * abs(p_di - m_di) / (p_di + m_di)
        return dx.ewm(alpha=1/period, adjust=False).mean(), p_di, m_di
    except:
        z = pd.Series([0]*len(df), index=df.index)
        return z, z, z
