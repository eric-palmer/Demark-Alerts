# indicators.py - Vectorized Institutional Indicators
import pandas as pd
import numpy as np

def calc_rsi(series, period=14):
    """Wilder's RSI (Vectorized)"""
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(series), index=series.index)

def calc_squeeze(df):
    """TTM Squeeze (Vectorized)"""
    try:
        if len(df) < 20: return None
        
        # Bollinger Bands (20, 2)
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        upper_bb = sma + (std * 2)
        lower_bb = sma - (std * 2)
        
        # Keltner Channels (20, 1.5 ATR)
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        
        upper_kc = sma + (atr * 1.5)
        lower_kc = sma - (atr * 1.5)
        
        # Squeeze Logic
        in_squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        if not in_squeeze.iloc[-1]:
            return None
            
        # Momentum Slope (Linear Regression on last 20 bars)
        y = df['Close'].iloc[-20:].values
        x = np.arange(len(y))
        # Vectorized polyfit
        slope = np.polyfit(x, y, 1)[0]
        
        return {
            'bias': "BULLISH" if slope > 0 else "BEARISH",
            'move': atr.iloc[-1] * 2,
            'tf': 'Daily'
        }
    except:
        return None

def calc_demark(df):
    """
    TD Sequential - Fully Vectorized for Speed
    """
    try:
        df = df.copy()
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        
        # --- Setup Phase (Vectorized) ---
        # Compare close to close 4 bars ago
        c_shift_4 = np.roll(close, 4)
        c_shift_4[:4] = np.nan # invalid first 4
        
        # Boolean arrays for Bullish/Bearish Price Flips
        bearish_flip = close < c_shift_4
        bullish_flip = close > c_shift_4
        
        # Vectorized Sequential Counter
        def get_sequence(condition):
            # Cumulatively sum 1s where condition is true, reset where false
            s = pd.Series(condition)
            return s.groupby((s != s.shift()).cumsum()).cumsum() * s
            
        df['Buy_Setup'] = get_sequence(bearish_flip)
        df['Sell_Setup'] = get_sequence(bullish_flip)
        
        # --- Perfection Logic (Bar 8 or 9 High/Low check) ---
        # We only calculate "Perfected" on the last row for efficiency
        last_idx = len(df) - 1
        perfected = False
        
        if df['Buy_Setup'].iloc[-1] == 9:
            # Buy Perfected: Low of 9 <= Low of 6 and 7
            lows = df['Low'].values
            if lows[last_idx] <= lows[last_idx-2] and lows[last_idx] <= lows[last_idx-3]:
                perfected = True
                
        elif df['Sell_Setup'].iloc[-1] == 9:
            # Sell Perfected: High of 9 >= High of 6 and 7
            highs = df['High'].values
            if highs[last_idx] >= highs[last_idx-2] and highs[last_idx] >= highs[last_idx-3]:
                perfected = True

        df['Perfected'] = perfected
        return df
        
    except Exception as e:
        print(f"DM Error: {e}")
        return df

def calc_shannon(df):
    """AlphaTrends Anchored VWAP Logic"""
    try:
        df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        breakout = (
            prev['SMA10'] <= prev['SMA20'] and
            last['SMA10'] > last['SMA20'] and
            last['Close'] > last['SMA50']
        )
        
        return {
            'near_term': "BULLISH" if last['Close'] > last['EMA5'] else "BEARISH",
            'breakout': breakout
        }
    except:
        return {'near_term': 'NEUTRAL', 'breakout': False}

def calc_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calc_stoch(df):
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    k = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    d = k.rolling(3).mean()
    return k, d

def calc_adx(df, period=14):
    """Vectorized ADX"""
    try:
        up = df['High'].diff()
        down = -df['Low'].diff()
        
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    except:
        z = pd.Series([0]*len(df), index=df.index)
        return z, z, z
