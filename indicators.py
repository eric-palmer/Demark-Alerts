# indicators.py - Technical indicator calculations
import pandas as pd
import numpy as np

def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(series), index=series.index)

def calc_squeeze(df):
    try:
        if len(df) < 20:
            return None
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        upper_bb = sma + (std * 2)
        lower_bb = sma - (std * 2)
        tr = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': abs(df['High'] - df['Close'].shift(1)),
            'lc': abs(df['Low'] - df['Close'].shift(1))
        })
        atr = tr.max(axis=1).rolling(20).mean()
        upper_kc = sma + (atr * 1.5)
        lower_kc = sma - (atr * 1.5)
        in_squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        if not in_squeeze.iloc[-1]:
            return None
        y = df['Close'].iloc[-20:].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        bias = "BULLISH" if slope > 0 else "BEARISH"
        return {'bias': bias, 'move': atr.iloc[-1] * 2}
    except:
        return None

def calc_fib(df):
    try:
        lookback = 126
        if len(df) < lookback:
            return None
        high = df['High'].iloc[-lookback:].max()
        low = df['Low'].iloc[-lookback:].min()
        price = df['Close'].iloc[-1]
        fibs = {
            0.382: high - (high - low) * 0.382,
            0.618: high - (high - low) * 0.618
        }
        for level, val in fibs.items():
            if abs(price - val) / price < 0.01:
                action = "BOUNCE" if price > val else "REJECTION"
                return {'level': f"{level*100:.1f}%", 'action': action, 'price': val}
        return None
    except:
        return None

def calc_demark(df):
    try:
        df = df.copy()
        df['Close_4'] = df['Close'].shift(4)
        df['Buy_Setup'] = 0
        df['Sell_Setup'] = 0
        df['Buy_Countdown'] = 0
        df['Sell_Countdown'] = 0
        buy_seq = 0
        sell_seq = 0
        buy_cd = 0
        sell_cd = 0
        active_buy = False
        active_sell = False
        closes = df['Close'].values
        closes_4 = df['Close_4'].values
        lows = df['Low'].values
        highs = df['High'].values
        for i in range(4, len(df)):
            if pd.notna(closes[i]) and pd.notna(closes_4[i]):
                if closes[i] < closes_4[i]:
                    buy_seq += 1
                    sell_seq = 0
                elif closes[i] > closes_4[i]:
                    sell_seq += 1
                    buy_seq = 0
                else:
                    buy_seq = 0
                    sell_seq = 0
                df.iloc[i, df.columns.get_loc('Buy_Setup')] = buy_seq
                df.iloc[i, df.columns.get_loc('Sell_Setup')] = sell_seq
                if buy_seq == 9:
                    active_buy = True
                    buy_cd = 0
                    active_sell = False
                if sell_seq == 9:
                    active_sell = True
                    sell_cd = 0
                    active_buy = False
            if active_buy and i >= 2:
                if pd.notna(closes[i]) and pd.notna(lows[i-2]) and closes[i] <= lows[i-2]:
                    buy_cd += 1
                    df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_cd
                    if buy_cd == 13:
                        active_buy = False
            if active_sell and i >= 2:
                if pd.notna(closes[i]) and pd.notna(highs[i-2]) and closes[i] >= highs[i-2]:
                    sell_cd += 1
                    df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_cd
                    if sell_cd == 13:
                        active_sell = False
        return df
    except:
        return df

def calc_shannon(df):
    try:
        df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VP'] = df['Typical'] * df['Volume']
        df['AVWAP'] = df['VP'].cumsum() / df['Volume'].cumsum().replace(0, np.nan)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        near_term = "BULLISH" if last['Close'] > last['EMA5'] else "BEARISH"
        breakout = (prev['SMA10'] < prev['SMA20'] and 
                   last['SMA10'] > last['SMA20'] and 
                   last['Close'] > last['SMA50'])
        return {'near_term': near_term, 'avwap': last['AVWAP'], 'breakout': breakout}
    except:
        return {'near_term': 'NEUTRAL', 'avwap': None, 'breakout': False}

def calc_macd(df, fast=12, slow=26, signal=9):
    try:
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - sig
        return macd, sig, hist
    except:
        z = pd.Series([0] * len(df), index=df.index)
        return z, z, z

def calc_stoch(df, k=14, d=3):
    try:
        low_min = df['Low'].rolling(k).min()
        high_max = df['High'].rolling(k).max()
        denom = (high_max - low_min).replace(0, np.nan)
        k_line = 100 * (df['Close'] - low_min) / denom
        k_line = k_line.fillna(50)
        d_line = k_line.rolling(d).mean()
        return k_line, d_line
    except:
        z = pd.Series([50] * len(df), index=df.index)
        return z, z

def calc_adx(df, period=14):
    try:
        high_diff = df['High'] - df['High'].shift(1)
        low_diff = df['Low'].shift(1) - df['Low']
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().replace(0, np.nan)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr)
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = (abs(plus_di - minus_di) / di_sum) * 100
        adx = dx.rolling(period).mean()
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    except:
        z = pd.Series([0] * len(df), index=df.index)
        return z, z, z
