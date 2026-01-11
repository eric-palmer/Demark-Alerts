# indicators.py - Enhanced technical indicators with institutional accuracy
import pandas as pd
import numpy as np

def calc_rsi(series, period=14):
    """Wilder's RSI - industry standard calculation"""
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Wilder's smoothing (EMA with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(series), index=series.index)

def calc_squeeze(df):
    """TTM Squeeze - Bollinger Bands inside Keltner Channels"""
    try:
        if len(df) < 20:
            return None
        
        # Bollinger Bands (20, 2)
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        upper_bb = sma + (std * 2)
        lower_bb = sma - (std * 2)
        
        # Keltner Channels (20, 1.5 ATR)
        tr = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': abs(df['High'] - df['Close'].shift(1)),
            'lc': abs(df['Low'] - df['Close'].shift(1))
        })
        atr = tr.max(axis=1).rolling(20).mean()
        
        upper_kc = sma + (atr * 1.5)
        lower_kc = sma - (atr * 1.5)
        
        # Squeeze is on when BB is inside KC
        in_squeeze = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        
        if not in_squeeze.iloc[-1]:
            return None
        
        # Linear regression for momentum direction
        y = df['Close'].iloc[-20:].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        bias = "BULLISH" if slope > 0 else "BEARISH"
        
        # Expected move is 2x ATR
        expected_move = atr.iloc[-1] * 2
        
        return {'bias': bias, 'move': expected_move}
    except:
        return None

def calc_fib(df):
    """Fibonacci retracement levels"""
    try:
        lookback = 126  # ~6 months
        if len(df) < lookback:
            return None
        
        high = df['High'].iloc[-lookback:].max()
        low = df['Low'].iloc[-lookback:].min()
        price = df['Close'].iloc[-1]
        
        # Key Fibonacci levels
        diff = high - low
        levels = {
            0.236: high - (diff * 0.236),
            0.382: high - (diff * 0.382),
            0.500: high - (diff * 0.500),
            0.618: high - (diff * 0.618),
            0.786: high - (diff * 0.786)
        }
        
        # Check if price is within 1% of any Fib level
        for level, val in levels.items():
            if abs(price - val) / price < 0.01:
                action = "SUPPORT" if price >= val else "RESISTANCE"
                return {'level': f"{level*100:.1f}%", 'action': action, 'price': val}
        
        return None
    except:
        return None

def calc_demark(df):
    """TD Sequential - Tom DeMark's timing indicator"""
    try:
        df = df.copy()
        df['Close_4'] = df['Close'].shift(4)
        
        # Initialize all columns
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
        setup_ref_high = None
        setup_ref_low = None
        
        closes = df['Close'].values
        closes_4 = df['Close_4'].values
        lows = df['Low'].values
        highs = df['High'].values
        
        for i in range(4, len(df)):
            if pd.notna(closes[i]) and pd.notna(closes_4[i]):
                # Setup phase
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
                
                # Setup completion triggers countdown
                if buy_seq == 9:
                    active_buy = True
                    buy_cd = 0
                    active_sell = False
                    setup_ref_high = max(highs[i-8:i+1])  # High of setup bars
                    
                if sell_seq == 9:
                    active_sell = True
                    sell_cd = 0
                    active_buy = False
                    setup_ref_low = min(lows[i-8:i+1])  # Low of setup bars
            
            # Countdown phase
            if active_buy and i >= 2:
                # Close must be <= close from 2 bars ago
                if pd.notna(closes[i]) and pd.notna(closes[i-2]) and closes[i] <= lows[i-2]:
                    buy_cd += 1
                    df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_cd
                    if buy_cd == 13:
                        active_buy = False
            
            if active_sell and i >= 2:
                # Close must be >= close from 2 bars ago
                if pd.notna(closes[i]) and pd.notna(closes[i-2]) and closes[i] >= highs[i-2]:
                    sell_cd += 1
                    df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_cd
                    if sell_cd == 13:
                        active_sell = False
        
        return df
    except Exception as e:
        print(f"DeMark calc error: {e}")
        return df

def calc_shannon(df):
    """Shannon Demon / AlphaTrends system"""
    try:
        # Moving averages
        df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        
        # AVWAP (approximation using cumulative)
        df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VP'] = df['Typical'] * df['Volume']
        cum_vp = df['VP'].cumsum()
        cum_vol = df['Volume'].cumsum()
        df['AVWAP'] = cum_vp / cum_vol.replace(0, np.nan)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Near-term trend based on 5 EMA
        near_term = "BULLISH" if last['Close'] > last['EMA5'] else "BEARISH"
        
        # Breakout signal: 10 crosses above 20 while price is above 50
        breakout = (
            prev['SMA10'] <= prev['SMA20'] and
            last['SMA10'] > last['SMA20'] and
            last['Close'] > last['SMA50']
        )
        
        return {
            'near_term': near_term,
            'avwap': last['AVWAP'],
            'breakout': breakout
        }
    except:
        return {'near_term': 'NEUTRAL', 'avwap': None, 'breakout': False}

def calc_macd(df, fast=12, slow=26, signal=9):
    """MACD - Moving Average Convergence Divergence"""
    try:
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    except:
        z = pd.Series([0] * len(df), index=df.index)
        return z, z, z

def calc_stoch(df, k_period=14, d_period=3):
    """Stochastic Oscillator - %K and %D"""
    try:
        low_min = df['Low'].rolling(k_period).min()
        high_max = df['High'].rolling(k_period).max()
        
        # Avoid division by zero
        denominator = (high_max - low_min).replace(0, np.nan)
        k_line = 100 * (df['Close'] - low_min) / denominator
        k_line = k_line.fillna(50)
        
        # %D is 3-period SMA of %K
        d_line = k_line.rolling(d_period).mean()
        
        return k_line, d_line
    except:
        z = pd.Series([50] * len(df), index=df.index)
        return z, z

def calc_adx(df, period=14):
    """ADX - Average Directional Index (Wilder's method)"""
    try:
        # Directional Movement
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # True Range
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        
        # Wilder's smoothing
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm_smooth / atr.replace(0, np.nan))
        
        # DX and ADX
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * abs(plus_di - minus_di) / di_sum
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)
    except:
        z = pd.Series([0] * len(df), index=df.index)
        return z, z, z
