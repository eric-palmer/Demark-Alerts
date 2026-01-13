# indicators.py - Institutional Math (Pro)
import pandas as pd
import numpy as np

def sanitize(series):
    return series.fillna(0)

def calc_fibs(df):
    """Auto-Fibonacci (6-Month Lookback)"""
    try:
        lookback = 126 if len(df) > 126 else len(df)
        recent = df.iloc[-lookback:]
        h = recent['High'].max(); l = recent['Low'].min()
        diff = h - l
        
        levels = {
            'High': h, '23.6%': h-(diff*0.236), '38.2%': h-(diff*0.382),
            '50%': h-(diff*0.5), '61.8%': h-(diff*0.618), 'Low': l
        }
        
        price = df['Close'].iloc[-1]
        nearest = min(levels.items(), key=lambda x: abs(x[1] - price))
        return {'nearest_name': nearest[0], 'nearest_val': nearest[1]}
    except: return {'nearest_name': 'None', 'nearest_val': 0}

def calc_vol_term(df):
    """Vol Term Structure: Cheap vs Expensive Options"""
    try:
        ret = np.log(df['Close']/df['Close'].shift(1))
        hv10 = ret.rolling(10).std()*np.sqrt(252)*100
        hv100 = ret.rolling(100).std()*np.sqrt(252)*100
        
        curr_10 = hv10.iloc[-1]; curr_100 = hv100.iloc[-1]
        
        if curr_10 < curr_100 * 0.8: return "Cheap (Buy Debit)"
        elif curr_10 > curr_100 * 1.2: return "Expensive (Sell Credit)"
        return "Fair Value"
    except: return "Normal"

# --- STANDARD INDICATORS ---
def calc_ma_trend(df):
    try:
        s50 = df['Close'].rolling(50).mean(); s200 = df['Close'].rolling(200).mean()
        e8 = df['Close'].ewm(span=8, adjust=False).mean(); e21 = df['Close'].ewm(span=21, adjust=False).mean()
        return {'sma50': s50, 'sma200': s200, 'ema8': e8, 'ema21': e21}
    except: return {'sma50': 0, 'sma200': 0, 'ema8': 0, 'ema21': 0}

def calc_trend_stack(df):
    try:
        c = df['Close']
        e8 = c.ewm(span=8, adjust=False).mean()
        e21 = c.ewm(span=21, adjust=False).mean()
        s50 = c.rolling(50).mean()
        lc = c.iloc[-1]; l8 = e8.iloc[-1]; l21 = e21.iloc[-1]; l50 = s50.iloc[-1]
        
        if lc > l8 > l21 > l50: return {'status': "Strong Uptrend (Stack Aligned)"}
        elif lc < l8 < l21 < l50: return {'status': "Strong Downtrend (Stack Aligned)"}
        elif lc > l8: return {'status': "Positive Momentum"}
        return {'status': "Choppy / Neutral"}
    except: return {'status': "Error"}

def calc_macd(df):
    try:
        e1 = df['Close'].ewm(span=12, adjust=False).mean()
        e2 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = e1 - e2
        sig = macd.ewm(span=9, adjust=False).mean()
        return {'macd': macd, 'signal': sig}
    except: return {'macd': 0, 'signal': 0}

def calc_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
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
        ubb = sma + (std*2); lbb = sma - (std*2)
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(20).mean()
        ukc = sma + (atr*1.5); lkc = sma - (atr*1.5)
        if (lbb > lkc).iloc[-1] and (ubb < ukc).iloc[-1]: return {'bias': "SQUEEZE"}
        return None
    except: return None

def calc_demark_detailed(df):
    try:
        c = df['Close'].values; l = df['Low'].values; h = df['High'].values
        bs = np.zeros(len(c), dtype=int); ss = np.zeros(len(c), dtype=int)
        
        # Setup
        for i in range(4, len(c)):
            if c[i] < c[i-4]: bs[i] = bs[i-1]+1
            else: bs[i] = 0
            if c[i] > c[i-4]: ss[i] = ss[i-1]+1
            else: ss[i] = 0
            
        # Countdown
        b_cnt = 0; s_cnt = 0; b_active = False; s_active = False
        start = max(0, len(c)-100)
        for i in range(start, len(c)):
            if bs[i] == 9: b_active = True; b_cnt = 0
            if ss[i] == 9: s_active = True; s_cnt = 0
            if b_active and i>=2 and c[i] <= l[i-2]:
                b_cnt += 1; 
                if b_cnt==13: b_active=False
            if s_active and i>=2 and c[i] >= h[i-2]:
                s_cnt += 1
                if s_cnt==13: s_active=False
                
        # Perfection
        last = len(c)-1; perf = False
        if bs[last] >= 9:
            if (l[last]<l[last-2] and l[last]<l[last-3]) or (l[last-1]<l[last-2] and l[last-1]<l[last-3]): perf=True
        elif ss[last] >= 9:
            if (h[last]>h[last-2] and h[last]>h[last-3]) or (h[last-1]>h[last-2] and h[last-1]>h[last-3]): perf=True
            
        curr_bs = bs[-1]; curr_ss = ss[-1]
        if curr_bs > 0: return {'type': 'Buy', 'count': curr_bs, 'countdown': b_cnt, 'perf': perf}
        elif curr_ss > 0: return {'type': 'Sell', 'count': curr_ss, 'countdown': s_cnt, 'perf': perf}
        return {'type': 'Neutral', 'count': 0, 'countdown': 0, 'perf': False}
    except: return {'type': 'Error', 'count': 0, 'countdown': 0, 'perf': False}

def calc_shannon(df):
    try:
        s10 = df['Close'].rolling(10).mean(); s20 = df['Close'].rolling(20).mean(); s50 = df['Close'].rolling(50).mean()
        l = df.iloc[-1]; p = df.iloc[-2]
        brk = (p['SMA10'] <= p['SMA20'] and l['SMA10'] > l['SMA20'] and l['Close'] > l['SMA50'])
        return {'breakout': brk}
    except: return {'breakout': False}

def calc_adx(df, period=14):
    try:
        up = df['High'].diff(); down = -df['Low'].diff()
        p_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
        m_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
        tr = pd.concat([df['High']-df['Low'], abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).replace(0, 0.0001)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        p = 100 * (p_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        m = 100 * (m_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        dx = 100 * abs(p - m) / (p + m)
        return sanitize(dx.rolling(window=period).mean())
    except: return pd.Series([0]*len(df), index=df.index)

def calc_rvol(df):
    try:
        v = df['Volume']; a = v.rolling(20).mean()
        if a.iloc[-1] == 0: return 1.0
        return v.iloc[-1] / a.iloc[-1]
    except: return 1.0

def calc_donchian(df):
    try:
        h20 = df['High'].rolling(20).max().iloc[-1]
        l10 = df['Low'].rolling(10).min().iloc[-1]
        return {'high': h20, 'low': l10}
    except: return {'high': 0, 'low': 0}
