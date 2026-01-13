# main.py - Institutional Engine (English Analyst Mode)
import time
import pandas as pd
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import (calc_rsi, calc_squeeze, calc_demark, 
                        calc_shannon, calc_adx, calc_ma_trend, calc_macd)
from utils import send_telegram, fmt_price

# --- CONFIGURATION ---
BATCH_SIZE = 40        
SLEEP_TIME = 3660      
MAX_WORKERS = 1        

# --- ASSETS ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']
SHORT_WATCHLIST = ['LAC', 'IBIT', 'ETHA', 'SPY', 'QQQ']

STRATEGIC_TICKERS = [
    'PENGU-USD', 'FARTCOIN-USD', 'DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'TRUMP-USD',
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BTDR', 'MARA', 'RIOT', 'HUT', 'CLSK', 
    'IREN', 'CIFR', 'BTBT', 'WYFI', 'CORZ', 'CRWV', 'APLD', 'NBIS', 'WULF', 
    'HIVE', 'BITF', 'WGMI', 'MNRS', 'OWNB', 'BMNR', 'SBET', 'FWDI', 'BKKT',
    'IBIT', 'ETHA', 'BITQ', 'BSOL', 'GSOL', 'SOLT', 'MSTR', 'COIN', 'HOOD', 
    'GLXY', 'STKE', 'DFDV', 'NODE', 'GEMI', 'BLSH', 'CRCL',
    'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F', 'PA=F', 'GLD', 'SLV', 
    'PALL', 'PPLT', 'NIKL', 'LIT', 'ILIT', 'REMX', 'VOLT', 'GRID', 'EQT', 
    'TAC', 'BE', 'OKLO', 'SMR', 'NEE', 'URA', 'SRUUF', 'CCJ', 'KAMJY', 'UNL',
    'NVDA', 'SMH', 'SMHX', 'TSM', 'AVGO', 'QCOM', 'MU', 'AMD', 'TER', 'NOW', 
    'AXON', 'SNOW', 'PLTR', 'GOOG', 'MSFT', 'META', 'AMZN', 'AAPL', 'TSLA', 
    'NFLX', 'SPOT', 'SHOP', 'UBER', 'DASH', 'NET', 'DXCM', 'ETSY', 'SQ', 
    'FIG', 'MAGS', 'MTUM', 'IVES', 'ARKK', 'ARKF', 'ARKG', 'GRNY', 'GRNI', 
    'GRNJ', 'XBI', 'XHB', 'XLK', 'XLI', 'XLU', 'XLRE', 'XLB', 'XLV', 'XLF', 
    'XLE', 'XLP', 'XLY', 'XLC', 'BABA', 'JD', 'BIDU', 'PDD', 'XIACY', 'BYDDY', 
    'LKNCY', 'TCEHY', 'MCHI', 'INDA', 'EWZ', 'EWJ', 'EWG', 'EWU', 'EWY', 'EWW', 
    'EWT', 'EWC', 'EEM', 'AMX', 'PBR', 'VALE', 'NSRGY', 'DEO', 'BLK', 'STT', 
    'ARES', 'SOFI', 'PYPL', 'IBKR', 'WU', 'RXRX', 'SDGR', 'TEM', 'ABSI', 
    'DNA', 'TWST', 'GLW', 'KHC', 'LULU', 'YETI', 'DLR', 'EQIX', 'ORCL', 'LSF'
]

def safe_int(val):
    try:
        if pd.isna(val) or val is None: return 0
        return int(float(val))
    except: return 0

def get_market_radar_regime(macro):
    try:
        growth = macro.get('growth')
        inflation = macro.get('inflation')
        if growth is None or inflation is None:
            return "NEUTRAL", "Macro Data Unavailable"
        g_imp = growth.pct_change(3).iloc[-1]
        i_imp = inflation.pct_change(63).iloc[-1]
        
        if g_imp > 0:
            if i_imp < 0: return "GOLDILOCKS", "Risk On (Longs Preferred)"
            else: return "REFLATION", "Inflationary (Commodities Long)"
        else:
            if i_imp < 0: return "SLOWDOWN", "Deflationary (Bonds/Quality)"
            else: return "STAGFLATION", "Risk Off (Cash/Shorts)"
    except: return "NEUTRAL", "Calc Error"

def get_demark_status(df):
    """Helper to extract DeMark status"""
    try:
        last = df.iloc[-1]
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        count = safe_int(bs if bs > ss else ss)
        setup_type = "Buy" if bs > ss else "Sell"
        perf = last.get('Perfected', False)
        return {
            'type': setup_type, 'count': count, 
            'perf': perf, 'is_9': (count == 9), 'is_13': (count == 13)
        }
    except: return {'type': 'None', 'count': 0, 'perf': False, 'is_9': False, 'is_13': False}

def analyze_ticker(ticker, regime):
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: return None

        # Filter Illiquid
        if '=F' not in ticker and '-USD' not in ticker:
            last_vol = df['Volume'].iloc[-5:].mean() * df['Close'].iloc[-1]
            if last_vol < 500000: return None 

        # --- INDICATOR CALCULATIONS ---
        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        sq_res = calc_squeeze(df)
        shannon = calc_shannon(df)
        adx = calc_adx(df)
        ma = calc_ma_trend(df)
        macd_data = calc_macd(df)
        
        last = df.iloc[-1]
        price = last['Close']
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr): atr = price * 0.02

        # --- TIME HORIZON ANALYSIS ---
        
        # 1. LONG TERM (6m+) - 200d Moving Average
        sma200 = ma['sma200'].iloc[-1]
        lt_bias = "Bullish" if price > sma200 else "Bearish"
        
        # 2. MEDIUM TERM (2-3m) - 50d MA & MACD
        sma50 = ma['sma50'].iloc[-1]
        macd_val = macd_data['macd'].iloc[-1]
        macd_sig = macd_data['signal'].iloc[-1]
        
        mt_bias = "Neutral"
        if price > sma50 and macd_val > macd_sig: mt_bias = "Bullish"
        elif price < sma50 and macd_val < macd_sig: mt_bias = "Bearish"
        
        # 3. SHORT TERM (1-2w) - DeMark, RSI, Squeeze
        daily_dm = get_demark_status(df)
        rsi_val = last['RSI']
        
        st_bias = "Neutral"
        if rsi_val < 30 or daily_dm['is_9'] and daily_dm['type'] == 'Buy': st_bias = "Bullish (Bounce)"
        elif rsi_val > 70 or daily_dm['is_9'] and daily_dm['type'] == 'Sell': st_bias = "Bearish (Pullback)"
        elif shannon['breakout']: st_bias = "Bullish (Breakout)"

        # --- SCORING (-10 to +10) ---
        score = 0
        if lt_bias == "Bullish": score += 2
        else: score -= 2
        
        if mt_bias == "Bullish": score += 2
        elif mt_bias == "Bearish": score -= 2
        
        if st_bias.startswith("Bullish"): score += 2
        elif st_bias.startswith("Bearish"): score -= 2
        
        if sq_res: score += 2 if sq_res['bias'] == "BULLISH" else -2

        # --- WEEKLY CONTEXT ---
        weekly_txt = "Neutral"
        try:
            df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            df_w = calc_demark(df_w)
            w_dm = get_demark_status(df_w)
            if w_dm['count'] > 0: weekly_txt = f"{w_dm['type']} {w_dm['count']}"
        except: pass

        # --- TARGETS ---
        if score > 0: # Long
            target = price + (atr * 3)
            stop = price - (atr * 1.5)
        else: # Short
            target = price - (atr * 3)
            stop = price + (atr * 1.5)

        # --- PLAIN ENGLISH INTERPRETATION ---
        rec = "⚪ NEUTRAL"
        if score >= 4: rec = "🟢 STRONG BUY"
        elif score >= 2: rec = "🟢 BUY"
        elif score <= -4: rec = "🔴 STRONG SHORT"
        elif score <= -2: rec = "🔴 SHORT"

        adx_val = adx.iloc[-1]
        adx_txt = "Trending" if adx_val > 25 else "Choppy"
        
        return {
            'ticker': ticker, 'price': price, 'score': score, 'rec': rec,
            'horizons': {'short': st_bias, 'med': mt_bias, 'long': lt_bias},
            'techs': {
                'demark': f"{daily_dm['type']} {daily_dm['count']}",
                'weekly': weekly_txt,
                'rsi': f"{rsi_val:.1f} ({'Oversold' if rsi_val<30 else ('Overbought' if rsi_val>70 else 'Neutral')})",
                'adx': f"{adx_val:.1f} ({adx_txt})",
                'macd': "Bull Cross" if macd_val > macd_sig else "Bear Cross",
                'squeeze': sq_res['bias'] if sq_res else "None"
            },
            'plan': {'target': target, 'stop': stop}
        }
    except: return None

def format_card(res):
    t = res['techs']
    msg = f"═════════════════════════\n"
    msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"**{res['rec']}** (Score: {res['score']})\n"
    msg += f"─────────────────────────\n"
    
    # Time Horizons
    msg += f"🕰️ **Outlook:**\n"
    msg += f"   • Short (1-2w): {res['horizons']['short']}\n"
    msg += f"   • Med (2-3m):   {res['horizons']['med']}\n"
    msg += f"   • Long (6m+):   {res['horizons']['long']}\n\n"
    
    # Technical Detail
    msg += f"📊 **Technical Drivers:**\n"
    msg += f"   • DeMark (D): {t['demark']}\n"
    msg += f"   • DeMark (W): {t['weekly']}\n"
    msg += f"   • RSI: {t['rsi']}\n"
    msg += f"   • Trend: {t['macd']} | {t['adx']}\n"
    
    if t['squeeze'] != "None":
        msg += f"   • **Vol:** Squeeze Firing ({t['squeeze']}) 🚀\n"
        
    msg += f"─────────────────────────\n"
    msg += f"🎯 **Target:** {fmt_price(res['plan']['target'])}\n"
    msg += f"🛑 **Stop:** {fmt_price(res['plan']['stop'])}\n"
    
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL BATCH SCANNER"); print("="*60)
    
    try:
        macro = get_macro()
        regime, desc = get_market_radar_regime(macro)
    except:
        regime = "NEUTRAL"; desc = "Macro Data Failed"
    send_telegram(f"📊 *MARKET REGIME: {regime}*\n{desc}")

    # 1. PRIORITY SCAN
    print("Scanning Portfolio...")
    priority_list = list(set(CURRENT_PORTFOLIO + SHORT_WATCHLIST))
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in priority_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: results.append(res)
            
    # Sort: Portfolio first
    results.sort(key=lambda x: x['ticker'] in CURRENT_PORTFOLIO, reverse=True)
    
    msg = "💼 *PORTFOLIO & WATCHLIST*\n"
    for r in results:
        msg += format_card(r)
    send_telegram(msg)

    # 2. SAMPLER SCAN (Test 5 Randoms)
    others = [t for t in STRATEGIC_TICKERS if t not in priority_list]
    scan_batch = random.sample(others, 5)
    
    print("Scanning Sample Batch...")
    scan_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in scan_batch}
        for future in as_completed(future_map):
            res = future.result()
            if res: scan_results.append(res)
            
    # Power Rankings
    power = [r for r in scan_results if abs(r['score']) >= 4]
    if power:
        msg = "🔥 *POWER RANKINGS*\n"
        for r in power: msg += format_card(r)
        send_telegram(msg)
        
    print("🛑 SAMPLE COMPLETE. Exiting.")
    exit()
