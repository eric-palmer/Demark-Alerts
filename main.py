# main.py - Institutional Engine (Ultimate Analyst Mode)
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
            if macro.get('net_liq') is not None:
                nl = macro['net_liq']
                if len(nl) > 63 and nl.iloc[-1] > nl.iloc[-63]:
                    return "LIQUIDITY EXPANSION", "Fed Adding Liquidity (Growth Data Missing)"
            return "NEUTRAL", "Macro Data Unavailable"
        g_impulse = growth.pct_change(3).iloc[-1]
        i_impulse = inflation.pct_change(63).iloc[-1]
        if g_impulse > 0:
            if i_impulse < 0: return "RISK_ON", "GOLDILOCKS (Growth ⬆️ Inf ⬇️)"
            else: return "REFLATION", "HEATING UP (Growth ⬆️ Inf ⬆️)"
        else:
            if i_impulse < 0: return "SLOWDOWN", "COOLING (Growth ⬇️ Inf ⬇️)"
            else: return "RISK_OFF", "STAGFLATION (Growth ⬇️ Inf ⬆️)"
    except: return "NEUTRAL", "Calc Error"

def analyze_ticker(ticker, regime, detailed=False):
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: return None

        if '=F' not in ticker and '-USD' not in ticker:
            last_vol = df['Volume'].iloc[-5:].mean() * df['Close'].iloc[-1]
            if last_vol < 500000: return None 

        # --- WEEKLY AGGREGATION (For Portfolio Context) ---
        df_weekly = None
        if detailed:
            try:
                df_weekly = df.resample('W-FRI').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
            except: pass

        # --- CALCULATE DAILY INDICATORS ---
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
        
        # --- CALCULATE WEEKLY (If detailed) ---
        weekly_context = "Neutral"
        if df_weekly is not None and len(df_weekly) > 20:
            df_weekly = calc_demark(df_weekly)
            w_last = df_weekly.iloc[-1]
            w_bs = w_last.get('Buy_Setup', 0); w_ss = w_last.get('Sell_Setup', 0)
            if w_bs >= 9: weekly_context = "Weekly Buy Exhaustion ⚠️"
            elif w_ss >= 9: weekly_context = "Weekly Sell Exhaustion ⚠️"
            elif w_last['Close'] > df_weekly['Close'].rolling(20).mean().iloc[-1]: weekly_context = "Weekly Bull Trend"
            else: weekly_context = "Weekly Bear Trend"

        # --- SCORING & ANALYSIS ---
        score = 0
        bias = "Neutral"
        
        # 1. Trend (MACD + 50d)
        sma50 = ma['sma50'].iloc[-1]
        macd_val = macd_data['macd'].iloc[-1]
        macd_sig = macd_data['signal'].iloc[-1]
        
        if macd_val > macd_sig:
            bias = "Bullish"
            score += 1
        elif macd_val < macd_sig:
            bias = "Bearish"
            
        if price > sma50: score += 1
            
        # 2. DeMark (Daily)
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        dm_count = safe_int(bs if bs > ss else ss)
        dm_type = "Buy" if bs > ss else "Sell"
        dm_perf = last.get('Perfected', False)
        
        dm_status = f"{dm_type} {dm_count}"
        
        if dm_count == 9:
            status_txt = "PERFECTED" if dm_perf else "UNPERFECTED"
            dm_status = f"**{dm_type} 9 ({status_txt})**"
            score += 3
            # Contrarian Logic: Buying a 9 is risky if trend is crashing
            if dm_type == "Buy" and bias == "Bearish": score -= 1
        elif dm_count == 13:
            dm_status = f"**{dm_type} 13 (Terminator)**"
            score += 4
        elif dm_count >= 1:
            bars_left = 9 - dm_count
            dm_status = f"{dm_type} {dm_count} ({bars_left} to 9)"

        # 3. RSI & Volatility
        adx_val = adx.iloc[-1]
        rsi_val = last['RSI']
        
        if rsi_val > 70: score += 2; bias = "Bearish"
        elif rsi_val < 30: score += 2; bias = "Bullish"

        if sq_res: score += 2

        if shannon['breakout']: score += 3; bias = "Bullish"
        if "RISK_OFF" in regime and bias == "Bullish": score -= 1
        
        # Targets
        if bias == "Bullish":
            target = price + (atr * 3)
            stop = price - (atr * 2)
            timeframe = "1-4 Weeks"
        else:
            target = price - (atr * 3)
            stop = price + (atr * 2)
            timeframe = "1-4 Weeks"
        
        return {
            'ticker': ticker, 'price': price, 'score': score, 'bias': bias,
            'indicators': {
                'rsi': rsi_val,
                'adx': adx_val,
                'demark': dm_status,
                'weekly': weekly_context,
                'squeeze': sq_res
            },
            'plan': {'target': target, 'stop': stop, 'time': timeframe}
        }
    except: return None

def format_universal_card(res, reason):
    """Standard Output for Scanner Hits"""
    msg = f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"👉 *{reason}*\n"
    msg += f"   • Score: {res['score']}/10 | Bias: {res['bias']}\n"
    
    t = res['indicators']
    msg += f"   • DeMark: {t['demark']}\n"
    msg += f"   • RSI: {t['rsi']:.1f} | ADX: {t['adx']:.1f}\n"
    if t['squeeze']: msg += f"   • Squeeze: {t['squeeze']['bias']} Ready\n"
        
    p = res['plan']
    msg += f"🎯 Target: {fmt_price(p['target'])}\n"
    msg += f"🛑 Stop: {fmt_price(p['stop'])}\n"
    return msg + "\n"

def format_analyst_card(res):
    """Detailed Card for Portfolio Only"""
    t = res['indicators']
    p = res['plan']
    
    msg = f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"**Score:** {res['score']}/10 | **Bias:** {res['bias']}\n"
    msg += f"────────────────\n"
    
    # DeMark Detail
    msg += f"🔢 **DeMark:** {t['demark']}\n"
    msg += f"   └ Weekly Context: {t['weekly']}\n"
    
    # Trend Detail
    adx_txt = "Strong Trend" if t['adx'] > 25 else "Choppy"
    msg += f"📈 **Trend:** ADX {t['adx']:.1f} ({adx_txt})\n"
    
    # Momentum Detail
    rsi_txt = "Neutral"
    if t['rsi'] > 70: rsi_txt = "Overbought ⚠️"
    elif t['rsi'] < 30: rsi_txt = "Oversold 🟢"
    msg += f"🌊 **RSI:** {t['rsi']:.1f} ({rsi_txt})\n"
    
    if t['squeeze']: msg += f"🚀 **Volatility:** Squeeze Firing ({t['squeeze']['bias']})\n"
        
    msg += f"────────────────\n"
    msg += f"🎯 **Target:** {fmt_price(p['target'])}\n"
    msg += f"🛑 **Stop:** {fmt_price(p['stop'])}\n"
    msg += f"⏳ **Time:** {p['time']}\n"
    
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL BATCH SCANNER"); print("="*60)
    
    # 1. Macro
    try:
        macro = get_macro()
        regime, desc = get_market_radar_regime(macro)
    except:
        regime = "NEUTRAL"; desc = "Macro Data Failed"
    send_telegram(f"📊 *MARKET REGIME: {regime}*\n{desc}")

    # 2. PRIORITY: Portfolio + Watchlist (Detailed Mode)
    print("Scanning Priorities...")
    priority_list = list(set(CURRENT_PORTFOLIO + SHORT_WATCHLIST))
    port_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass 'detailed=True' to get Weekly context
        future_map = {executor.submit(analyze_ticker, t, regime, detailed=True): t for t in priority_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: port_results.append(res)
    
    # Sort: Portfolio first
    port_results.sort(key=lambda x: x['ticker'] in CURRENT_PORTFOLIO, reverse=True)
    
    msg = "💼 *PORTFOLIO & WATCHLIST*\n\n"
    for r in port_results:
        if r['ticker'] in SHORT_WATCHLIST and r['ticker'] not in CURRENT_PORTFOLIO:
            msg += "👀 **WATCHLIST: " + r['ticker'] + "**\n"
        msg += format_analyst_card(r)
    send_telegram(msg)

    # 3. SAMPLE SCANNER (Test Mode - 5 Randoms)
    full_list = list(set(STRATEGIC_TICKERS))
    others = [t for t in full_list if t not in priority_list]
    # Pick 5 random for speed test
    if len(others) > 5:
        scan_list = random.sample(others, 5)
    else:
        scan_list = others
        
    print(f"Scanning {len(scan_list)} Random Tickers...")
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in scan_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: results.append(res)
            
    # --- CATEGORIZED REPORT ---
    
    # Power Rankings
    power = [r for r in results if r['score'] >= 4]
    if power:
        msg = "🔥 *POWER RANKINGS (High Conviction)*\n\n"
        for r in power: msg += format_universal_card(r, "Multi-Signal Confluence")
        send_telegram(msg)
        
    # DeMark Desk
    dm_9 = [r for r in results if "9" in r['indicators']['demark'] and r not in power]
    if dm_9:
        msg = "9️⃣ *DEMARK SIGNALS*\n\n"
        for r in dm_9: msg += format_universal_card(r, "DeMark 9 Exhaustion")
        send_telegram(msg)
        
    # RSI Desk
    rsi_play = [r for r in results if (r['indicators']['rsi'] < 30 or r['indicators']['rsi'] > 70) and r not in power and r not in dm_9]
    if rsi_play:
        msg = "🌊 *RSI EXTREMES*\n\n"
        for r in rsi_play: msg += format_universal_card(r, "RSI Extreme")
        send_telegram(msg)

    print("🛑 SAMPLE COMPLETE. Exiting.")
    exit()
