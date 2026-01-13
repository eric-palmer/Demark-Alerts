# main.py - Institutional Engine (Shannon/Newton Logic)
import time
import pandas as pd
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import (calc_rsi, calc_squeeze, calc_demark, 
                        calc_shannon, calc_adx, calc_trend_stack, 
                        calc_rvol, calc_donchian)
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

        # --- CALCULATIONS ---
        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        sq_res = calc_squeeze(df)
        shannon = calc_shannon(df)
        adx = calc_adx(df)
        stack = calc_trend_stack(df)
        rvol = calc_rvol(df)
        struct = calc_donchian(df)
        
        last = df.iloc[-1]
        price = last['Close']
        
        # --- WEEKLY (If Detailed) ---
        weekly_dm_status = "Neutral"
        if detailed:
            try:
                df_weekly = df.resample('W-FRI').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
                df_weekly = calc_demark(df_weekly)
                w_last = df_weekly.iloc[-1]
                w_bs = w_last.get('Buy_Setup', 0); w_ss = w_last.get('Sell_Setup', 0)
                if w_bs >= 9: weekly_dm_status = f"Buy 9 (Exhaustion)"
                elif w_ss >= 9: weekly_dm_status = f"Sell 9 (Exhaustion)"
                elif w_bs > 0: weekly_dm_status = f"Buy {int(w_bs)}"
                elif w_ss > 0: weekly_dm_status = f"Sell {int(w_ss)}"
            except: pass

        # --- SCORING ---
        score = 0
        bias = "Neutral"
        
        # 1. Trend (200d SMA)
        sma200 = stack['sma200']
        lt_trend = "Bullish" if price > sma200 else "Bearish"
        if lt_trend == "Bullish": score += 2
        else: score -= 2
        
        # 2. DeMark (With Trend Filter)
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        dm_count = safe_int(bs if bs > ss else ss)
        dm_type = "Buy" if bs > ss else "Sell"
        dm_perf = last.get('Perfected', False)
        
        dm_note = ""
        if dm_count == 9:
            perf_txt = "Perfected" if dm_perf else "Unperfected"
            # Newton Logic: 
            if dm_type == "Buy" and lt_trend == "Bullish":
                dm_note = f"🟢 **BUY THE DIP** (9 {perf_txt} in Uptrend)"
                score += 3
            elif dm_type == "Buy" and lt_trend == "Bearish":
                dm_note = f"⚠️ **COUNTER-TREND BUY** (Risk of Failure)"
                score += 1 # Less confidence
            elif dm_type == "Sell" and lt_trend == "Bearish":
                dm_note = f"🔴 **TREND CONTINUATION SHORT**"
                score -= 3
            elif dm_type == "Sell" and lt_trend == "Bullish":
                dm_note = f"⚠️ **COUNTER-TREND SELL** (Top Picking)"
                score -= 1
        elif dm_count == 13:
            dm_note = f"🛑 **TERMINATION (13)**"
            score += 3 if dm_type == "Buy" else -3
        elif dm_count > 0:
            dm_note = f"{dm_type} {dm_count}"

        # 3. Indicators
        adx_val = adx.iloc[-1]
        rsi_val = last['RSI']
        
        if rsi_val > 70: score -= 2; bias = "Bearish"
        elif rsi_val < 30: score += 2; bias = "Bullish"
        
        if sq_res:
            if sq_res['bias'] == "BULLISH": score += 2
            else: score -= 2
            
        if shannon['breakout']: score += 3
        
        # 4. Volume Confirmation
        vol_txt = "Normal"
        if rvol > 1.5: 
            vol_txt = "High Conviction 🟢"
            score = score * 1.2 # Amplify score if volume supports it
        elif rvol < 0.7:
            vol_txt = "Low Participation ⚠️"
            
        # Targets (Donchian)
        if bias == "Bullish" or score > 0:
            target = struct['high_20']
            stop = struct['low_10']
        else:
            target = struct['low_10'] # Targeting lows for short
            stop = struct['high_20'] # Stop at highs
            
        # Fallback if structure is tight
        if abs(target - price) < price * 0.02:
            atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
            target = price + (atr*3) if score > 0 else price - (atr*3)
            stop = price - (atr*1.5) if score > 0 else price + (atr*1.5)

        return {
            'ticker': ticker, 'price': price, 'score': round(score, 1), 'bias': bias,
            'details': {
                'stack': stack['status'],
                'vol': f"{rvol:.1f}x ({vol_txt})",
                'demark': dm_note if dm_note else "Neutral",
                'weekly_dm': weekly_dm_status,
                'rsi': f"{rsi_val:.1f}",
                'adx': f"{adx_val:.1f}",
                'squeeze': sq_res['bias'] if sq_res else "None"
            },
            'plan': {'target': target, 'stop': stop},
            'dm_raw': {'count': dm_count, 'type': dm_type, 'perf': dm_perf, 'is_13': (dm_count==13)}
        }
    except: return None

def format_analyst_card(res):
    """Deep Dive for Portfolio"""
    d = res['details']
    p = res['plan']
    
    score_icon = "🟢" if res['score'] > 2 else ("🔴" if res['score'] < -2 else "⚪")
    
    msg = f"{score_icon} *{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"Score: {res['score']}/10 | Trend: {d['stack']}\n"
    msg += f"────────────────\n"
    
    # DeMark Section
    msg += f"🔢 **DeMark:**\n"
    msg += f"   • Daily:  {d['demark']}\n"
    msg += f"   • Weekly: {d['weekly_dm']}\n"
    
    # Indicators
    msg += f"📊 **Vitals:**\n"
    msg += f"   • Vol: {d['vol']}\n"
    msg += f"   • RSI: {d['rsi']} | ADX: {d['adx']}\n"
    if d['squeeze'] != "None": msg += f"   • Squeeze: {d['squeeze']} 🚀\n"
        
    msg += f"────────────────\n"
    msg += f"🎯 **Target:** {fmt_price(p['target'])}\n"
    msg += f"🛑 **Stop:** {fmt_price(p['stop'])}\n"
    
    return msg + "\n"

def format_scanner_card(res, title):
    d = res['details']
    msg = f"*{res['ticker']}* ({res['score']})\n"
    msg += f"👉 *{title}*\n"
    msg += f"Signal: {d['demark']}\n"
    msg += f"Trend: {d['stack']}\n"
    msg += f"Target: {fmt_price(res['plan']['target'])} | Stop: {fmt_price(res['plan']['stop'])}\n"
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

    # 2. PRIORITY: Portfolio + Watchlist
    print("Scanning Priorities...")
    priority_list = list(set(CURRENT_PORTFOLIO + SHORT_WATCHLIST))
    port_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # detailed=True activates Weekly DeMark
        future_map = {executor.submit(analyze_ticker, t, regime, detailed=True): t for t in priority_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: port_results.append(res)
            
    port_results.sort(key=lambda x: x['ticker'] in CURRENT_PORTFOLIO, reverse=True)
    
    msg = "💼 *PORTFOLIO & WATCHLIST*\n\n"
    for r in port_results:
        if r['ticker'] in SHORT_WATCHLIST and r['ticker'] not in CURRENT_PORTFOLIO:
            msg += "👀 **WATCHLIST: " + r['ticker'] + "**\n"
        msg += format_analyst_card(r)
    send_telegram(msg)

    # 3. SAMPLE SCANNER (Test Mode)
    others = [t for t in STRATEGIC_TICKERS if t not in priority_list]
    scan_batch = random.sample(others, 5) # 5 Randoms for speed
    
    print(f"Scanning {len(scan_batch)} Random Tickers...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in scan_batch}
        for future in as_completed(future_map):
            res = future.result()
            if res: results.append(res)
            
    # --- CATEGORIZED REPORTING ---
    
    # Power Rankings (High Score)
    power = [r for r in results if abs(r['score']) >= 4]
    if power:
        msg = "🔥 *POWER RANKINGS*\n\n"
        for r in power: msg += format_scanner_card(r, "High Conviction")
        send_telegram(msg)
        
    # DeMark 9s (Daily)
    dm_9 = [r for r in results if r['dm_raw']['count'] == 9 and r not in power]
    if dm_9:
        msg = "🔢 *DAILY DEMARK 9s*\n\n"
        for r in dm_9: msg += format_scanner_card(r, "Exhaustion Setup")
        send_telegram(msg)
        
    # RSI Extremes
    rsi_ext = [r for r in results if (float(r['details']['rsi']) > 70 or float(r['details']['rsi']) < 30) and r not in power and r not in dm_9]
    if rsi_ext:
        msg = "🌊 *RSI EXTREMES*\n\n"
        for r in rsi_ext: msg += format_scanner_card(r, "Overbought/Oversold")
        send_telegram(msg)

    print("🛑 SAMPLE COMPLETE. Exiting.")
    exit()
