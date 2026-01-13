# main.py - Institutional Engine (Options Enabled)
import time
import pandas as pd
import numpy as np
import random
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import (calc_rsi, calc_squeeze, calc_demark, 
                        calc_shannon, calc_adx, calc_ma_trend, 
                        calc_macd, calc_hv, calc_rvol, calc_donchian)
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
                return "NEUTRAL", "Macro Data Unavailable"
            return "NEUTRAL", "Macro Data Unavailable"
        g_imp = growth.pct_change(3).iloc[-1]
        i_imp = inflation.pct_change(63).iloc[-1]
        if g_imp > 0:
            if i_imp < 0: return "RISK_ON", "GOLDILOCKS (Growth ⬆️ Inf ⬇️)"
            else: return "REFLATION", "HEATING UP (Growth ⬆️ Inf ⬆️)"
        else:
            if i_imp < 0: return "SLOWDOWN", "COOLING (Growth ⬇️ Inf ⬇️)"
            else: return "RISK_OFF", "STAGFLATION (Growth ⬇️ Inf ⬆️)"
    except: return "NEUTRAL", "Calc Error"

def get_demark_status(df):
    try:
        last = df.iloc[-1]
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        count = safe_int(bs if bs > ss else ss)
        setup_type = "Buy" if bs > ss else "Sell"
        perf = last.get('Perfected', False)
        return {'type': setup_type, 'count': count, 'perf': perf, 'is_9': (count == 9)}
    except: return {'type': 'None', 'count': 0, 'perf': False, 'is_9': False}

def analyze_ticker(ticker, regime, detailed=False):
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: return None

        if '=F' not in ticker and '-USD' not in ticker:
            last_vol = df['Volume'].iloc[-5:].mean() * df['Close'].iloc[-1]
            if last_vol < 500000: return None 

        # --- INDICATORS ---
        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        sq_res = calc_squeeze(df)
        shannon = calc_shannon(df)
        adx = calc_adx(df)
        ma = calc_ma_trend(df)
        macd_data = calc_macd(df)
        hv = calc_hv(df)
        rvol = calc_rvol(df)
        struct = calc_donchian(df)
        
        last = df.iloc[-1]
        price = last['Close']
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr): atr = price * 0.02

        # --- WEEKLY ---
        weekly_txt = "Neutral"
        if detailed:
            try:
                df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                df_w = calc_demark(df_w)
                w_dm = get_demark_status(df_w)
                if w_dm['count'] > 0:
                    ctx = "Setup"
                    if w_dm['count'] >= 8: ctx = "Near Exhaustion"
                    if w_dm['is_9']: ctx = "REVERSAL RISK"
                    weekly_txt = f"{w_dm['type']} {w_dm['count']} ({ctx})"
            except: pass

        # --- SCORING & BIAS ---
        score = 0
        bias = "Neutral"
        
        # 1. Trend (Adaptive for Young Assets)
        sma200 = ma['sma200'].iloc[-1]
        if sma200 > 0:
            lt_bias = "Bullish" if price > sma200 else "Bearish"
            score += 2 if "Bull" in lt_bias else -2
        else: lt_bias = "Unknown (New Asset)"
            
        # 2. Momentum
        macd_val = macd_data['macd'].iloc[-1]
        macd_sig = macd_data['signal'].iloc[-1]
        mt_bias = "Positive" if macd_val > macd_sig else "Negative"
        score += 1 if macd_val > macd_sig else -1
        
        # 3. Short Term
        daily_dm = get_demark_status(df)
        st_bias = "Neutral"
        
        if daily_dm['is_9']:
            st_bias = f"{daily_dm['type']} 9 (Reversal)"
            score += 3 if daily_dm['type'] == 'Buy' else -3
        elif daily_dm['count'] > 0:
            st_bias = f"{daily_dm['type']} {daily_dm['count']}"
            
        rsi_val = last['RSI']
        if rsi_val > 70: score -= 2; st_bias = "Overbought"
        elif rsi_val < 30: score += 2; st_bias = "Oversold"
        
        if sq_res: score += 2 if sq_res['bias'] == "BULLISH" else -2
        if shannon['breakout']: score += 3

        # Volume
        rvol_val = rvol
        if rvol_val > 1.5: score = score * 1.2

        # --- TARGETS & TIMING ---
        # Donchian or ATR fallback
        if score > 0:
            target = struct['high'] if struct['high'] > price else price + (atr * 3)
            stop = struct['low'] if struct['low'] < price else price - (atr * 1.5)
        else:
            target = struct['low'] if struct['low'] < price else price - (atr * 3)
            stop = struct['high'] if struct['high'] > price else price + (atr * 1.5)
            
        # ATR Velocity (Days to Target)
        dist = abs(target - price)
        # Assuming 1.5x ATR daily move is optimistic, so use 0.8x for conservative timing
        daily_move = atr * 0.8 
        days_to_target = max(1, int(dist / daily_move))
        
        # --- OPTIONS STRATEGY ---
        hv_val = hv.iloc[-1]
        # Logic: High HV + High RSI = Sell Premium. Low HV + Squeeze = Buy Premium.
        opt_strat = "Shares"
        
        if days_to_target < 10: expiry = "Weekly Exp"
        elif days_to_target < 30: expiry = "Monthly Exp"
        else: expiry = "LEAPS / Shares"
        
        strike = round(target)
        
        if score >= 3: # Bullish Options
            if hv_val < 30 or sq_res: opt_strat = f"Long Call (Strike ${strike})" # Cheap Vol
            elif hv_val > 50: opt_strat = f"Bull Put Spread (Sold under ${int(stop)})" # Expensive Vol
            else: opt_strat = f"Call Debit Spread (Target ${strike})"
        elif score <= -3: # Bearish Options
            if hv_val < 30 or sq_res: opt_strat = f"Long Put (Strike ${strike})"
            elif hv_val > 50: opt_strat = f"Bear Call Spread (Sold over ${int(stop)})"
            else: opt_strat = f"Put Debit Spread (Target ${strike})"

        # Verdict
        rec = "⚪ NEUTRAL"
        if score >= 4: rec = "🟢 STRONG BUY"
        elif score >= 2: rec = "🟢 BUY"
        elif score <= -4: rec = "🔴 STRONG SHORT"
        elif score <= -2: rec = "🔴 SHORT"

        adx_val = adx.iloc[-1]
        adx_txt = "Trending" if adx_val > 25 else "Ranging"

        return {
            'ticker': ticker, 'price': price, 'score': round(score, 1), 'rec': rec,
            'horizons': {'short': st_bias, 'med': mt_bias, 'long': lt_bias},
            'techs': {
                'demark': f"{daily_dm['type']} {daily_dm['count']}",
                'weekly': weekly_txt,
                'rsi': f"{rsi_val:.1f} ({'Oversold' if rsi_val<30 else ('Overbought' if rsi_val>70 else 'Neutral')})",
                'adx': f"{adx_val:.1f} ({adx_txt})",
                'stack': ma['status'],
                'vol': f"{rvol_val:.1f}x ({'High' if rvol_val>1.5 else 'Normal'})",
                'squeeze': sq_res['bias'] if sq_res else "None"
            },
            'plan': {'target': target, 'stop': stop, 'days': days_to_target},
            'options': {'strat': opt_strat, 'expiry': expiry}
        }
    except: return None

def format_card(res):
    t = res['techs']
    p = res['plan']
    o = res['options']
    
    msg = f"═════════════════════════\n"
    msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"**{res['rec']}** ({res['score']})\n"
    
    msg += f"🕰️ **Outlook:**\n"
    msg += f"   • Short: {res['horizons']['short']}\n"
    msg += f"   • Med:   {res['horizons']['med']}\n"
    msg += f"   • Long:  {res['horizons']['long']}\n\n"
    
    msg += f"📊 **Vitals:**\n"
    msg += f"   • Trend: {t['stack']}\n"
    msg += f"   • DeMark: D:{t['demark']} | W:{t['weekly']}\n"
    msg += f"   • RSI: {t['rsi']}\n"
    msg += f"   • Vol: {t['vol']}\n"
    
    if t['squeeze'] != "None":
        msg += f"   • **Vol:** Squeeze Firing ({t['squeeze']}) 🚀\n"
        
    msg += f"🎯 **Target:** {fmt_price(p['target'])} (~{p['days']} Days)\n"
    msg += f"🛑 **Stop:** {fmt_price(p['stop'])}\n"
    
    # Only show options if directional
    if "NEUTRAL" not in res['rec']:
        msg += f"💡 **Options:** {o['strat']} ({o['expiry']})\n"
    
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
    print("Scanning Priorities...")
    priority_list = list(set(CURRENT_PORTFOLIO + SHORT_WATCHLIST))
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime, detailed=True): t for t in priority_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: results.append(res)
            
    results.sort(key=lambda x: x['ticker'] in CURRENT_PORTFOLIO, reverse=True)
    
    msg = "💼 *PORTFOLIO & WATCHLIST*\n"
    for r in results: msg += format_card(r)
    send_telegram(msg)

    # 2. MARKET SCAN (BATCHED)
    others = [t for t in STRATEGIC_TICKERS if t not in priority_list]
    # Remove random sample to scan ALL (or keep random for testing)
    # scan_list = random.sample(others, 5) # TEST MODE
    scan_list = others # FULL MODE
    
    batches = [scan_list[i:i + BATCH_SIZE] for i in range(0, len(scan_list), BATCH_SIZE)]
    print(f"Scanning {len(scan_list)} Tickers in {len(batches)} Batches...")
    
    all_results = []
    for i, batch in enumerate(batches):
        print(f"Processing Batch {i+1}...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(analyze_ticker, t, regime): t for t in batch}
            for future in as_completed(future_map):
                res = future.result()
                if res: all_results.append(res)
        
        if i < len(batches) - 1:
            print(f"Sleeping {SLEEP_TIME}s..."); time.sleep(SLEEP_TIME)

    # 3. RANKINGS
    power = [r for r in all_results if abs(r['score']) >= 4]
    power.sort(key=lambda x: abs(x['score']), reverse=True)
    
    if power:
        msg = "🔥 *POWER RANKINGS*\n"
        for r in power[:10]: msg += format_card(r)
        send_telegram(msg)
        
    print("DONE")
