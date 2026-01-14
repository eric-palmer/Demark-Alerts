# main.py - Institutional Pro Engine (Final Production)
import os
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import (calc_rsi, calc_squeeze, calc_demark_detailed, 
                        calc_shannon, calc_adx, calc_ma_trend, 
                        calc_macd, calc_trend_stack, calc_rvol, 
                        calc_donchian, calc_fibs, calc_vol_term)
from tickers import get_universe
from utils import send_telegram, fmt_price

# --- CONFIGURATION ---
BATCH_SIZE = 100       # High throughput for 800 tickers
SLEEP_TIME = 1         # 1s pause between batches
MAX_WORKERS = 20       # Parallel processing power

# --- ASSETS ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']
SHORT_WATCHLIST = ['LAC', 'IBIT', 'ETHA', 'SPY', 'QQQ']

def check_env():
    """Double check secrets before starting"""
    if not os.environ.get('TIINGO_API_KEY') or not os.environ.get('TELEGRAM_BOT_TOKEN'):
        print("❌ CRITICAL: Secrets missing. Check YAML.")
        return False
    return True

def analyze_ticker(ticker, regime, detailed=False):
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: return None

        # Liquidity Gate: Skip illiquid assets (<$2M/day) unless in Portfolio
        if ticker not in CURRENT_PORTFOLIO and ticker not in SHORT_WATCHLIST:
            avg_vol_usd = (df['Close'] * df['Volume']).rolling(20).mean().iloc[-1]
            if avg_vol_usd < 2000000: return None 

        # --- INDICATORS ---
        df['RSI'] = calc_rsi(df['Close'])
        dm_d = calc_demark_detailed(df)
        sq = calc_squeeze(df)
        adx = calc_adx(df)
        stack = calc_trend_stack(df)
        ma = calc_ma_trend(df)
        macd = calc_macd(df)
        rvol = calc_rvol(df)
        struct = calc_donchian(df)
        fibs = calc_fibs(df)
        vol_term = calc_vol_term(df)
        
        last = df.iloc[-1]; price = last['Close']
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr): atr = price * 0.02

        # --- WEEKLY (Deep History) ---
        dm_w = {'type': 'Neutral', 'count': 0, 'countdown': 0, 'is_9': False}
        if detailed:
            try:
                df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                dm_w = calc_demark_detailed(df_w)
            except: pass

        # --- SCORING ---
        score = 0
        
        # Trend & Momentum
        if ma['sma200'].iloc[-1] > 0: score += 2 if price > ma['sma200'].iloc[-1] else -2
        score += 1 if macd['macd'].iloc[-1] > macd['signal'].iloc[-1] else -1
        
        # DeMark
        st_bias = "Neutral"
        if dm_d['is_9']:
            st_bias = f"Reversal Setup ({dm_d['type']} 9)"
            score += 3 if dm_d['type'] == 'Buy' else -3
        elif dm_d['count'] > 0:
            st_bias = f"{dm_d['type']} {dm_d['count']}"
            
        # Timeframe Conflict (The "Safety Valve")
        conflict = False
        if dm_d['count'] >= 5 and dm_w['count'] >= 5:
            if dm_d['type'] != dm_w['type']:
                conflict = True
                score = score / 2 # Slash score if conflicting signals

        # Fibs
        fib_note = ""
        if abs(price - fibs['nearest_val']) / price < 0.02:
            fib_note = f"At {fibs['nearest_name']}"
            if dm_d['is_9']: score += 2

        # RSI/Vol
        if last['RSI'] > 70: score -= 2
        elif last['RSI'] < 30: score += 2
        if sq: score += 2
        if rvol > 1.5: score *= 1.2

        # --- VERDICT ---
        rec = "⚪ NEUTRAL"
        if score >= 4: rec = "🟢 STRONG BUY"
        elif score >= 2: rec = "🟢 BUY"
        elif score <= -4: rec = "🔴 STRONG SHORT"
        elif score <= -2: rec = "🔴 SHORT"
        
        if conflict: rec = "⚠️ CONFLICT (Wait)"

        adx_val = adx.iloc[-1]
        adx_txt = "Trending" if adx_val > 25 else "Flat"
        
        # Targets
        target = struct['high'] if score > 0 else struct['low']
        stop = struct['low'] if score > 0 else struct['high']
        days = max(1, int(abs(target - price) / (atr * 0.8)))

        return {
            'ticker': ticker, 'price': price, 'score': round(score, 1), 'rec': rec,
            'horizons': {'short': st_bias, 'med': "Bullish" if score>0 else "Bearish"},
            'techs': {
                'demark_d': dm_d, 'demark_w': dm_w,
                'rsi': f"{last['RSI']:.1f}",
                'adx': f"{adx_val:.1f} ({adx_txt})",
                'stack': stack['status'],
                'vol': f"{rvol:.1f}x",
                'squeeze': "FIRING" if sq else "None",
                'fib': fib_note,
                'opt': vol_term
            },
            'plan': {'target': target, 'stop': stop, 'days': days}
        }
    except: return None

def format_portfolio_card(res):
    """Deep Dive for Portfolio"""
    t = res['techs']
    d_dm = t['demark_d']; w_dm = t['demark_w']
    
    dd_txt = f"{d_dm['type']} {d_dm['count']}"
    if d_dm['is_9']: dd_txt += f" ({'PERFECTED' if d_dm['perf'] else 'UNPERFECTED'})"
    if d_dm['countdown'] > 0: dd_txt += f" | Countdown: {d_dm['countdown']}/13"
    
    wd_txt = f"{w_dm['type']} {w_dm['count']}"
    if w_dm['is_9']: wd_txt += " (SETUP COMPLETE)"
    
    msg = f"═════════════════════════\n"
    msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"**{res['rec']}** (Score: {res['score']})\n"
    msg += f"─────────────────────────\n"
    msg += f"📊 **FULL TECHNICALS:**\n"
    msg += f"   • Daily DeMark: {dd_txt}\n"
    msg += f"   • Weekly DeMark: {wd_txt}\n"
    msg += f"   • Trend: {t['stack']}\n"
    msg += f"   • RSI: {t['rsi']} | Vol: {t['vol']}\n"
    if t['fib']: msg += f"   • Fib: {t['fib']}\n"
    if t['squeeze'] != "None": msg += f"   • Squeeze: FIRING 🚀\n"
    msg += f"   • Options: {t['opt']}\n"
    msg += f"\n🎯 Target: {fmt_price(res['plan']['target'])} (~{res['plan']['days']} Days)\n"
    msg += f"🛑 Stop: {fmt_price(res['plan']['stop'])}\n"
    return msg + "\n"

def format_scanner_card(res):
    """Highlight Card for Universe"""
    t = res['techs']
    msg = f"═════════════════════════\n"
    msg += f"🔥 *{res['ticker']}* ({res['rec']})\n"
    msg += f"   • Signal: {t['demark_d']['type']} {t['demark_d']['count']}\n"
    if t['fib']: msg += f"   • Fib: {t['fib']}\n"
    msg += f"🎯 {fmt_price(res['plan']['target'])} | 🛑 {fmt_price(res['plan']['stop'])}\n"
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL PRO ENGINE (Production)"); print("="*60)
    
    if not check_env(): exit()

    # 1. PORTFOLIO SCAN
    print("Scanning Portfolio...")
    priority_list = list(set(CURRENT_PORTFOLIO + SHORT_WATCHLIST))
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, "NEUTRAL", detailed=True): t for t in priority_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: results.append(res)
    results.sort(key=lambda x: x['ticker'] in CURRENT_PORTFOLIO, reverse=True)
    msg = "💼 *PORTFOLIO & WATCHLIST*\n"
    for r in results: msg += format_portfolio_card(r)
    send_telegram(msg)

    # 2. UNIVERSE SCAN (UNLOCKED)
    print("Fetching Universe...")
    try: universe = get_universe()
    except: universe = SHORT_WATCHLIST
    
    others = [t for t in universe if t not in priority_list]
    print(f"Scanning {len(others)} Global Assets...")
    
    batches = [others[i:i + BATCH_SIZE] for i in range(0, len(others), BATCH_SIZE)]
    all_results = []
    
    for i, batch in enumerate(batches):
        print(f"Batch {i+1}/{len(batches)}...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(analyze_ticker, t, "NEUTRAL"): t for t in batch}
            for future in as_completed(future_map):
                res = future.result()
                if res: all_results.append(res)
        if i < len(batches) - 1: time.sleep(SLEEP_TIME)

    # 3. PRO DESKS
    power = [r for r in all_results if abs(r['score']) >= 4]
    power.sort(key=lambda x: abs(x['score']), reverse=True)
    if power:
        msg = "🔥 *POWER RANKINGS (Top 10)*\n"
        for r in power[:10]: msg += format_scanner_card(r)
        send_telegram(msg)
        
    dm_desk = [r for r in all_results if (r['techs']['demark_d']['is_9'] or r['techs']['demark_d']['countdown']==13) and r not in power]
    if dm_desk:
        msg = "🔢 *DEMARK SIGNALS (Daily)*\n"
        for r in dm_desk[:10]: msg += format_scanner_card(r)
        send_telegram(msg)
        
    print("DONE")
