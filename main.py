# main.py - Institutional Pro Engine (Final)
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
BATCH_SIZE = 50        
SLEEP_TIME = 1         
MAX_WORKERS = 10       

# --- ASSETS ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']
SHORT_WATCHLIST = ['LAC', 'IBIT', 'ETHA', 'SPY', 'QQQ']

def check_connection():
    key = os.environ.get('TIINGO_API_KEY')
    if not key:
        print("❌ CRITICAL: TIINGO_API_KEY missing.")
        return False
    return True

def analyze_ticker(ticker, regime, detailed=False):
    print(f"...Scanning {ticker}")
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: return None

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

        # --- WEEKLY ---
        dm_w = {'type': 'Neutral', 'count': 0, 'countdown': 0, 'is_9': False}
        if detailed:
            try:
                df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                dm_w = calc_demark_detailed(df_w)
            except: pass

        # --- SCORING ---
        score = 0
        
        # Trend
        if ma['sma200'].iloc[-1] > 0:
            lt_bias = "Bullish" if price > ma['sma200'].iloc[-1] else "Bearish"
            score += 2 if "Bull" in lt_bias else -2
        else: lt_bias = "Unknown"
            
        # Momentum
        if macd['macd'].iloc[-1] > macd['signal'].iloc[-1]: score += 1
        else: score -= 1
        
        # DeMark
        st_bias = "Neutral"
        if dm_d['is_9']:
            st_bias = f"Reversal Setup ({dm_d['type']} 9)"
            score += 3 if dm_d['type'] == 'Buy' else -3
        elif dm_d['count'] > 0:
            st_bias = f"{dm_d['type']} {dm_d['count']}"
            
        # Fibs
        fib_note = ""
        if abs(price - fibs['nearest_val']) / price < 0.02:
            fib_note = f"At {fibs['nearest_name']}"
            if dm_d['is_9']: score += 2

        # RSI & Vol
        if last['RSI'] > 70: score -= 2
        elif last['RSI'] < 30: score += 2
        if sq: score += 2
        if rvol > 1.5: score *= 1.2

        # --- TARGETS ---
        if score > 0: 
            target = struct['high'] if struct['high'] > price else price + (atr * 3)
            stop = struct['low'] if struct['low'] < price else price - (atr * 1.5)
        else:
            target = struct['low'] if struct['low'] < price else price - (atr * 3)
            stop = struct['high'] if struct['high'] > price else price + (atr * 1.5)
            
        days = max(1, int(abs(target - price) / (atr * 0.8)))

        rec = "⚪ NEUTRAL"
        if score >= 4: rec = "🟢 STRONG BUY"
        elif score >= 2: rec = "🟢 BUY"
        elif score <= -4: rec = "🔴 STRONG SHORT"
        elif score <= -2: rec = "🔴 SHORT"

        adx_val = adx.iloc[-1]
        adx_txt = "Trending" if adx_val > 25 else "Flat"

        return {
            'ticker': ticker, 'price': price, 'score': round(score, 1), 'rec': rec,
            'horizons': {'short': st_bias, 'med': "Bullish" if score>0 else "Bearish"},
            'techs': {
                'demark_d': dm_d, # Pass Full Object
                'demark_w': dm_w, # Pass Full Object
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
    except Exception as e:
        print(f"❌ Error {ticker}: {e}")
        return None

def format_portfolio_card(res):
    """Deep Dive for Portfolio Assets"""
    t = res['techs']
    d_dm = t['demark_d']
    w_dm = t['demark_w']
    
    # Daily DeMark Text
    dd_txt = f"{d_dm['type']} {d_dm['count']}"
    if d_dm['is_9']: dd_txt += f" ({'PERFECTED' if d_dm['perf'] else 'UNPERFECTED'})"
    if d_dm['countdown'] > 0: dd_txt += f" | Countdown: {d_dm['countdown']}/13"
    
    # Weekly DeMark Text
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
    """Concise Card for Universe"""
    t = res['techs']
    msg = f"═════════════════════════\n"
    msg += f"🔥 *{res['ticker']}* ({res['rec']})\n"
    msg += f"   • Signal: {t['demark_d']['type']} {t['demark_d']['count']}\n"
    if t['fib']: msg += f"   • Fib: {t['fib']}\n"
    msg += f"🎯 {fmt_price(res['plan']['target'])} | 🛑 {fmt_price(res['plan']['stop'])}\n"
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL PRO SCANNER (Test Mode)"); print("="*60)
    
    if not check_connection(): exit()

    # 1. PRIORITY SCAN (Portfolio Only)
    print("Scanning Portfolio & Watchlist...")
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

    print("🛑 TEST COMPLETE. Universe Scan Paused.")
    exit()
