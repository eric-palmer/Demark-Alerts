# main.py - Institutional Pro Engine (Test Mode: Portfolio Only)
import os
import time
import pandas as pd
import numpy as np
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure data_fetcher and indicators are the PRO versions
from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import (calc_rsi, calc_squeeze, calc_demark_detailed, 
                        calc_shannon, calc_adx, calc_ma_trend, 
                        calc_macd, calc_trend_stack, calc_rvol, 
                        calc_donchian, calc_fibs, calc_vol_term)
from tickers import get_universe # Imported but not used in this test run
from utils import send_telegram, fmt_price

# --- CONFIGURATION ---
MAX_WORKERS = 10       # Fast parallel processing

# --- ASSETS ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']
SHORT_WATCHLIST = ['LAC', 'IBIT', 'ETHA', 'SPY', 'QQQ']

def check_connection():
    """Verifies API Key availability before starting"""
    key = os.environ.get('TIINGO_API_KEY')
    if not key:
        print("❌ CRITICAL: TIINGO_API_KEY missing from Environment.")
        return False
    return True

def get_market_radar_regime(macro):
    # Simplified logic to prevent external dependency failures during testing
    return "NEUTRAL", "Macro Data Unavailable"

def analyze_ticker(ticker, regime, detailed=False):
    """
    Core Analysis Engine:
    - Fetches Data (Tiingo Pro)
    - Runs Institutional Math (Fibs, Vol, DeMark, Trends)
    - Generates 'English' Strategy Card
    """
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

        # --- WEEKLY ANALYSIS (Deep History) ---
        weekly_txt = "Neutral"; dm_w = {'count': 0, 'type': 'Neutral'}
        if detailed:
            try:
                df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                dm_w = calc_demark_detailed(df_w)
                if dm_w['count'] > 0:
                    ctx = "Setup"
                    if dm_w['count'] >= 9: ctx = "REVERSAL"
                    weekly_txt = f"{dm_w['type']} {dm_w['count']} ({ctx})"
            except: pass

        # --- SCORING SYSTEM (-10 to +10) ---
        score = 0
        
        # 1. Trend (200d)
        if ma['sma200'].iloc[-1] > 0:
            lt_bias = "Bullish" if price > ma['sma200'].iloc[-1] else "Bearish"
            score += 2 if "Bull" in lt_bias else -2
        else: lt_bias = "Unknown (New Asset)"
            
        # 2. Momentum (MACD)
        mt_bias = "Positive" if macd['macd'].iloc[-1] > macd['signal'].iloc[-1] else "Negative"
        score += 1 if "Pos" in mt_bias else -1
        
        # 3. Short Term (DeMark/RSI)
        st_bias = "Neutral"
        if dm_d['is_9']:
            st_bias = f"{dm_d['type']} 9 Reversal"
            score += 3 if dm_d['type'] == 'Buy' else -3
        elif dm_d['count'] > 0:
            st_bias = f"{dm_d['type']} {dm_d['count']}"
            
        # Fib Confluence
        fib_note = ""
        if abs(price - fibs['nearest_val']) / price < 0.02:
            fib_note = f"At {fibs['nearest_name']}"
            if dm_d['is_9']: score += 2

        if last['RSI'] > 70: score -= 2; st_bias = "Overbought"
        elif last['RSI'] < 30: score += 2; st_bias = "Oversold"
        
        if sq: score += 2 if sq['bias'] == "BULLISH" else -2
        if rvol > 1.5: score *= 1.2

        # --- TARGETS & TIMING ---
        if score > 0: 
            target = struct['high'] if struct['high'] > price else price + (atr * 3)
            stop = struct['low'] if struct['low'] < price else price - (atr * 1.5)
        else:
            target = struct['low'] if struct['low'] < price else price - (atr * 3)
            stop = struct['high'] if struct['high'] > price else price + (atr * 1.5)
            
        days = max(1, int(abs(target - price) / (atr * 0.8)))

        # --- VERDICT ---
        rec = "⚪ NEUTRAL"
        if score >= 4: rec = "🟢 STRONG BUY"
        elif score >= 2: rec = "🟢 BUY"
        elif score <= -4: rec = "🔴 STRONG SHORT"
        elif score <= -2: rec = "🔴 SHORT"

        adx_val = adx.iloc[-1]
        adx_txt = "Trending" if adx_val > 25 else "Flat"

        return {
            'ticker': ticker, 'price': price, 'score': round(score, 1), 'rec': rec,
            'horizons': {'short': st_bias, 'med': mt_bias, 'long': lt_bias},
            'techs': {
                'demark_d': f"{dm_d['type']} {dm_d['count']}",
                'demark_w': weekly_txt,
                'rsi': f"{last['RSI']:.1f}",
                'adx': f"{adx_val:.1f} ({adx_txt})",
                'stack': stack['status'],
                'vol': f"{rvol:.1f}x",
                'squeeze': "FIRING" if sq else "None",
                'fib': fib_note,
                'opt': vol_term,
                'dm_obj_d': dm_d,
                'dm_obj_w': dm_w
            },
            'plan': {'target': target, 'stop': stop, 'days': days}
        }
    except Exception as e:
        print(f"❌ Error {ticker}: {e}")
        # traceback.print_exc() # Uncomment if you need deep debugging
        return None

def format_card(res):
    t = res['techs']
    msg = f"═════════════════════════\n"
    msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"**{res['rec']}** ({res['score']})\n\n"
    
    msg += f"🕰️ **Outlook:**\n"
    msg += f"   • Short: {res['horizons']['short']}\n"
    msg += f"   • Med:   {res['horizons']['med']}\n"
    msg += f"   • Long:  {res['horizons']['long']}\n\n"
    
    msg += f"📊 **Vitals:**\n"
    msg += f"   • Trend: {t['stack']}\n"
    msg += f"   • DeMark (D): {t['demark_d']}\n"
    msg += f"   • DeMark (W): {t['demark_w']}\n"
    msg += f"   • RSI: {t['rsi']}\n"
    if t['fib']: msg += f"   • **Fib:** {t['fib']}\n"
    msg += f"   • **Opt:** {t['opt']}\n"
    
    msg += f"\n🎯 Target: {fmt_price(res['plan']['target'])} (~{res['plan']['days']} Days)\n"
    msg += f"🛑 Stop: {fmt_price(res['plan']['stop'])}\n"
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL PRO SCANNER (Test Mode)"); print("="*60)
    
    if not check_connection(): exit()

    try:
        macro = get_macro()
        regime, desc = get_market_radar_regime(macro)
    except: regime="NEUTRAL"; desc="Data Error"
    send_telegram(f"📊 *MARKET REGIME: {regime}*\n{desc}")

    # 1. PRIORITY SCAN (Portfolio Only)
    print("Scanning Portfolio & Watchlist...")
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

    print("🛑 TEST COMPLETE. Universe Scan Paused.")
    exit()
