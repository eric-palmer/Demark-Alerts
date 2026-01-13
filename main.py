# main.py - Institutional Pro Engine (Full Universe)
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import (calc_rsi, calc_squeeze, calc_demark_detailed, 
                        calc_shannon, calc_adx, calc_ma_trend, 
                        calc_macd, calc_trend_stack, calc_rvol, calc_donchian, 
                        calc_fibs, calc_vol_term)
from tickers import get_universe
from utils import send_telegram, fmt_price

# --- CONFIGURATION ---
BATCH_SIZE = 50        # Pro Account Speed
SLEEP_TIME = 1         # Pro Account Speed
MAX_WORKERS = 10       # Pro Account Power

# --- ASSETS ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']
SHORT_WATCHLIST = ['LAC', 'IBIT', 'ETHA', 'SPY', 'QQQ']

def get_market_radar_regime(macro):
    # Simplified for robustness
    return "NEUTRAL", "Macro Data Unavailable"

def analyze_ticker(ticker, regime, detailed=False):
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: return None

        # Liquidity Gate: Skip zombie assets (< $2M daily vol)
        # avg_vol_usd = (df['Close'] * df['Volume']).rolling(20).mean().iloc[-1]
        # if avg_vol_usd < 2000000 and ticker not in CURRENT_PORTFOLIO: return None

        # --- INDICATORS ---
        df['RSI'] = calc_rsi(df['Close'])
        dm_daily = calc_demark_detailed(df)
        sq_res = calc_squeeze(df)
        shannon = calc_shannon(df)
        adx = calc_adx(df)
        stack = calc_trend_stack(df)
        ma = calc_ma_trend(df)
        macd_data = calc_macd(df)
        rvol = calc_rvol(df)
        struct = calc_donchian(df)
        fibs = calc_fibs(df)
        vol_term = calc_vol_term(df)
        
        last = df.iloc[-1]
        price = last['Close']
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr): atr = price * 0.02

        # --- WEEKLY ---
        weekly_txt = "Neutral"
        if detailed:
            try:
                df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                dm_w = calc_demark_detailed(df_w)
                if dm_w['count'] > 0:
                    ctx = "Setup"
                    if dm_w['count'] >= 8: ctx = "Near Exhaustion"
                    if dm_w['is_9']: ctx = "REVERSAL SIGNAL"
                    weekly_txt = f"{dm_w['type']} {dm_w['count']} ({ctx})"
            except: pass

        # --- SCORING ---
        score = 0
        
        # 1. Trend
        sma200 = ma['sma200'].iloc[-1]
        if sma200 > 0:
            lt_bias = "Bullish" if price > sma200 else "Bearish"
            score += 2 if "Bull" in lt_bias else -2
        else: lt_bias = "Unknown"
            
        # 2. Momentum
        macd_val = macd_data['macd'].iloc[-1]
        macd_sig = macd_data['signal'].iloc[-1]
        mt_bias = "Positive" if macd_val > macd_sig else "Negative"
        score += 1 if macd_val > macd_sig else -1
        
        # 3. DeMark
        st_bias = "Neutral"
        if dm_daily['is_9']:
            st_bias = f"{dm_daily['type']} 9 Reversal"
            score += 3 if dm_daily['type'] == 'Buy' else -3
        elif dm_daily['countdown'] > 0:
            st_bias = f"{dm_daily['type']} Countdown {dm_daily['countdown']}"
            
        # 4. Fib Bonus
        fib_dist = abs(price - fibs['nearest_val']) / price
        fib_note = ""
        if fib_dist < 0.015: # Within 1.5% of a Fib
            fib_note = f"Testing {fibs['nearest_name']}"
            # If DeMark 9 aligns with Fib, Boost Score
            if dm_daily['is_9']: score += 2 

        # 5. RSI/Vol
        if last['RSI'] > 70: score -= 2
        elif last['RSI'] < 30: score += 2
        
        if sq_res: score += 2 if sq_res['bias'] == "BULLISH" else -2
        if rvol > 1.5: score *= 1.2

        # --- VERDICT ---
        rec = "⚪ NEUTRAL"
        if score >= 4: rec = "🟢 STRONG BUY"
        elif score >= 2: rec = "🟢 BUY"
        elif score <= -4: rec = "🔴 STRONG SHORT"
        elif score <= -2: rec = "🔴 SHORT"

        # --- TARGETS ---
        if score > 0: 
            target = struct['high'] if struct['high'] > price else price + (atr * 3)
            stop = struct['low'] if struct['low'] < price else price - (atr * 1.5)
        else:
            target = struct['low'] if struct['low'] < price else price - (atr * 3)
            stop = struct['high'] if struct['high'] > price else price + (atr * 1.5)
            
        days = max(1, int(abs(target - price) / (atr * 0.8)))

        adx_val = adx.iloc[-1]
        adx_txt = "Trending" if adx_val > 25 else "Flat"

        return {
            'ticker': ticker, 'price': price, 'score': round(score, 1), 'rec': rec,
            'horizons': {'short': st_bias, 'med': mt_bias, 'long': lt_bias},
            'techs': {
                'demark': f"{dm_daily['type']} {dm_daily['count']}",
                'weekly': weekly_txt,
                'rsi': f"{last['RSI']:.1f}",
                'adx': f"{adx_val:.1f} ({adx_txt})",
                'stack': stack['status'],
                'vol': f"{rvol:.1f}x",
                'squeeze': sq_res['bias'] if sq_res else "None",
                'fib': fib_note,
                'opt_strat': vol_term['strat'],
                'dm_obj_d': dm_daily,
                'dm_obj_w': dm_w if detailed else {'count':0},
                'rsi_val': last['RSI']
            },
            'plan': {'target': target, 'stop': stop, 'days': days}
        }
    except: return None

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
    msg += f"   • DeMark (D): {t['demark']}\n"
    msg += f"   • DeMark (W): {t['weekly']}\n"
    msg += f"   • RSI: {t['rsi']}\n"
    
    if t['fib']: msg += f"   • **Fib:** {t['fib']}\n"
    if t['squeeze'] != "None": msg += f"   • **Vol:** Squeeze Firing ({t['squeeze']}) 🚀\n"
    
    # Options Strategy (Pro Only)
    if "NEUTRAL" not in res['rec']:
        msg += f"   • **Opt:** {t['opt_strat']}\n"
        
    msg += f"\n🎯 Target: {fmt_price(res['plan']['target'])} (~{res['plan']['days']} Days)\n"
    msg += f"🛑 Stop: {fmt_price(res['plan']['stop'])}\n"
    
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL PRO ENGINE"); print("="*60)
    
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
        future_map = {executor.submit(analyze_ticker, t, regime, detailed=True): t for t in priority_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: results.append(res)
    results.sort(key=lambda x: x['ticker'] in CURRENT_PORTFOLIO, reverse=True)
    msg = "💼 *PORTFOLIO & WATCHLIST*\n"
    for r in results: msg += format_card(r)
    send_telegram(msg)

    # 2. UNIVERSE SCAN
    print("Fetching Universe...")
    try:
        universe = get_universe() # Dynamic fetch
    except:
        print("Fallback to Static Universe")
        universe = STRATEGIC_TICKERS
        
    # Remove duplicates
    others = [t for t in universe if t not in priority_list]
    
    print(f"Scanning {len(others)} Global Assets...")
    batches = [others[i:i + BATCH_SIZE] for i in range(0, len(others), BATCH_SIZE)]
    all_results = []
    
    for i, batch in enumerate(batches):
        print(f"Batch {i+1}/{len(batches)}...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(analyze_ticker, t, regime): t for t in batch}
            for future in as_completed(future_map):
                res = future.result()
                if res: all_results.append(res)
        if i < len(batches) - 1: time.sleep(SLEEP_TIME)

    # --- PRO DESKS ---
    
    # Power Rankings (Score >= 4 + Perfected)
    power = [r for r in all_results if abs(r['score']) >= 4]
    power.sort(key=lambda x: abs(x['score']), reverse=True)
    if power:
        msg = "🔥 *POWER RANKINGS (Top 10)*\n"
        for r in power[:10]: msg += format_card(r)
        send_telegram(msg)
        
    # DeMark Desk (Daily/Weekly 9s)
    dm_desk = [r for r in all_results if (r['techs']['dm_obj_d']['is_9'] or r['techs']['dm_obj_d']['countdown']==13) and r not in power]
    if dm_desk:
        msg = "🔢 *DEMARK SIGNALS*\n"
        for r in dm_desk[:10]: msg += format_card(r)
        send_telegram(msg)
        
    print("DONE")
