# main.py - Institutional Engine (Portfolio Debug Mode)
import time
import pandas as pd
import numpy as np
import traceback # Added to see the exact error line
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
# VERIFY IMPORTS MATCH INDICATORS.PY
from indicators import (calc_rsi, calc_squeeze, calc_demark, 
                        calc_shannon, calc_adx, calc_ma_trend, 
                        calc_macd, calc_trend_stack, calc_rvol, calc_donchian, calc_hv)
from utils import send_telegram, fmt_price

# --- CONFIGURATION ---
MAX_WORKERS = 1        

# --- ASSETS ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']
SHORT_WATCHLIST = ['LAC', 'IBIT', 'ETHA', 'SPY', 'QQQ']

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
    try:
        last = df.iloc[-1]
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        count = safe_int(bs if bs > ss else ss)
        setup_type = "Buy" if bs > ss else "Sell"
        perf = last.get('Perfected', False)
        return {'type': setup_type, 'count': count, 'perf': perf, 'is_9': (count == 9)}
    except: return {'type': 'None', 'count': 0, 'perf': False, 'is_9': False}

def analyze_ticker(ticker, regime, detailed=False):
    print(f"   ...Analyzing {ticker}") # Debug Print
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: 
            print(f"   ❌ {ticker}: No Data Found")
            return None

        # --- CALCULATIONS ---
        # If one of these crashes, we need to know WHICH one
        try:
            df['RSI'] = calc_rsi(df['Close'])
            df = calc_demark(df)
            sq_res = calc_squeeze(df)
            shannon = calc_shannon(df)
            adx = calc_adx(df)
            stack = calc_trend_stack(df)
            ma = calc_ma_trend(df)
            macd_data = calc_macd(df)
            rvol = calc_rvol(df)
            struct = calc_donchian(df)
        except Exception as e:
            print(f"   ❌ {ticker} MATH ERROR: {e}")
            traceback.print_exc() # Print the exact line number
            return None
        
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
                    if w_dm['is_9']: ctx = "REVERSAL SIGNAL"
                    weekly_txt = f"{w_dm['type']} {w_dm['count']} ({ctx})"
            except: pass

        # --- SCORING ---
        score = 0
        
        # 1. Trend
        sma200 = ma['sma200'].iloc[-1]
        if sma200 > 0:
            lt_bias = "Bullish (Above 200d)" if price > sma200 else "Bearish (Below 200d)"
            score += 2 if "Bull" in lt_bias else -2
        else: lt_bias = "Unknown (New Asset)"
            
        # 2. Momentum
        macd_val = macd_data['macd'].iloc[-1]
        macd_sig = macd_data['signal'].iloc[-1]
        mt_bias = "Positive Momentum" if macd_val > macd_sig else "Negative Momentum"
        score += 1 if macd_val > macd_sig else -1
        
        # 3. Short Term
        daily_dm = get_demark_status(df)
        st_bias = "Neutral"
        
        if daily_dm['is_9']:
            st_bias = f"{daily_dm['type']} 9 Reversal"
            score += 3 if daily_dm['type'] == 'Buy' else -3
        elif daily_dm['count'] > 0:
            st_bias = f"{daily_dm['type']} {daily_dm['count']}"
            
        rsi_val = last['RSI']
        if rsi_val > 70: score -= 2; st_bias = "Overbought"
        elif rsi_val < 30: score += 2; st_bias = "Oversold"
        
        if sq_res: score += 2 if sq_res['bias'] == "BULLISH" else -2
        if shannon['breakout']: score += 3
        if rvol > 1.5: score = score * 1.2

        # --- TARGETS ---
        if score > 0: 
            target = struct['high'] if struct['high'] > price else price + (atr * 3)
            stop = struct['low'] if struct['low'] < price else price - (atr * 1.5)
        else:
            target = struct['low'] if struct['low'] < price else price - (atr * 3)
            stop = struct['high'] if struct['high'] > price else price + (atr * 1.5)
            
        dist = abs(target - price)
        daily_move = atr * 0.8 
        days = max(1, int(dist / daily_move))

        rec = "⚪ NEUTRAL"
        if score >= 4: rec = "🟢 STRONG BUY"
        elif score >= 2: rec = "🟢 BUY"
        elif score <= -4: rec = "🔴 STRONG SHORT"
        elif score <= -2: rec = "🔴 SHORT"

        adx_val = adx.iloc[-1]
        adx_txt = "Strong Trend" if adx_val > 25 else "No Trend (Wait)"

        return {
            'ticker': ticker, 'price': price, 'score': round(score, 1), 'rec': rec,
            'horizons': {'short': st_bias, 'med': mt_bias, 'long': lt_bias},
            'techs': {
                'demark': f"{daily_dm['type']} {daily_dm['count']}",
                'weekly': weekly_txt,
                'rsi': f"{rsi_val:.1f} ({'Oversold' if rsi_val<30 else ('Overbought' if rsi_val>70 else 'Neutral')})",
                'adx': f"{adx_val:.1f} ({adx_txt})",
                'stack': stack['status'],
                'vol': f"{rvol:.1f}x",
                'squeeze': sq_res['bias'] if sq_res else "None"
            },
            'plan': {'target': target, 'stop': stop, 'days': days}
        }
    except Exception as e:
        print(f"   ❌ CRITICAL ERROR for {ticker}: {e}")
        traceback.print_exc()
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
    msg += f"   • DeMark (D): {t['demark']}\n"
    msg += f"   • DeMark (W): {t['weekly']}\n"
    msg += f"   • RSI: {t['rsi']}\n"
    msg += f"   • Vol: {t['vol']}\n"
    
    if t['squeeze'] != "None":
        msg += f"   • **Vol:** Squeeze Firing ({t['squeeze']}) 🚀\n"
        
    msg += f"\n🎯 Target: {fmt_price(res['plan']['target'])} (~{res['plan']['days']} Days)\n"
    msg += f"🛑 Stop: {fmt_price(res['plan']['stop'])}\n"
    
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL PORTFOLIO SCANNER"); print("="*60)
    
    try:
        macro = get_macro()
        regime, desc = get_market_radar_regime(macro)
    except:
        regime = "NEUTRAL"; desc = "Macro Data Failed"
    send_telegram(f"📊 *MARKET REGIME: {regime}*\n{desc}")

    # 1. PRIORITY SCAN (NO BATCHING, NO SLEEP)
    print("Scanning Portfolio & Watchlist...")
    priority_list = list(set(CURRENT_PORTFOLIO + SHORT_WATCHLIST))
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime, detailed=True): t for t in priority_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: results.append(res)
            
    results.sort(key=lambda x: x['ticker'] in CURRENT_PORTFOLIO, reverse=True)
    
    if results:
        msg = "💼 *PORTFOLIO & WATCHLIST*\n"
        for r in results: msg += format_card(r)
        send_telegram(msg)
        print(f"✅ Success! Sent {len(results)} reports.")
    else:
        print("❌ FAILURE: No results generated. Check the logs above for specific errors.")

    print("🛑 STOPPING (Safe Mode).")
    exit()
