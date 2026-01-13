# main.py - Institutional Engine (Multi-Desk Final)
import time
import pandas as pd
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import (calc_rsi, calc_squeeze, calc_demark_detailed, 
                        calc_shannon, calc_adx, calc_ma_trend, 
                        calc_macd, calc_trend_stack, calc_rvol, calc_donchian, calc_hv)
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
        dm_daily = calc_demark_detailed(df)
        sq_res = calc_squeeze(df)
        shannon = calc_shannon(df)
        adx = calc_adx(df)
        stack = calc_trend_stack(df)
        ma = calc_ma_trend(df)
        macd_data = calc_macd(df)
        rvol = calc_rvol(df)
        struct = calc_donchian(df)
        
        last = df.iloc[-1]
        price = last['Close']
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr): atr = price * 0.02

        # --- WEEKLY (4 Year History) ---
        dm_weekly = {'type': 'Neutral', 'count': 0}
        if detailed:
            try:
                df_w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
                dm_weekly = calc_demark_detailed(df_w)
            except: pass

        # --- SCORING & BIAS ---
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
        mt_bias = "Positive" if macd_val > macd_sig else "Negative"
        score += 1 if "Pos" in mt_bias else -1
        
        # 3. DeMark (Short Term)
        st_bias = "Neutral"
        if dm_daily['count'] == 9:
            st_bias = "Reversal Setup (9)"
            score += 3 if dm_daily['type'] == 'Buy' else -3
        elif dm_daily['countdown'] == 13:
            st_bias = "Trend Exhaustion (13)"
            score += 4 if dm_daily['type'] == 'Buy' else -4
            
        # 4. RSI
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

        # --- TEXT FORMATTING ---
        rec = "⚪ NEUTRAL"
        if score >= 4: rec = "🟢 STRONG BUY"
        elif score >= 2: rec = "🟢 BUY"
        elif score <= -4: rec = "🔴 STRONG SHORT"
        elif score <= -2: rec = "🔴 SHORT"

        # DeMark English
        dm_d_txt = f"{dm_daily['type']} {dm_daily['count']}"
        if dm_daily['count'] == 9: dm_d_txt += f" ({'PERFECTED' if dm_daily['perf'] else 'UNPERFECTED'})"
        if dm_daily['countdown'] > 0: dm_d_txt += f" (Countdown: {dm_daily['countdown']}/13)"
        
        dm_w_txt = f"{dm_weekly['type']} {dm_weekly['count']}"
        if dm_weekly['count'] == 9: dm_w_txt += " (Setup Complete)"

        adx_val = adx.iloc[-1]
        adx_txt = "Trending" if adx_val > 25 else "No Trend"

        return {
            'ticker': ticker, 'price': price, 'score': round(score, 1), 'rec': rec,
            'horizons': {'short': st_bias, 'med': mt_bias, 'long': lt_bias},
            'techs': {
                'demark_d': dm_d_txt,
                'demark_w': dm_w_txt,
                'rsi': f"{rsi_val:.1f} ({'Oversold' if rsi_val<30 else ('Overbought' if rsi_val>70 else 'Neutral')})",
                'adx': f"{adx_val:.1f} ({adx_txt})",
                'stack': stack['status'],
                'vol': f"{rvol:.1f}x",
                'squeeze': sq_res['bias'] if sq_res else "None",
                'dm_obj': dm_daily # Object for filtering
            },
            'plan': {'target': target, 'stop': stop, 'days': days}
        }
    except: return None

def format_card(res, simple=False):
    t = res['techs']
    msg = f"═════════════════════════\n"
    msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"**{res['rec']}** ({res['score']})\n\n"
    
    if not simple:
        msg += f"🕰️ **Outlook:**\n"
        msg += f"   • Short: {res['horizons']['short']}\n"
        msg += f"   • Med:   {res['horizons']['med']}\n"
        msg += f"   • Long:  {res['horizons']['long']}\n\n"
    
    msg += f"📊 **Vitals:**\n"
    msg += f"   • Trend: {t['stack']}\n"
    msg += f"   • DeMark (D): {t['demark_d']}\n"
    if not simple: msg += f"   • DeMark (W): {t['demark_w']}\n"
    msg += f"   • RSI: {t['rsi']}\n"
    
    if t['squeeze'] != "None":
        msg += f"   • **Vol:** Squeeze Firing ({t['squeeze']}) 🚀\n"
        
    msg += f"\n🎯 Target: {fmt_price(res['plan']['target'])} (~{res['plan']['days']} Days)\n"
    msg += f"🛑 Stop: {fmt_price(res['plan']['stop'])}\n"
    
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

    # 2. SAMPLER (Test Mode - Switch to Full when ready)
    others = [t for t in STRATEGIC_TICKERS if t not in priority_list]
    scan_batch = random.sample(others, 5) 
    
    print("Scanning Sample Batch...")
    scan_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in scan_batch}
        for future in as_completed(future_map):
            res = future.result()
            if res: scan_results.append(res)
            
    # --- DESKS ---
    
    # Power Rankings
    power = [r for r in scan_results if abs(r['score']) >= 4]
    if power:
        msg = "🔥 *POWER RANKINGS*\n"
        for r in power: msg += format_card(r)
        send_telegram(msg)
        
    # DeMark Desk (Any 9s or 13s not in Power)
    dm_desk = [r for r in scan_results if (r['techs']['dm_obj']['count'] == 9 or r['techs']['dm_obj']['countdown'] == 13) and r not in power]
    if dm_desk:
        msg = "🔢 *DEMARK DESK (Signals)*\n"
        for r in dm_desk: msg += format_card(r, simple=True)
        send_telegram(msg)
        
    # RSI Desk
    rsi_desk = [r for r in scan_results if ("Over" in r['techs']['rsi']) and r not in power and r not in dm_desk]
    if rsi_desk:
        msg = "🌊 *RSI EXTREMES*\n"
        for r in rsi_desk: msg += format_card(r, simple=True)
        send_telegram(msg)
        
    print("🛑 SAMPLE COMPLETE. Exiting.")
    exit()
