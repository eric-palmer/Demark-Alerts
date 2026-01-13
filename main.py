# main.py - Institutional Engine (Conflict Aware)
import time
import pandas as pd
import numpy as np
import traceback
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

def get_market_radar_regime(macro):
    return "NEUTRAL", "Macro Data Unavailable" # Simplified for robustness

def analyze_ticker(ticker, regime, detailed=False):
    print(f"...Analyzing {ticker}")
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: 
            print(f"❌ {ticker}: No Data")
            return None

        # --- CALCULATIONS ---
        df['RSI'] = calc_rsi(df['Close'])
        dm_d = calc_demark_detailed(df)
        sq = calc_squeeze(df)
        adx = calc_adx(df)
        stack = calc_trend_stack(df)
        ma = calc_ma_trend(df)
        macd = calc_macd(df)
        rvol = calc_rvol(df)
        struct = calc_donchian(df)
        
        last = df.iloc[-1]; price = last['Close']
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr): atr = price * 0.02

        # --- WEEKLY ---
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

        # --- SCORING ---
        score = 0
        bias = "Neutral"
        
        # Trend
        if ma['sma200'].iloc[-1] > 0:
            score += 2 if price > ma['sma200'].iloc[-1] else -2
            
        # Momentum
        if macd['macd'].iloc[-1] > macd['signal'].iloc[-1]: score += 1
        else: score -= 1
        
        # DeMark
        st_bias = "Neutral"
        if dm_d['is_9']:
            st_bias = f"{dm_d['type']} 9 Reversal"
            score += 3 if dm_d['type'] == 'Buy' else -3
        elif dm_d['count'] > 0:
            st_bias = f"{dm_d['type']} {dm_d['count']}"
            
        # TIMEFRAME CONFLICT CHECK
        conflict = False
        if dm_d['count'] >= 5 and dm_w['count'] >= 5:
            if dm_d['type'] != dm_w['type']:
                conflict = True
                st_bias += " ⚠️ CONFLICT"
                score = score / 2 # Penalize score for conflict

        # RSI & Vol
        if last['RSI'] > 70: score -= 2
        elif last['RSI'] < 30: score += 2
        
        if sq: score += 2 if sq['bias'] == "BULLISH" else -2
        if rvol > 1.5: score *= 1.2

        # --- TARGETS ---
        if score > 0: 
            target = struct['high'] if struct['high'] > price else price + (atr * 3)
            stop = struct['low'] if struct['low'] < price else price - (atr * 1.5)
        else:
            target = struct['low'] if struct['low'] < price else price - (atr * 3)
            stop = struct['high'] if struct['high'] > price else price + (atr * 1.5)
            
        days = max(1, int(abs(target - price) / (atr * 0.8)))

        # Verdict
        rec = "⚪ NEUTRAL"
        if score >= 4: rec = "🟢 STRONG BUY"
        elif score >= 2: rec = "🟢 BUY"
        elif score <= -4: rec = "🔴 STRONG SHORT"
        elif score <= -2: rec = "🔴 SHORT"
        
        if conflict: rec = "⚠️ CONFLICT (Wait)"

        return {
            'ticker': ticker, 'price': price, 'score': round(score, 1), 'rec': rec,
            'horizons': {'short': st_bias, 'med': "Bullish" if score>0 else "Bearish"},
            'techs': {
                'demark_d': f"{dm_d['type']} {dm_d['count']}",
                'demark_w': weekly_txt,
                'rsi': f"{last['RSI']:.1f}",
                'adx': f"{adx.iloc[-1]:.1f}",
                'stack': stack['status'],
                'vol': f"{rvol:.1f}x",
                'squeeze': sq['bias'] if sq else "None",
                'dm_obj_d': dm_d, # For filtering
                'dm_obj_w': dm_w,
                'rsi_val': last['RSI']
            },
            'plan': {'target': target, 'stop': stop, 'days': days}
        }
    except Exception as e:
        print(f"❌ Error {ticker}: {e}")
        return None

def format_card(res):
    t = res['techs']
    msg = f"═════════════════════════\n"
    msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"**{res['rec']}** ({res['score']})\n\n"
    
    msg += f"🕰️ **Outlook:**\n"
    msg += f"   • Short: {res['horizons']['short']}\n"
    msg += f"   • Med:   {res['horizons']['med']}\n\n"
    
    msg += f"📊 **Vitals:**\n"
    msg += f"   • Trend: {t['stack']}\n"
    msg += f"   • DeMark (D): {t['demark_d']}\n"
    msg += f"   • DeMark (W): {t['demark_w']}\n"
    msg += f"   • RSI: {t['rsi']} | Vol: {t['vol']}\n"
    
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

    # 2. MARKET SCAN (FULL)
    others = [t for t in STRATEGIC_TICKERS if t not in priority_list]
    batches = [others[i:i + BATCH_SIZE] for i in range(0, len(others), BATCH_SIZE)]
    print(f"Scanning {len(others)} Tickers...")
    
    all_results = []
    for i, batch in enumerate(batches):
        print(f"Batch {i+1}...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(analyze_ticker, t, regime, detailed=True): t for t in batch}
            for future in as_completed(future_map):
                res = future.result()
                if res: all_results.append(res)
        if i < len(batches) - 1: time.sleep(SLEEP_TIME)

    # --- DESK REPORTING ---
    
    # Power Rankings (Top 10 High Score)
    power = [r for r in all_results if abs(r['score']) >= 4]
    power.sort(key=lambda x: abs(x['score']), reverse=True)
    if power:
        msg = "🔥 *POWER RANKINGS*\n"
        for r in power[:10]: msg += format_card(r)
        send_telegram(msg)
        
    # DeMark Desk (Any 9s or 13s, even if low score)
    dm_desk = [r for r in all_results if (r['techs']['dm_obj_d']['count'] == 9 or r['techs']['dm_obj_d']['countdown'] == 13) and r not in power]
    if dm_desk:
        msg = "🔢 *DEMARK SIGNALS (Daily)*\n"
        for r in dm_desk: msg += format_card(r)
        send_telegram(msg)
        
    # Weekly Desk (Weekly 9s are rare and important)
    w_desk = [r for r in all_results if r['techs']['dm_obj_w']['count'] == 9 and r not in power]
    if w_desk:
        msg = "🗓️ *WEEKLY EXHAUSTION (Macro)*\n"
        for r in w_desk: msg += format_card(r)
        send_telegram(msg)

    print("DONE")
