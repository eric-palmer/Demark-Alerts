# main.py - Institutional Engine (Detailed & Explained)
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

def analyze_ticker(ticker, regime):
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: return None

        if '=F' not in ticker and '-USD' not in ticker:
            last_vol = df['Volume'].iloc[-5:].mean() * df['Close'].iloc[-1]
            if last_vol < 500000: return None 

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
        
        # --- ENGLISH TRANSLATION & SCORING ---
        score = 0
        explanations = [] # Collect reasons for the score
        
        # 1. Trend Stack (Shannon)
        # Check alignment: Price > 8 > 21 > 50
        c = df['Close']
        ema8 = c.ewm(span=8, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        sma50 = ma['sma50']
        sma200 = ma['sma200']
        
        l8 = ema8.iloc[-1]; l21 = ema21.iloc[-1]; l50 = sma50.iloc[-1]; l200 = sma200.iloc[-1]
        
        if price > l8 > l21 > l50:
            trend_desc = "STRONG UPTREND (Price > 8 > 21 > 50)"
            score += 2
            explanations.append("Full Bullish Alignment")
        elif price < l8 < l21 < l50:
            trend_desc = "STRONG DOWNTREND (Price < 8 < 21 < 50)"
            score -= 2
            explanations.append("Full Bearish Alignment")
        elif price > l200:
            trend_desc = "Long Term Bull (Above 200d)"
            score += 1
        else:
            trend_desc = "Choppy / Weak"
            
        # 2. DeMark
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        dm_count = safe_int(bs if bs > ss else ss)
        dm_type = "Buy" if bs > ss else "Sell"
        perf = last.get('Perfected', False)
        
        dm_desc = f"{dm_type} {dm_count}"
        if dm_count == 9:
            score += 3
            perf_txt = "PERFECTED" if perf else "UNPERFECTED"
            dm_desc = f"**{dm_type} 9 ({perf_txt})**"
            explanations.append(f"DeMark 9 Reversal Signal ({perf_txt})")
        elif dm_count >= 1:
            bars = 9 - dm_count
            dm_desc += f" ({bars} days to 9)"

        # 3. RSI
        rsi_val = last['RSI']
        if rsi_val > 70:
            rsi_desc = "Overbought (>70)"
            score += 2 # Counter trend short score
            explanations.append("RSI Overbought (Sell Risk)")
        elif rsi_val < 30:
            rsi_desc = "Oversold (<30)"
            score += 2 # Counter trend buy score
            explanations.append("RSI Oversold (Bounce Likely)")
        else:
            rsi_desc = "Neutral"

        # 4. ADX
        adx_val = adx.iloc[-1]
        if adx_val > 25:
            adx_desc = "Trending"
        else:
            adx_desc = "Non-Trending"

        # 5. Squeeze
        if sq_res:
            sq_desc = f"FIRING ({sq_res['bias']})"
            score += 2
            explanations.append("Volatility Squeeze Firing")
        else:
            sq_desc = "None"

        # Final Direction
        if score >= 4: direction = "STRONG BUY"
        elif score >= 2: direction = "BUY"
        elif score <= -4: direction = "STRONG SELL"
        elif score <= -2: direction = "SELL"
        else: direction = "NEUTRAL"
        
        # Targets
        if "BUY" in direction:
            target = price + (atr * 3)
            stop = price - (atr * 1.5)
        elif "SELL" in direction:
            target = price - (atr * 3)
            stop = price + (atr * 1.5)
        else:
            target = 0; stop = 0

        return {
            'ticker': ticker, 'price': price, 'score': score, 'direction': direction,
            'reasons': explanations,
            'details': {
                'trend': trend_desc,
                'demark': dm_desc,
                'rsi': f"{rsi_val:.1f} ({rsi_desc})",
                'adx': f"{adx_val:.1f} ({adx_desc})",
                'squeeze': sq_desc,
                'macd': "Bullish" if macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1] else "Bearish"
            },
            'plan': {'target': target, 'stop': stop}
        }
    except: return None

def format_detailed_card(res):
    d = res['details']
    p = res['plan']
    
    # Header
    icon = "🟢" if "BUY" in res['direction'] else ("🔴" if "SELL" in res['direction'] else "⚪")
    msg = f"═════════════════════════\n"
    msg += f"{icon} *{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"**Rating:** {res['direction']} ({res['score']}/10)\n"
    
    if res['reasons']:
        msg += f"🔥 **Drivers:** {', '.join(res['reasons'])}\n"
    
    msg += f"─────────────────────────\n"
    
    # 3-Horizon Outlook
    msg += f"🕰️ **TIME HORIZONS:**\n"
    # Short Term: Driven by RSI/DeMark
    st_bias = "Bullish" if "BUY" in res['direction'] else "Bearish"
    msg += f"   • Short (1-2w): {st_bias} (Signal Based)\n"
    # Med Term: Driven by MACD/50d
    mt_bias = "Bullish" if "Bullish" in d['macd'] else "Bearish"
    msg += f"   • Med (2-3m):   {mt_bias} ({d['macd']} MACD)\n"
    # Long Term: Driven by 200d
    lt_bias = "Bullish" if "Bull" in d['trend'] else "Bearish"
    msg += f"   • Long (6m+):   {lt_bias}\n\n"
    
    # Technical Deep Dive
    msg += f"📊 **TECHNICAL DETAILS:**\n"
    msg += f"   • **Trend:** {d['trend']}\n"
    msg += f"   • **DeMark:** {d['demark']}\n"
    msg += f"   • **RSI:** {d['rsi']}\n"
    msg += f"   • **ADX:** {d['adx']}\n"
    if d['squeeze'] != "None": msg += f"   • **Vol:** {d['squeeze']}\n"
        
    # Plan
    if res['direction'] != "NEUTRAL":
        msg += f"─────────────────────────\n"
        msg += f"🎯 **Target:** {fmt_price(p['target'])}\n"
        msg += f"🛑 **Stop:** {fmt_price(p['stop'])}\n"
        
    return msg + "\n"

def format_ranking_card(res, title):
    d = res['details']
    msg = f"═════════════════════════\n"
    msg += f"🚨 *{res['ticker']}* ({res['direction']})\n"
    msg += f"👉 *{title}*\n"
    msg += f"   • Logic: {', '.join(res['reasons'])}\n"
    msg += f"   • Trend: {d['trend']}\n"
    msg += f"🎯 {fmt_price(res['plan']['target'])} | 🛑 {fmt_price(res['plan']['stop'])}\n"
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
    port_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in priority_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: port_results.append(res)
            
    port_results.sort(key=lambda x: x['ticker'] in CURRENT_PORTFOLIO, reverse=True)
    
    msg = "💼 *PORTFOLIO & WATCHLIST*\n"
    for r in port_results:
        msg += format_detailed_card(r)
    send_telegram(msg)

    # 2. SAMPLER (Test Mode)
    others = [t for t in STRATEGIC_TICKERS if t not in priority_list]
    scan_batch = random.sample(others, 5)
    
    print(f"Scanning {len(scan_batch)} Random Tickers...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in scan_batch}
        for future in as_completed(future_map):
            res = future.result()
            if res: results.append(res)
            
    # Rankings
    power = [r for r in results if abs(r['score']) >= 4]
    if power:
        msg = "🔥 *POWER RANKINGS*\n"
        for r in power: msg += format_ranking_card(r, "High Conviction Setup")
        send_telegram(msg)
        
    print("🛑 SAMPLE COMPLETE. Exiting.")
    exit()
