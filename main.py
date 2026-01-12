# main.py - Institutional Engine (Debug Mode: Portfolio Only)
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import calc_rsi, calc_squeeze, calc_demark, calc_shannon, calc_adx
from utils import send_telegram, fmt_price

# --- CONFIGURATION ---
BATCH_SIZE = 40        
SLEEP_TIME = 3660      
MAX_WORKERS = 1        

# --- ASSETS ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']

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

# --- CRITICAL FIX: Safe Integer Converter ---
def safe_int(val):
    """Prevents crash when converting NaN/None to int"""
    try:
        if pd.isna(val) or val is None: return 0
        return int(float(val))
    except: return 0

def get_market_radar_regime(macro):
    try:
        growth = macro.get('growth')
        inflation = macro.get('inflation')
        
        if growth is None or inflation is None:
            # Fallback
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
        
        last = df.iloc[-1]
        price = last['Close']
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        if pd.isna(sma_200): sma_200 = price 
        trend = "BULLISH" if price > sma_200 else "BEARISH"
        
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr): atr = price * 0.02
        
        setup = {'active': False, 'msg': "None", 'target': 0, 'stop': 0, 'time': ''}
        score = 0
        
        # DeMark
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        
        if bs == 9:
            perf = last.get('Perfected')
            score += 3 if perf else 2
            setup = {'active': True, 'msg': f"DeMark BUY 9 {'(Perfected)' if perf else ''}", 'target': price+(atr*3), 'stop': price-(atr*1.5), 'time': '1-4 Weeks'}
        elif ss == 9:
            perf = last.get('Perfected')
            score += 3 if perf else 2
            setup = {'active': True, 'msg': f"DeMark SELL 9 {'(Perfected)' if perf else ''}", 'target': price-(atr*3), 'stop': price+(atr*1.5), 'time': '1-4 Weeks'}
            
        # Squeeze
        if sq_res and not setup['active']:
            score += 2
            d = sq_res['bias']
            setup = {'active': True, 'msg': f"TTM Squeeze ({d})", 'target': price+(atr*4) if d=="BULLISH" else price-(atr*4), 'stop': price-(atr*2) if d=="BULLISH" else price+(atr*2), 'time': '3-10 Days'}
            
        # RSI
        if last['RSI'] < 30: 
            score += 2
            if not setup['active']: setup = {'active': True, 'msg': "RSI Oversold", 'target': price+(atr*2), 'stop': price-atr, 'time': '1-3 Days'}
        elif last['RSI'] > 70:
            score += 2
            if not setup['active']: setup = {'active': True, 'msg': "RSI Overbought", 'target': price-(atr*2), 'stop': price+atr, 'time': '1-3 Days'}
                
        if shannon['breakout']: score += 3
        
        if "RISK_OFF" in regime and "BUY" in setup['msg']: score -= 2
        
        if not setup['active']:
            if trend == "BULLISH" and last['RSI'] > 50: setup['msg'] = "Trend: Bullish Hold"
            elif trend == "BEARISH": setup['msg'] = "Trend: Bearish Avoid"
            else: setup['msg'] = "Trend: Neutral"
        
        # Calculate final counts safely
        cnt = bs if bs > ss else ss
        
        return {
            'ticker': ticker, 'price': price, 'trend': trend, 'score': score,
            'setup': setup, 'squeeze': sq_res, 'shannon': shannon, 
            'rsi': last['RSI'], 'adx': adx.iloc[-1], 'demark_count': cnt
        }
    except: return None

def format_portfolio_card(res):
    icon = "🟢" if res['trend'] == "BULLISH" else "🔴"
    msg = f"{icon} *{res['ticker']}* @ {fmt_price(res['price'])}\n"
    
    adx_val = res['adx'] if pd.notna(res['adx']) else 0.0
    rsi_val = res['rsi'] if pd.notna(res['rsi']) else 50.0
    
    msg += f"Score: {res['score']}/10 | ADX: {adx_val:.1f}\n"
    msg += f"────────────────\n"
    
    # CRITICAL FIX: safe_int usage
    dm_c = safe_int(res.get('demark_count', 0))
    msg += f"• **DeMark:** Count {dm_c}\n"
    
    rsi_state = "Neutral"
    if rsi_val > 70: rsi_state = "Overbought ⚠️"
    elif rsi_val < 30: rsi_state = "Oversold 🛒"
    msg += f"• **RSI:** {rsi_val:.1f} ({rsi_state})\n"
    
    sq_txt = f"{res['squeeze']['bias']} (Firing)" if res['squeeze'] else "None"
    msg += f"• **Squeeze:** {sq_txt}\n"
    
    if res['shannon']['breakout']: msg += f"• **Momentum:** 🚀 BREAKOUT\n"
        
    msg += f"────────────────\n"
    if res['setup']['active']:
        msg += f"🎯 **Action:** {res['setup']['msg']}\n"
        msg += f"   Target: {fmt_price(res['setup']['target'])}\n"
        msg += f"   Stop: {fmt_price(res['setup']['stop'])}\n"
    else:
        msg += f"**Status:** {res['setup']['msg']}\n"
    
    return msg + "\n"

def format_scanner_alert(res):
    msg = f"🚨 *{res['ticker']}* ({res['score']}/10)\n"
    msg += f"Signal: {res['setup']['msg']}\n"
    msg += f"Target: {fmt_price(res['setup']['target'])} | Stop: {fmt_price(res['setup']['stop'])}\n"
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL BATCH SCANNER"); print("="*60)
    
    full_list = list(set(CURRENT_PORTFOLIO + STRATEGIC_TICKERS))
    for t in CURRENT_PORTFOLIO:
        if t in full_list: full_list.remove(t)
    full_list = CURRENT_PORTFOLIO + full_list
    
    batches = [full_list[i:i + BATCH_SIZE] for i in range(0, len(full_list), BATCH_SIZE)]
    est_time = (len(batches) - 1) * 61
    
    send_telegram(f"🏗️ *SCAN STARTED*\nAssets: {len(full_list)}\nBatches: {len(batches)}\nEst. Time: {est_time} mins")

    try:
        macro = get_macro()
        regime, desc = get_market_radar_regime(macro)
    except Exception as e:
        print(f"Macro Failed: {e}")
        regime = "NEUTRAL"; desc = "Macro Data Failed"
        
    send_telegram(f"📊 *MARKET REGIME: {regime}*\n{desc}")

    print("Scanning Portfolio...")
    port_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in CURRENT_PORTFOLIO}
        for future in as_completed(future_map):
            res = future.result()
            if res: port_results.append(res)
            
    port_msg = "💼 *PORTFOLIO DEEP DIVE*\n\n"
    for r in port_results:
        port_msg += format_portfolio_card(r)
    send_telegram(port_msg)

    # --- DEBUG EXIT ---
    # This stops the script right here so you don't wait 4 hours.
    print("🛑 DEBUG STOP: Portfolio sent. Ending early.")
    exit()
    # ------------------

    # (Code below is skipped)
    remaining_list = [t for t in STRATEGIC_TICKERS if t not in CURRENT_PORTFOLIO]
    batches = [remaining_list[i:i + BATCH_SIZE] for i in range(0, len(remaining_list), BATCH_SIZE)]
    # ... rest of loop
