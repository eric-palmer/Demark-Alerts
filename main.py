# main.py - Institutional Engine (Detailed Analyst Mode)
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
        
        last = df.iloc[-1]
        price = last['Close']
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        if pd.isna(sma_200): sma_200 = price 
        trend = "BULLISH" if price > sma_200 else "BEARISH"
        
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr): atr = price * 0.02
        
        setup = {'active': False, 'msg': "None", 'target': 0, 'stop': 0, 'time': '', 'analysis': ''}
        score = 0
        
        # --- DeMark Status ---
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        dm_count = safe_int(bs if bs > ss else ss)
        dm_type = "Buy" if bs > ss else "Sell"
        
        # Detailed DeMark Analysis
        if dm_count == 9:
            perf = last.get('Perfected')
            status = "PERFECTED" if perf else "UNPERFECTED"
            action = "Reversal Imminent" if perf else "Wait for Confirmation"
            dm_desc = f"{dm_type} Setup 9 ({status}) - {action}"
            
            setup = {'active': True, 'msg': dm_desc, 
                     'target': price+(atr*3) if dm_type=="Buy" else price-(atr*3), 
                     'stop': price-(atr*1.5) if dm_type=="Buy" else price+(atr*1.5), 
                     'time': '1-4 Weeks'}
            score += 3 if perf else 2
        elif dm_count >= 1:
            dm_desc = f"{dm_type} Setup {dm_count}/9 (Building)"
        else:
            dm_desc = "Neutral (No Setup)"

        # --- Squeeze ---
        if sq_res and not setup['active']:
            score += 2
            d = sq_res['bias']
            setup = {'active': True, 'msg': f"TTM Squeeze ({d})", 'target': price+(atr*4) if d=="BULLISH" else price-(atr*4), 'stop': price-(atr*2) if d=="BULLISH" else price+(atr*2), 'time': '3-10 Days'}
            
        # --- RSI ---
        rsi_val = last['RSI']
        rsi_desc = "Neutral"
        if rsi_val < 30: 
            score += 2
            rsi_desc = "Oversold (Buy Zone)"
            if not setup['active']: setup = {'active': True, 'msg': "RSI Oversold", 'target': price+(atr*2), 'stop': price-atr, 'time': '1-3 Days'}
        elif rsi_val > 70:
            score += 2
            rsi_desc = "Overbought (Sell Zone)"
            if not setup['active']: setup = {'active': True, 'msg': "RSI Overbought", 'target': price-(atr*2), 'stop': price+atr, 'time': '1-3 Days'}
                
        if shannon['breakout']: score += 3
        if "RISK_OFF" in regime and "BUY" in setup['msg']: score -= 2
        
        # --- TREND PLAN (If no active signal) ---
        if not setup['active']:
            if trend == "BULLISH":
                setup['msg'] = "Trend: Bullish"
                setup['target'] = price + (atr * 3) # Trend Following Target
                setup['stop'] = price - (atr * 2)   # Trailing Stop
                setup['time'] = "Hold / Trend Follow"
            else:
                setup['msg'] = "Trend: Bearish"
                setup['target'] = price - (atr * 3)
                setup['stop'] = price + (atr * 2)
                setup['time'] = "Avoid / Short"
        
        return {
            'ticker': ticker, 'price': price, 'trend': trend, 'score': score,
            'setup': setup, 'squeeze': sq_res, 'shannon': shannon, 
            'rsi': rsi_val, 'rsi_desc': rsi_desc,
            'adx': adx.iloc[-1], 'demark_desc': dm_desc
        }
    except: return None

def format_portfolio_card(res):
    icon = "🟢" if res['trend'] == "BULLISH" else "🔴"
    
    # ADX Interpretation
    adx_val = res['adx'] if pd.notna(res['adx']) else 0.0
    if adx_val > 25: adx_txt = "Strong Trend"
    else: adx_txt = "Weak/Choppy"
    
    msg = f"{icon} *{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"Score: {res['score']}/10 | ADX: {adx_val:.1f} ({adx_txt})\n"
    msg += f"────────────────\n"
    
    # Detailed Indicators
    msg += f"• **DeMark:** {res['demark_desc']}\n"
    msg += f"• **RSI:** {res['rsi']:.1f} ({res['rsi_desc']})\n"
    
    sq_txt = f"{res['squeeze']['bias']} (Firing)" if res['squeeze'] else "None"
    msg += f"• **Squeeze:** {sq_txt}\n"
    
    if res['shannon']['breakout']: msg += f"• **Momentum:** 🚀 BREAKOUT\n"
        
    msg += f"────────────────\n"
    # Always show Trade Plan for Portfolio
    msg += f"🧠 **Analysis:** {res['setup']['msg']}\n"
    msg += f"   🎯 Target: {fmt_price(res['setup']['target'])}\n"
    msg += f"   🛑 Stop: {fmt_price(res['setup']['stop'])}\n"
    msg += f"   ⏳ Time: {res['setup']['time']}\n"
    
    return msg + "\n"

def format_scanner_alert(res):
    msg = f"🚨 *{res['ticker']}* ({res['score']}/10)\n"
    msg += f"Signal: {res['setup']['msg']}\n"
    msg += f"Target: {fmt_price(res['setup']['target'])} | Stop: {fmt_price(res['setup']['stop'])}\n"
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL BATCH SCANNER"); print("="*60)
    
    # 1. Macro
    try:
        macro = get_macro()
        regime, desc = get_market_radar_regime(macro)
    except:
        regime = "NEUTRAL"; desc = "Macro Data Failed"
    send_telegram(f"📊 *MARKET REGIME: {regime}*\n{desc}")

    # 2. Portfolio Scan (Priority)
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

    # DEBUG EXIT (For fast testing)
    print("🛑 DEBUG STOP: Portfolio sent. Ending early.")
    exit() 
