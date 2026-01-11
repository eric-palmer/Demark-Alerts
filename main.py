# main.py - Institutional Engine (Detailed Portfolio Report)
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

def analyze_ticker(ticker, regime):
    try:
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: return None

        # Filter Illiquid (except crypto/futures)
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
        
        # --- SCORE & SIGNAL LOGIC ---
        score = 0
        active_signal = None # Stores the "Headline" signal if any
        
        # 1. DeMark
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        dm_status = f"Buy {int(bs)}" if bs > ss else f"Sell {int(ss)}"
        
        if bs == 9:
            perf = last.get('Perfected')
            score += 3 if perf else 2
            dm_status = f"BUY 9 {'(Perfected)' if perf else '(Unperfected)'}"
            active_signal = dm_status
        elif ss == 9:
            perf = last.get('Perfected')
            score += 3 if perf else 2
            dm_status = f"SELL 9 {'(Perfected)' if perf else '(Unperfected)'}"
            active_signal = dm_status
            
        # 2. Squeeze
        if sq_res:
            score += 2
            if not active_signal: active_signal = f"TTM Squeeze ({sq_res['bias']})"
            
        # 3. RSI
        rsi_val = last['RSI']
        if rsi_val < 30: 
            score += 2
            if not active_signal: active_signal = "RSI Oversold"
        elif rsi_val > 70:
            score += 2
            if not active_signal: active_signal = "RSI Overbought"
            
        if shannon['breakout']: score += 3
        if regime == 'RISK_OFF' and active_signal and "BUY" in active_signal: score -= 1
        
        # --- TARGET CALCULATION (For Portfolio View) ---
        # If no active signal, default to Trend Following Logic
        if "BUY" in str(active_signal) or (not active_signal and trend == "BULLISH"):
            target = price + (atr * 3)
            stop = price - (atr * 1.5)
            horizon = "1-3 Weeks"
        else:
            target = price - (atr * 3)
            stop = price + (atr * 1.5)
            horizon = "1-3 Weeks"
            
        return {
            'ticker': ticker, 'price': price, 'trend': trend, 'score': score,
            'signal': active_signal, 'squeeze': sq_res, 'shannon': shannon, 
            'rsi': rsi_val, 'adx': adx.iloc[-1], 'demark_status': dm_status,
            'target': target, 'stop': stop, 'horizon': horizon
        }
    except: return None

def format_portfolio_card(res):
    """Detailed Card for Portfolio Holdings"""
    icon = "🟢" if res['trend'] == "BULLISH" else "🔴"
    
    msg = f"{icon} *{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"Score: {res['score']}/10 | ADX: {res['adx']:.1f} ({'Strong' if res['adx']>25 else 'Weak'})\n"
    msg += f"────────────────\n"
    
    # Detailed Indicator Breakdown
    msg += f"• **DeMark:** {res['demark_status']}\n"
    
    rsi_state = "Neutral"
    if res['rsi'] > 70: rsi_state = "Overbought ⚠️"
    elif res['rsi'] < 30: rsi_state = "Oversold 🛒"
    msg += f"• **RSI:** {res['rsi']:.1f} ({rsi_state})\n"
    
    sq_txt = f"{res['squeeze']['bias']} (Firing)" if res['squeeze'] else "None"
    msg += f"• **Squeeze:** {sq_txt}\n"
    
    if res['shannon']['breakout']:
        msg += f"• **Momentum:** 🚀 BREAKOUT DETECTED\n"
        
    msg += f"────────────────\n"
    msg += f"🎯 **Target:** {fmt_price(res['target'])}\n"
    msg += f"🛑 **Stop:** {fmt_price(res['stop'])}\n"
    msg += f"⏳ **Horizon:** {res['horizon']}\n"
    
    return msg + "\n"

def format_scanner_alert(res):
    """Brief Card for Scanner Hits"""
    msg = f"🚨 *{res['ticker']}* ({res['score']}/10)\n"
    msg += f"Signal: {res['signal']}\n"
    msg += f"Target: {fmt_price(res['target'])} | Stop: {fmt_price(res['stop'])}\n"
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

    # Macro
    regime = "NEUTRAL"
    try:
        macro = get_macro()
        if macro['net_liq'] is not None:
            nl = macro['net_liq']
            if isinstance(nl, pd.Series):
                if nl.pct_change(63).iloc[-1] > 0: regime = "RISK_ON"
                else: regime = "RISK_OFF"
    except: pass
    send_telegram(f"📊 *REGIME: {regime}*")

    # Execution
    all_results = []
    
    for i, batch in enumerate(batches):
        print(f"Processing Batch {i+1}/{len(batches)}...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(analyze_ticker, t, regime): t for t in batch}
            for future in as_completed(future_map):
                res = future.result()
                if res: all_results.append(res)

        # IMMEDIATE REPORT: Portfolio (Detailed View)
        if i == 0:
            port_msg = "💼 *PORTFOLIO DEEP DIVE*\n\n"
            found_port = False
            for r in all_results:
                if r['ticker'] in CURRENT_PORTFOLIO:
                    found_port = True
                    port_msg += format_portfolio_card(r)
            if found_port: send_telegram(port_msg)

        if i < len(batches) - 1:
            print(f"Sleeping {SLEEP_TIME}s..."); time.sleep(SLEEP_TIME)

    # Final Report
    print("Generating Final Report...")
    scan_msg = "🚨 *HIGH CONVICTION OPPORTUNITIES*\n\n"
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    found_opp = False
    for r in all_results:
        # Show High Scores NOT in portfolio
        if r['ticker'] not in CURRENT_PORTFOLIO and r['score'] >= 4:
            found_opp = True
            scan_msg += format_scanner_alert(r)
            
    if found_opp:
        if len(scan_msg) > 4000:
            parts = [scan_msg[i:i+4000] for i in range(0, len(scan_msg), 4000)]
            for p in parts: send_telegram(p)
        else:
            send_telegram(scan_msg)
    else:
        send_telegram("✅ Scan Complete. No other setups found.")
        
    print("DONE")
