# main.py - Institutional Batch Engine (Full Universe)
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import calc_rsi, calc_squeeze, calc_demark, calc_shannon, calc_adx
from utils import send_telegram, fmt_price

# --- CONFIGURATION ---
BATCH_SIZE = 40        # Safety limit for Tiingo (50/hr)
SLEEP_TIME = 3660      # 61 minutes
MAX_WORKERS = 1        # Serial processing

# --- ASSETS ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']

STRATEGIC_TICKERS = [
    # -- Meme / PolitiFi --
    'PENGU-USD', 'FARTCOIN-USD', 'DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'TRUMP-USD',
    
    # -- Crypto: Majors --
    'BTC-USD', 'ETH-USD', 'SOL-USD',
    
    # -- Crypto: Miners & Infrastructure --
    'BTDR', 'MARA', 'RIOT', 'HUT', 'CLSK', 'IREN', 'CIFR', 'BTBT',
    'WYFI', 'CORZ', 'CRWV', 'APLD', 'NBIS', 'WULF', 'HIVE', 'BITF',
    'WGMI', 'MNRS', 'OWNB', 'BMNR', 'SBET', 'FWDI', 'BKKT',
    
    # -- Crypto: ETFs & Proxies --
    'IBIT', 'ETHA', 'BITQ', 'BSOL', 'GSOL', 'SOLT',
    'MSTR', 'COIN', 'HOOD', 'GLXY', 'STKE', 'DFDV', 'NODE', 'GEMI', 'BLSH',
    'CRCL',
    
    # -- Commodities --
    'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F', 'PA=F',
    'GLD', 'SLV', 'PALL', 'PPLT', 'NIKL', 'LIT', 'ILIT', 'REMX',
    
    # -- Energy / Grid / Uranium --
    'VOLT', 'GRID', 'EQT', 'TAC', 'BE', 'OKLO', 'SMR', 'NEE', 
    'URA', 'SRUUF', 'CCJ', 'KAMJY', 'UNL',
    
    # -- Tech / AI / Mag 7 --
    'NVDA', 'SMH', 'SMHX', 'TSM', 'AVGO', 'QCOM', 'MU', 'AMD', 'TER', 
    'NOW', 'AXON', 'SNOW', 'PLTR', 'GOOG', 'MSFT', 'META', 'AMZN', 'AAPL',
    'TSLA', 'NFLX', 'SPOT', 'SHOP', 'UBER', 'DASH', 'NET', 'DXCM', 'ETSY',
    'SQ', 'FIG', 'MAGS', 'MTUM', 'IVES',
    
    # -- Innovation --
    'ARKK', 'ARKF', 'ARKG', 'GRNY', 'GRNI', 'GRNJ', 'XBI', 'XHB',
    
    # -- Sectors --
    'XLK', 'XLI', 'XLU', 'XLRE', 'XLB', 'XLV', 'XLF', 'XLE', 'XLP', 'XLY', 'XLC',
    
    # -- International --
    'BABA', 'JD', 'BIDU', 'PDD', 'XIACY', 'BYDDY', 'LKNCY', 'TCEHY',
    'MCHI', 'INDA', 'EWZ', 'EWJ', 'EWG', 'EWU', 'EWY', 'EWW', 'EWT', 'EWC', 'EEM',
    'AMX', 'PBR', 'VALE', 'NSRGY', 'DEO',
    
    # -- Financials / Other --
    'BLK', 'STT', 'ARES', 'SOFI', 'PYPL', 'IBKR', 'WU',
    'RXRX', 'SDGR', 'TEM', 'ABSI', 'DNA', 'TWST', 'GLW', 
    'KHC', 'LULU', 'YETI', 'DLR', 'EQIX', 'ORCL', 'LSF'
]

def analyze_ticker(ticker, regime):
    try:
        # Create fresh client for every batch to avoid stale sessions
        client = get_tiingo_client()
        df = safe_download(ticker, client)
        if df is None: return None

        # Liquidity Check (Skip <$500k volume unless it's a major index/crypto)
        # Exception for Futures (=F) which sometimes report volume differently
        if '=F' not in ticker:
            last_vol = df['Volume'].iloc[-5:].mean() * df['Close'].iloc[-1]
            if last_vol < 500000 and '-USD' not in ticker: return None 

        # Indicators
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
        
        # --- TRADE CARD ---
        setup = {'active': False, 'msg': "None", 'target': 0, 'stop': 0, 'time': ''}
        score = 0
        
        # 1. DeMark (Primary)
        if last.get('Buy_Setup') == 9:
            perf = last.get('Perfected')
            score += 3 if perf else 2
            setup = {
                'active': True, 'msg': f"DeMark BUY 9 {'(Perfected)' if perf else ''}",
                'target': price + (atr * 3), 'stop': price - (atr * 1.5), 'time': '1-4 Weeks'
            }
        elif last.get('Sell_Setup') == 9:
            perf = last.get('Perfected')
            score += 3 if perf else 2
            setup = {
                'active': True, 'msg': f"DeMark SELL 9 {'(Perfected)' if perf else ''}",
                'target': price - (atr * 3), 'stop': price + (atr * 1.5), 'time': '1-4 Weeks'
            }
            
        # 2. Squeeze
        if sq_res and not setup['active']:
            score += 2
            d = sq_res['bias']
            setup = {
                'active': True, 'msg': f"TTM Squeeze ({d})",
                'target': price + (atr * 4) if d == "BULLISH" else price - (atr * 4),
                'stop': price - (atr * 2) if d == "BULLISH" else price + (atr * 2), 'time': '3-10 Days'
            }
            
        # 3. RSI Extremes
        if last['RSI'] < 30: 
            score += 2
            if not setup['active']:
                setup = {'active': True, 'msg': "RSI Oversold", 'target': price+(atr*2), 'stop': price-atr, 'time': '1-3 Days'}
        elif last['RSI'] > 70:
            score += 2
            if not setup['active']:
                setup = {'active': True, 'msg': "RSI Overbought", 'target': price-(atr*2), 'stop': price+atr, 'time': '1-3 Days'}
                
        if shannon['breakout']: score += 3
        if regime == 'RISK_OFF' and "BUY" in setup['msg']: score -= 1
        
        return {
            'ticker': ticker, 'price': price, 'trend': trend, 'score': score,
            'setup': setup, 'squeeze': sq_res, 'shannon': shannon, 
            'rsi': last['RSI'], 'adx': adx.iloc[-1]
        }
    except: return None

def format_alert(res):
    s = res['setup']
    icon = "🟢" if res['trend'] == "BULLISH" else "🔴"
    msg = f"{icon} *{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"Score: {res['score']}/10 | Trend: {res['trend']}\n"
    
    if s['active']:
        msg += f"🎯 *ACTION:* {s['msg']}\n"
        msg += f"   • Target: {fmt_price(s['target'])}\n"
        msg += f"   • Stop: {fmt_price(s['stop'])}\n"
        msg += f"   • Horizon: {s['time']}\n"
    elif res['ticker'] in CURRENT_PORTFOLIO:
        msg += f"   • Status: No Signal\n"
            
    msg += f"   • RSI: {res['rsi']:.1f} | ADX: {res['adx']:.1f}\n"
    if res['shannon']['breakout']: msg += f"   • Momentum: BREAKOUT 🚀\n"
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL BATCH SCANNER"); print("="*60)
    
    # 1. Organization
    full_list = list(set(CURRENT_PORTFOLIO + STRATEGIC_TICKERS))
    # Move Portfolio to front
    for t in CURRENT_PORTFOLIO:
        if t in full_list: full_list.remove(t)
    full_list = CURRENT_PORTFOLIO + full_list
    
    batches = [full_list[i:i + BATCH_SIZE] for i in range(0, len(full_list), BATCH_SIZE)]
    est_time = (len(batches) - 1) * 61
    
    send_telegram(f"🏗️ *SCAN STARTED*\nAssets: {len(full_list)}\nBatches: {len(batches)}\nEst. Time: {est_time} mins")

    # 2. Macro
    macro = get_macro()
    regime = "NEUTRAL"
    if macro['net_liq'] and macro['net_liq'] > 0: regime = "RISK_ON"

    # 3. Execution
    all_results = []
    
    for i, batch in enumerate(batches):
        print(f"Processing Batch {i+1}/{len(batches)}...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(analyze_ticker, t, regime): t for t in batch}
            for future in as_completed(future_map):
                res = future.result()
                if res: all_results.append(res)

        # IMMEDIATE REPORT: Send Portfolio updates ASAP
        if i == 0:
            port_msg = "💼 *PORTFOLIO UPDATE*\n\n"
            found_port = False
            for r in all_results:
                if r['ticker'] in CURRENT_PORTFOLIO:
                    found_port = True
                    port_msg += format_alert(r)
            if found_port: send_telegram(port_msg)

        if i < len(batches) - 1:
            print(f"Sleeping {SLEEP_TIME}s..."); time.sleep(SLEEP_TIME)

    # 4. Final Scanner Report
    print("Generating Final Report...")
    scan_msg = "🚨 *HIGH CONVICTION OPPORTUNITIES*\n\n"
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    found_opp = False
    for r in all_results:
        if r['ticker'] not in CURRENT_PORTFOLIO and r['score'] >= 4:
            found_opp = True
            scan_msg += format_alert(r)
            
    if found_opp:
        if len(scan_msg) > 4000:
            parts = [scan_msg[i:i+4000] for i in range(0, len(scan_msg), 4000)]
            for p in parts: send_telegram(p)
        else:
            send_telegram(scan_msg)
    else:
        send_telegram("✅ Scan Complete. No other setups found.")
        
    print("DONE")
