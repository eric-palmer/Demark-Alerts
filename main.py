# main.py - Institutional Strategy Engine (Batch Mode)
import time
import math
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import safe_download, get_macro, get_tiingo_client
from indicators import calc_rsi, calc_squeeze, calc_demark, calc_shannon, calc_adx
from utils import send_telegram, fmt_price

# --- YOUR UNIVERSE ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']

STRATEGIC_TICKERS = [
    # Meme / Crypto Proxies
    'PENGU-USD', 'FARTCOIN-USD', 'DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'TRUMP-USD',
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BTDR', 'MARA', 'RIOT', 'HUT', 'CLSK', 
    'IREN', 'CIFR', 'BTBT', 'WYFI', 'CORZ', 'CRWV', 'APLD', 'NBIS', 'WULF', 
    'HIVE', 'BITF', 'WGMI', 'MNRS', 'OWNB', 'BMNR', 'SBET', 'FWDI', 'BKKT',
    'IBIT', 'ETHA', 'BITQ', 'BSOL', 'GSOL', 'SOLT', 'MSTR', 'COIN', 'HOOD', 
    'GLXY', 'STKE', 'DFDV', 'NODE', 'GEMI', 'BLSH', 'CRCL',
    # Commodities / Energy
    'GLD', 'SLV', 'PALL', 'PPLT', 'NIKL', 'LIT', 'ILIT', 'REMX', 'VOLT', 'GRID', 
    'EQT', 'TAC', 'BE', 'OKLO', 'SMR', 'NEE', 'URA', 'SRUUF', 'CCJ', 'KAMJY', 'UNL',
    # Tech / Mag 7 / Growth
    'NVDA', 'SMH', 'SMHX', 'TSM', 'AVGO', 'QCOM', 'MU', 'AMD', 'TER', 'NOW', 
    'AXON', 'SNOW', 'PLTR', 'GOOG', 'MSFT', 'META', 'AMZN', 'AAPL', 'TSLA', 
    'NFLX', 'SPOT', 'SHOP', 'UBER', 'DASH', 'NET', 'DXCM', 'ETSY', 'SQ', 
    'FIG', 'MAGS', 'MTUM', 'IVES', 'ARKK', 'ARKF', 'ARKG', 'GRNY', 'GRNI', 
    'GRNJ', 'XBI', 'XHB',
    # Sectors / Intl
    'XLK', 'XLI', 'XLU', 'XLRE', 'XLB', 'XLV', 'XLF', 'XLE', 'XLP', 'XLY', 
    'XLC', 'BABA', 'JD', 'BIDU', 'PDD', 'XIACY', 'BYDDY', 'LKNCY', 'TCEHY', 
    'MCHI', 'INDA', 'EWZ', 'EWJ', 'EWG', 'EWU', 'EWY', 'EWW', 'EWT', 'EWC', 
    'EEM', 'AMX', 'PBR', 'VALE', 'NSRGY', 'DEO',
    # Financials / Other
    'BLK', 'STT', 'ARES', 'SOFI', 'PYPL', 'IBKR', 'WU', 'RXRX', 'SDGR', 
    'TEM', 'ABSI', 'DNA', 'TWST', 'GLW', 'KHC', 'LULU', 'YETI', 'DLR', 
    'EQIX', 'ORCL', 'LSF'
]

# --- Config ---
BATCH_SIZE = 40        # Stay safely under 50/hr limit
SLEEP_TIME = 3660      # 61 minutes
MAX_WORKERS = 1        # Serial processing to be gentle on API

def analyze_ticker(ticker, client, regime):
    try:
        df = safe_download(ticker, client)
        if df is None: return None

        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        sq_res = calc_squeeze(df)
        shannon = calc_shannon(df)
        adx = calc_adx(df)
        
        last = df.iloc[-1]
        price = last['Close']
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        
        # Trend Context
        if pd.isna(sma_200): sma_200 = price 
        trend = "BULLISH" if price > sma_200 else "BEARISH"
        
        # Signal Verdict
        verdict = "WAIT"
        score = 0
        setup = None
        
        # DeMark Logic
        if last.get('Buy_Setup') == 9:
            perf = last.get('Perfected')
            setup = {'type': 'BUY', 'perf': perf, 'count': 9}
            score += 3 if perf else 2
        elif last.get('Sell_Setup') == 9:
            perf = last.get('Perfected')
            setup = {'type': 'SELL', 'perf': perf, 'count': 9}
            score += 3 if perf else 2
        else:
            # Show raw counts for portfolio
            bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
            cnt = f"Buy {int(bs)}" if bs > ss else f"Sell {int(ss)}"
            setup = {'type': 'COUNT', 'perf': False, 'count': cnt}

        # Confluence
        if last['RSI'] < 30: score += 2
        if last['RSI'] > 70: score += 2
        if sq_res: score += 2
        if shannon['breakout']: score += 3
        
        # Macro Filter
        if regime == 'RISK_OFF' and setup['type'] == 'BUY': score -= 1
        
        # Text Verdict
        if score >= 4: verdict = "ACTION"
        
        return {
            'ticker': ticker, 'price': price, 'trend': trend, 'score': score,
            'setup': setup, 'squeeze': sq_res, 'shannon': shannon, 
            'rsi': last['RSI'], 'adx': adx.iloc[-1]
        }
    except: return None

def format_portfolio(res):
    s = res['setup']
    icon = "🟢" if res['trend'] == "BULLISH" else "🔴"
    
    msg = f"{icon} *{res['ticker']}* @ {fmt_price(res['price'])}\n"
    
    # DeMark Detail
    if s['type'] in ['BUY', 'SELL']:
        perf_icon = "⭐ PERFECTED" if s['perf'] else "⚪ Imperfected"
        msg += f"   • DeMark: {s['type']} 9 ({perf_icon})\n"
    else:
        msg += f"   • DeMark: {s['count']}\n"
        
    msg += f"   • Trend: {res['trend']} (ADX: {res['adx']:.1f})\n"
    msg += f"   • RSI: {res['rsi']:.1f}\n"
    
    if res['squeeze']: 
        msg += f"   • Squeeze: {res['squeeze']['bias']} Ready ⚠️\n"
    if res['shannon']['breakout']:
        msg += f"   • Momentum: BREAKOUT 🚀\n"
        
    return msg + "\n"

def format_scanner(res):
    # Only show if meaningful
    s = res['setup']
    msg = f"*{res['ticker']}* ({res['score']}/10)\n"
    
    if s['type'] in ['BUY', 'SELL']:
        perf = "⭐" if s['perf'] else "(Imperfect)"
        msg += f"   • {s['type']} 9 {perf}\n"
        
    if res['shannon']['breakout']:
        msg += f"   • Momentum Breakout 🚀\n"
        
    if res['squeeze']:
        msg += f"   • {res['squeeze']['bias']} Squeeze\n"
        
    return msg + "\n"

if __name__ == "__main__":
    print("="*60)
    print("INSTITUTIONAL BATCH SCANNER")
    print("="*60)
    
    # 1. Init
    client = get_tiingo_client()
    full_list = list(set(CURRENT_PORTFOLIO + STRATEGIC_TICKERS))
    
    # Prioritize Portfolio
    # Put portfolio items at the front of the list so they scan first
    for t in CURRENT_PORTFOLIO:
        if t in full_list: full_list.remove(t)
    full_list = CURRENT_PORTFOLIO + full_list
    
    # Batching logic
    batches = [full_list[i:i + BATCH_SIZE] for i in range(0, len(full_list), BATCH_SIZE)]
    est_time = (len(batches) - 1) * 61
    
    send_telegram(f"🏗️ *SCANNING STARTED*\nTargets: {len(full_list)}\nBatches: {len(batches)}\nEst. Time: {est_time} mins")

    # 2. Macro
    macro = get_macro()
    regime = "NEUTRAL"
    if macro['net_liq'] and macro['net_liq'] > 0: regime = "RISK_ON"

    # 3. Execution Loop
    all_results = []
    
    for i, batch in enumerate(batches):
        print(f"Processing Batch {i+1}/{len(batches)}...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(analyze_ticker, t, client, regime): t for t in batch}
            for future in as_completed(future_map):
                res = future.result()
                if res: all_results.append(res)
        
        # Sleep if not last batch
        if i < len(batches) - 1:
            print(f"Sleeping {SLEEP_TIME}s...")
            time.sleep(SLEEP_TIME)

    # 4. Final Reporting
    print("Generating Report...")
    
    # Portfolio Report (Detailed)
    port_msg = "💼 *PORTFOLIO REPORT*\n\n"
    for r in all_results:
        if r['ticker'] in CURRENT_PORTFOLIO:
            port_msg += format_portfolio(r)
    send_telegram(port_msg)
    
    # Power Rankings (Scanner)
    # Filter for Score >= 4 OR DeMark 9s
    power_picks = [r for r in all_results 
                   if r['score'] >= 4 
                   or r['setup']['type'] in ['BUY', 'SELL']]
                   
    power_picks.sort(key=lambda x: x['score'], reverse=True)
    
    if power_picks:
        # Split into chunks
        chunks = [power_picks[i:i + 10] for i in range(0, len(power_picks), 10)]
        for chunk in chunks:
            scan_msg = "🚨 *POWER RANKINGS*\n\n"
            for r in chunk:
                scan_msg += format_scanner(r)
            send_telegram(scan_msg)
    else:
        send_telegram("✅ No high-conviction setups found.")
        
    print("DONE")
