# main.py - Institutional Engine (3-Horizon Analysis)
import time
import pandas as pd
import numpy as np
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
SHORT_WATCHLIST = ['LAC']

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

        # --- CALCULATIONS ---
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
        
        # --- 3-HORIZON ANALYSIS ---
        
        # 1. LONG TERM (6m+): Moving Averages
        sma50 = ma['sma50'].iloc[-1]
        sma200 = ma['sma200'].iloc[-1]
        if price > sma200:
            long_term = "Bullish"
            lt_icon = "🟢"
        else:
            long_term = "Bearish"
            lt_icon = "🔴"
            
        # 2. MED TERM (2-3m): MACD + 50d
        macd_val = macd_data['macd'].iloc[-1]
        macd_sig = macd_data['signal'].iloc[-1]
        
        med_term = "Neutral"
        mt_icon = "⚪"
        if price > sma50 and macd_val > macd_sig:
            med_term = "Bullish Momentum"
            mt_icon = "🟢"
        elif price < sma50 and macd_val < macd_sig:
            med_term = "Bearish Momentum"
            mt_icon = "🔴"
        elif macd_val > macd_sig:
            med_term = "Recovering"
            mt_icon = "🟡"
            
        # 3. SHORT TERM (1-2w): DeMark + RSI + Squeeze
        bs = last.get('Buy_Setup', 0); ss = last.get('Sell_Setup', 0)
        dm_count = safe_int(bs if bs > ss else ss)
        dm_type = "Buy" if bs > ss else "Sell"
        
        short_term = "Neutral"
        st_icon = "⚪"
        setup_msg = "No Signal"
        
        score = 0
        
        # Scoring & Signals
        if dm_count == 9:
            score += 3
            short_term = "Reversal Likely"
            st_icon = "⚠️"
            perf = "Perfected" if last.get('Perfected') else "Unperfected"
            setup_msg = f"DeMark {dm_type} 9 ({perf})"
        elif dm_count >= 1:
            setup_msg = f"{dm_type} Count {dm_count}"
            
        if sq_res:
            score += 2
            short_term = "Explosive Move"
            st_icon = "🚀"
            setup_msg = f"TTM Squeeze ({sq_res['bias']})"
            
        rsi_val = last['RSI']
        if rsi_val < 30: 
            score += 2
            short_term = "Oversold Bounce"
            st_icon = "🟢"
        elif rsi_val > 70:
            score += 2
            short_term = "Overbought Pullback"
            st_icon = "🔴"
            
        if shannon['breakout']: score += 3
        if "RISK_OFF" in regime and "Bullish" in med_term: score -= 1
        
        # Targets based on Horizon
        targets = {
            'stop': price - (atr * 2) if "Bull" in med_term else price + (atr * 2),
            'target': price + (atr * 3) if "Bull" in med_term else price - (atr * 3)
        }
        
        return {
            'ticker': ticker, 'price': price, 'score': score,
            'horizons': {
                'long': f"{lt_icon} {long_term}",
                'med': f"{mt_icon} {med_term}",
                'short': f"{st_icon} {short_term}"
            },
            'techs': {
                'rsi': rsi_val, 'adx': adx.iloc[-1], 
                'demark': f"{dm_type} {dm_count}",
                'macd': "Bullish" if macd_val > macd_sig else "Bearish",
                'sma50': sma50, 'sma200': sma200
            },
            'setup': setup_msg,
            'plan': targets
        }
    except: return None

def format_analyst_card(res, title="ANALYSIS"):
    msg = f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"Score: {res['score']}/10 | Trend: {res['horizons']['med']}\n"
    msg += f"────────────────\n"
    
    # Time Horizons
    msg += f"🕰️ **Outlook:**\n"
    msg += f"   • Short (1-2w): {res['horizons']['short']}\n"
    msg += f"   • Med (2-3m):   {res['horizons']['med']}\n"
    msg += f"   • Long (6m+):   {res['horizons']['long']}\n"
    
    # Technicals
    t = res['techs']
    msg += f"\n📊 **Technicals:**\n"
    msg += f"   • DeMark: {t['demark']}\n"
    msg += f"   • RSI: {t['rsi']:.1f} | ADX: {t['adx']:.1f}\n"
    msg += f"   • MACD: {t['macd']}\n"
    
    # Plan
    msg += f"────────────────\n"
    msg += f"🎯 **Target:** {fmt_price(res['plan']['target'])}\n"
    msg += f"🛑 **Stop:** {fmt_price(res['plan']['stop'])}\n"
    
    return msg + "\n"

if __name__ == "__main__":
    print("="*60); print("INSTITUTIONAL BATCH SCANNER"); print("="*60)
    
    full_list = list(set(CURRENT_PORTFOLIO + STRATEGIC_TICKERS + SHORT_WATCHLIST))
    # Remove priority items from general pool
    for t in CURRENT_PORTFOLIO + SHORT_WATCHLIST:
        if t in full_list: full_list.remove(t)
    
    # Create batches for the rest
    batches = [full_list[i:i + BATCH_SIZE] for i in range(0, len(full_list), BATCH_SIZE)]
    
    # Macro
    try:
        macro = get_macro()
        regime, desc = get_market_radar_regime(macro)
    except:
        regime = "NEUTRAL"; desc = "Macro Data Failed"
    send_telegram(f"📊 *MARKET REGIME: {regime}*\n{desc}")

    # 1. PRIORITY SCAN: Portfolio + Short Watchlist
    print("Scanning Priorities...")
    priority_list = CURRENT_PORTFOLIO + SHORT_WATCHLIST
    port_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in priority_list}
        for future in as_completed(future_map):
            res = future.result()
            if res: port_results.append(res)
            
    # Format Priorities
    msg = "💼 *PORTFOLIO & WATCHLIST*\n\n"
    for r in port_results:
        # Check if it's watchlist or portfolio for header? 
        # (Simplified: Just list them all, user knows which is which)
        if r['ticker'] in SHORT_WATCHLIST:
            msg += "👀 **WATCHLIST: " + r['ticker'] + "**\n"
        msg += format_analyst_card(r)
    send_telegram(msg)

    # DEBUG EXIT (Remove this when ready for full run)
    print("🛑 DEBUG STOP: Priorities sent.")
    exit()
