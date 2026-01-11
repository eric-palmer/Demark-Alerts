# main.py - Institutional Trading Engine with State Management
import os
import sqlite3
import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Local Modules ---
from data_fetcher import safe_download, get_macro, get_futures
from indicators import (calc_rsi, calc_squeeze, calc_demark, 
                        calc_shannon, calc_macd, calc_stoch, calc_adx)
from utils import send_telegram, fmt_price

# --- Configuration ---
DB_FILE = "trading_state.db"
MAX_WORKERS = 6  # Reduced to prevent API choking
RISK_UNIT = 10000 # Dollar amount for position sizing calculations

# --- Ticker Universe ---
CURRENT_PORTFOLIO = ['SLV', 'DJT']
STRATEGIC_TICKERS = [
    'DJT', 'DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'BTC-USD', 'ETH-USD', 'SOL-USD',
    'BTDR', 'MARA', 'RIOT', 'HUT', 'CLSK', 'IREN', 'CIFR', 'BTBT', 'WYFI', 'CORZ',
    'CRWV', 'APLD', 'NBIS', 'WULF', 'HIVE', 'BITF', 'IBIT', 'ETHA', 'BITQ', 'BSOL',
    'GSOL', 'SOLT', 'MSTR', 'COIN', 'HOOD', 'GLXY', 'STKE', 'DFDV', 'NODE', 'GEMI',
    'BLSH', 'CRCL', 'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F', 'PA=F', 'GLD',
    'SLV', 'PALL', 'PPLT', 'NIKL', 'LIT', 'ILIT', 'REMX', 'VOLT', 'GRID', 'EQT',
    'TAC', 'BE', 'OKLO', 'SMR', 'NEE', 'URA', 'SRUUF', 'CCJ', 'KAMJY', 'UNL',
    'NVDA', 'SMH', 'SMHX', 'TSM', 'AVGO', 'QCOM', 'MU', 'AMD', 'TER', 'NOW',
    'AXON', 'SNOW', 'PLTR', 'GOOG', 'MSFT', 'META', 'AMZN', 'AAPL', 'TSLA',
    'NFLX', 'SPOT', 'SHOP', 'UBER', 'DASH', 'NET', 'DXCM', 'ETSY', 'SQ', 'MAGS',
    'MTUM', 'IVES', 'XLK', 'XLI', 'XLU', 'XLRE', 'XLB', 'XLV', 'XLF', 'XLE',
    'XLP', 'XLY', 'XLC', 'BLK', 'STT', 'ARES', 'SOFI', 'PYPL', 'IBKR', 'WU',
    'RXRX', 'SDGR', 'TEM', 'ABSI', 'DNA', 'TWST', 'GLW', 'KHC', 'LULU', 'YETI',
    'DLR', 'EQIX', 'ORCL', 'LSF'
]

# --- Database & State Management ---

def init_db():
    """Initialize SQLite database for scan state tracking"""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_log (
                ticker TEXT,
                scan_date TEXT,
                status TEXT,
                PRIMARY KEY (ticker, scan_date)
            )
        """)

def is_scanned_today(ticker):
    """Check if ticker was already processed today"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    with sqlite3.connect(DB_FILE) as conn:
        cur = conn.cursor()
        res = cur.execute(
            "SELECT 1 FROM scan_log WHERE ticker=? AND scan_date=? AND status='OK'", 
            (ticker, today)
        ).fetchone()
        return res is not None

def log_scan(ticker, status="OK"):
    """Mark ticker as scanned"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scan_log VALUES (?, ?, ?)",
            (ticker, today, status)
        )

# --- Analytic Logic ---

def analyze_ticker(ticker, macro_regime=None):
    """
    Core Analysis Engine
    Returns dict of signals or None if no data
    """
    try:
        df = safe_download(ticker)
        if df is None: return None

        # 1. Calculate Indicators
        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        
        # Squeeze
        sq_res = calc_squeeze(df)
        
        # Trend
        shannon = calc_shannon(df)
        macd, macd_sig, macd_hist = calc_macd(df)
        stoch_k, stoch_d = calc_stoch(df)
        adx, pdi, mdi = calc_adx(df)
        
        # 2. Extract Latest Values
        last = df.iloc[-1]
        price = last['Close']
        
        # 3. Trend Definition
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        trend = "BULLISH" if price > sma_200 else "BEARISH"
        
        # 4. Signal Synthesis
        signals = {
            'ticker': ticker,
            'price': price,
            'trend': trend,
            'score': 0,
            'setup': None,
            'squeeze': sq_res,
            'rsi': last['RSI'],
            'shannon': shannon,
            'adx': adx.iloc[-1]
        }
        
        # --- Scoring Logic (The "Brain") ---
        score = 0
        
        # A. DeMark Setup
        if last['Buy_Setup'] == 9:
            signals['setup'] = {'type': 'BUY', 'perfected': last['Perfected']}
            score += 3 if last['Perfected'] else 2
        elif last['Sell_Setup'] == 9:
            signals['setup'] = {'type': 'SELL', 'perfected': last['Perfected']}
            score += 3 if last['Perfected'] else 2
            
        # B. RSI Extremes
        if last['RSI'] < 30: score += 2  # Oversold
        if last['RSI'] > 70: score += 2  # Overbought
        
        # C. Squeeze
        if sq_res: score += 2
        
        # D. Momentum Breakout (AlphaTrends)
        if shannon['breakout']: score += 3
        
        # E. MACD Cross
        if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0: score += 1
        
        # F. Liquidity Filter (Macro Overlay)
        # If Risk Off, penalize Buy signals, boost Sell signals
        if macro_regime == 'RISK_OFF':
            if signals.get('setup') and signals['setup']['type'] == 'BUY':
                score -= 2 # Fade buys in bad macro
            if trend == 'BEARISH':
                score += 1 # Trend alignment
                
        signals['score'] = score
        
        # 5. Target/Stop Calculation (ATR Based)
        tr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        signals['stop'] = price - (tr * 2) # Wide stop
        signals['target'] = price + (tr * 3)
        
        return signals

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None

def format_alert(s):
    """Format single alert for Telegram"""
    icon = "🟢" if s['trend'] == "BULLISH" else "🔴"
    msg = f"{icon} *{s['ticker']}* @ {fmt_price(s['price'])}\n"
    msg += f"Score: {s['score']}/10\n"
    
    if s['setup']:
        p_mark = "⭐" if s['setup']['perfected'] else "○"
        msg += f"• DeMark: {s['setup']['type']} 9 {p_mark}\n"
        
    if s['squeeze']:
        msg += f"• Squeeze: {s['squeeze']['bias']} Ready\n"
        
    if s['shannon']['breakout']:
        msg += f"• Momentum: BREAKOUT 🚀\n"
        
    msg += f"• RSI: {s['rsi']:.1f}\n"
    return msg + "\n"

# --- Main Execution Flow ---

if __name__ == "__main__":
    print("="*60)
    print("INSTITUTIONAL ENGINE STARTUP")
    print("="*60)
    
    # 1. Initialize State
    init_db()
    
    # 2. Get Macro Regime
    print("[1/4] Analyzing Macro...")
    macro = get_macro()
    regime = "NEUTRAL"
    if macro and 'net_liq' in macro:
        # Simple 3-month ROC of Net Liq
        liq_chg = macro['net_liq'].pct_change(63).iloc[-1]
        regime = "RISK_ON" if liq_chg > 0 else "RISK_OFF"
        
    msg = f"📊 *MARKET REGIME: {regime}*\n"
    if macro:
        msg += f"SPY: {fmt_price(macro['spy'].iloc[-1])}\n"
        if 'net_liq' in macro:
            msg += f"Liquidity Trend: {'⬆️' if regime=='RISK_ON' else '⬇️'}\n"
    send_telegram(msg)
    
    # 3. Scan Portfolio (Always Scan These)
    print("\n[2/4] Scanning Portfolio...")
    port_msg = "💼 *PORTFOLIO UPDATE*\n\n"
    for t in CURRENT_PORTFOLIO:
        res = analyze_ticker(t, regime)
        if res:
            port_msg += format_alert(res)
    send_telegram(port_msg)
    
    # 4. Batch Scan Universe
    print("\n[3/4] Scanning Universe...")
    universe = list(set(STRATEGIC_TICKERS + get_futures()))
    to_scan = [t for t in universe if not is_scanned_today(t)]
    
    print(f"   Total Universe: {len(universe)}")
    print(f"   Remaining to scan: {len(to_scan)}")
    
    high_conviction = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map futures to tickers
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in to_scan}
        
        completed = 0
        for future in as_completed(future_map):
            ticker = future_map[future]
            try:
                res = future.result()
                log_scan(ticker, "OK" if res else "FAIL")
                
                if res and res['score'] >= 4:
                    print(f"   ⭐ FOUND: {ticker} (Score: {res['score']})")
                    high_conviction.append(res)
                
                completed += 1
                if completed % 10 == 0:
                    print(f"   Progress: {completed}/{len(to_scan)}")
                    
            except Exception as e:
                print(f"   Fail {ticker}: {e}")
                log_scan(ticker, "ERROR")

    # 5. Alerting
    print("\n[4/4] Sending Alerts...")
    if high_conviction:
        # Sort by score highest first
        high_conviction.sort(key=lambda x: x['score'], reverse=True)
        
        alert_msg = "🚨 *HIGH CONVICTION SETUP*\n\n"
        for res in high_conviction[:10]: # Top 10 only
            alert_msg += format_alert(res)
            
        send_telegram(alert_msg)
    else:
        print("   No high conviction setups found today.")
        
    print("\n✓ ENGINE SHUTDOWN")
