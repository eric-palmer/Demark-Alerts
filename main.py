# main.py - Institutional Engine (Crash Safe)
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
MAX_WORKERS = 6 

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
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_log (
                ticker TEXT, scan_date TEXT, status TEXT,
                PRIMARY KEY (ticker, scan_date)
            )
        """)

def is_scanned_today(ticker):
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    with sqlite3.connect(DB_FILE) as conn:
        res = conn.execute(
            "SELECT 1 FROM scan_log WHERE ticker=? AND scan_date=? AND status='OK'", 
            (ticker, today)
        ).fetchone()
        return res is not None

def log_scan(ticker, status="OK"):
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scan_log VALUES (?, ?, ?)",
            (ticker, today, status)
        )

# --- Analytic Logic ---

def analyze_ticker(ticker, macro_regime=None):
    try:
        df = safe_download(ticker)
        if df is None: return None

        # 1. Calculate Indicators
        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        sq_res = calc_squeeze(df)
        shannon = calc_shannon(df)
        macd, macd_sig, macd_hist = calc_macd(df)
        adx, pdi, mdi = calc_adx(df)
        
        last = df.iloc[-1]
        price = last['Close']
        
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        trend = "BULLISH" if price > sma_200 else "BEARISH"
        
        signals = {
            'ticker': ticker,
            'price': price,
            'trend': trend,
            'score': 0,
            'setup': None,
            'demark_count': 0,
            'squeeze': sq_res,
            'rsi': last['RSI'],
            'shannon': shannon,
            'adx': adx.iloc[-1]
        }
        
        # --- Scoring ---
        score = 0
        
        # DeMark
        buy_seq = last.get('Buy_Setup', 0)
        sell_seq = last.get('Sell_Setup', 0)
        
        if buy_seq >= 1:
            signals['demark_count'] = f"Buy {int(buy_seq)}"
            if buy_seq == 9:
                signals['setup'] = {'type': 'BUY', 'perfected': last.get('Perfected', False)}
                score += 3 if last.get('Perfected', False) else 2
        elif sell_seq >= 1:
            signals['demark_count'] = f"Sell {int(sell_seq)}"
            if sell_seq == 9:
                signals['setup'] = {'type': 'SELL', 'perfected': last.get('Perfected', False)}
                score += 3 if last.get('Perfected', False) else 2
            
        # Indicators
        if last['RSI'] < 30: score += 2
        if last['RSI'] > 70: score += 2
        if sq_res: score += 2
        if shannon['breakout']: score += 3
        if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0: score += 1
        
        # Liquidity Filter
        if macro_regime == 'RISK_OFF':
            if signals.get('setup') and signals['setup']['type'] == 'BUY': score -= 2
            if trend == 'BEARISH': score += 1
                
        signals['score'] = score
        return signals

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None

def format_alert(s, is_portfolio=False):
    icon = "🟢" if s['trend'] == "BULLISH" else "🔴"
    msg = f"{icon} *{s['ticker']}* @ {fmt_price(s['price'])}\n"
    msg += f"Score: {s['score']}/10\n"
    
    if s['setup']:
        p_mark = "⭐" if s['setup']['perfected'] else "○"
        msg += f"• DeMark: {s['setup']['type']} 9 {p_mark}\n"
    elif is_portfolio and s['demark_count']:
        msg += f"• DeMark: {s['demark_count']}\n"
            
    if s['squeeze']:
        msg += f"• Squeeze: {s['squeeze']['bias']} Ready\n"
            
    if s['shannon']['breakout']:
        msg += f"• Momentum: BREAKOUT 🚀\n"
    
    if is_portfolio:
        msg += f"• RSI: {s['rsi']:.1f}\n"
        msg += f"• ADX: {s['adx']:.1f}\n"
    else:
        if s['rsi'] < 30 or s['rsi'] > 70:
            msg += f"• RSI: {s['rsi']:.1f}\n"

    return msg + "\n"

# --- Main Execution ---

if __name__ == "__main__":
    print("="*60)
    print("INSTITUTIONAL ENGINE STARTUP")
    print("="*60)
    
    init_db()
    
    print("[1/4] Analyzing Macro...")
    macro = get_macro()
    regime = "NEUTRAL"
    liq_msg = "UNKNOWN"
    
    # CRASH PROTECTION: Check if net_liq exists before math
    if macro and macro.get('net_liq') is not None:
        try:
            liq_chg = macro['net_liq'].pct_change(63).iloc[-1]
            regime = "RISK_ON" if liq_chg > 0 else "RISK_OFF"
            liq_msg = "⬆️" if regime == "RISK_ON" else "⬇️"
        except Exception as e:
            print(f"Liquidity math error: {e}")
            
    msg = f"📊 *MARKET REGIME: {regime}*\n"
    if macro and macro.get('spy') is not None:
        msg += f"SPY: {fmt_price(macro['spy'].iloc[-1])}\n"
    msg += f"Liquidity Trend: {liq_msg}\n"
    send_telegram(msg)
    
    print("\n[2/4] Scanning Portfolio...")
    port_msg = "💼 *PORTFOLIO UPDATE*\n\n"
    for t in CURRENT_PORTFOLIO:
        res = analyze_ticker(t, regime)
        if res:
            port_msg += format_alert(res, is_portfolio=True)
    send_telegram(port_msg)
    
    print("\n[3/4] Scanning Universe...")
    universe = list(set(STRATEGIC_TICKERS + get_futures()))
    to_scan = [t for t in universe if not is_scanned_today(t)]
    
    print(f"   Remaining to scan: {len(to_scan)}")
    
    high_conviction = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(analyze_ticker, t, regime): t for t in to_scan}
        completed = 0
        for future in as_completed(future_map):
            ticker = future_map[future]
            try:
                res = future.result()
                log_scan(ticker, "OK" if res else "FAIL")
                if res and res['score'] >= 4:
                    print(f"   ⭐ {ticker} ({res['score']})")
                    high_conviction.append(res)
                completed += 1
            except:
                pass

    print("\n[4/4] Sending Alerts...")
    if high_conviction:
        high_conviction.sort(key=lambda x: x['score'], reverse=True)
        alert_msg = "🚨 *HIGH CONVICTION SETUP*\n\n"
        for res in high_conviction[:10]:
            alert_msg += format_alert(res, is_portfolio=False)
        send_telegram(alert_msg)
        
    print("\n✓ ENGINE COMPLETE")
