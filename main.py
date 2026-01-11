# main.py - Entry point for the trading scanner
import os
from data_fetcher import safe_download, get_macro, get_futures
from indicators import (calc_rsi, calc_squeeze, calc_fib, calc_demark, 
                       calc_shannon, calc_macd, calc_stoch, calc_adx)
from utils import send_telegram, fmt_price
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

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

def analyze_ticker(ticker, is_portfolio=False):
    try:
        df = safe_download(ticker)
        if df is None or len(df) < 50:
            return None
        if not is_portfolio and (df['Volume'].iloc[-5:].sum() == 0 or df['Close'].iloc[-1] < 0.00000001):
            return None
        
        df_w = df.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        if len(df_w) < 20:
            return None
        
        df['RSI'] = calc_rsi(df['Close'])
        df = calc_demark(df)
        df_w = calc_demark(df_w)
        d_sq = calc_squeeze(df)
        w_sq = calc_squeeze(df_w)
        fib = calc_fib(df)
        shannon = calc_shannon(df)
        macd_line, sig_line, macd_hist = calc_macd(df)
        sto_k, sto_d = calc_stoch(df)
        adx, plus_di, minus_di = calc_adx(df)
        
        last = df.iloc[-1]
        last_w = df_w.iloc[-1]
        price = last['Close']
        
        tr = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        target = price + (atr * 2)
        stop = price - (atr * 1.5)
        
        dm = None
        dm_perf = False
        if last['Buy_Countdown'] == 13:
            dm = {'type': 'BUY 13', 'tf': 'Daily'}
        elif last['Sell_Countdown'] == 13:
            dm = {'type': 'SELL 13', 'tf': 'Daily'}
        elif last['Buy_Setup'] == 9:
            dm = {'type': 'BUY 9', 'tf': 'Daily'}
        elif last['Sell_Setup'] == 9:
            dm = {'type': 'SELL 9', 'tf': 'Daily'}
        elif last_w['Buy_Countdown'] == 13:
            dm = {'type': 'BUY 13', 'tf': 'Weekly'}
        elif last_w['Sell_Countdown'] == 13:
            dm = {'type': 'SELL 13', 'tf': 'Weekly'}
        
        if dm:
            if '13' in dm['type']:
                dm_perf = True
            elif 'BUY' in dm['type'] and len(df) >= 4:
                dm_perf = (df['Low'].iloc[-1] < df['Low'].iloc[-3] and df['Low'].iloc[-1] < df['Low'].iloc[-4])
            elif 'SELL' in dm['type'] and len(df) >= 4:
                dm_perf = (df['High'].iloc[-1] > df['High'].iloc[-3] and df['High'].iloc[-1] > df['High'].iloc[-4])
            dm['perfected'] = dm_perf
        
        rsi_sig = None
        if last['RSI'] < 30:
            rsi_sig = {'type': 'OVERSOLD', 'val': last['RSI']}
        elif last['RSI'] > 70:
            rsi_sig = {'type': 'OVERBOUGHT', 'val': last['RSI']}
        
        sq = None
        if d_sq:
            sq = {'tf': 'Daily', 'bias': d_sq['bias'], 'move': d_sq['move']}
        elif w_sq:
            sq = {'tf': 'Weekly', 'bias': w_sq['bias'], 'move': w_sq['move']}
        
        macd_sig = None
        if len(macd_hist) >= 2:
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
                macd_sig = {'type': 'BULLISH CROSS'}
            elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
                macd_sig = {'type': 'BEARISH CROSS'}
        
        sto_sig = None
        if len(sto_k) >= 2:
            if sto_k.iloc[-1] > sto_d.iloc[-1] and sto_k.iloc[-2] <= sto_d.iloc[-2] and sto_k.iloc[-1] < 20:
                sto_sig = {'type': 'BULLISH CROSS OVERSOLD'}
            elif sto_k.iloc[-1] < sto_d.iloc[-1] and sto_k.iloc[-2] >= sto_d.iloc[-2] and sto_k.iloc[-1] > 80:
                sto_sig = {'type': 'BEARISH CROSS OVERBOUGHT'}
        
        adx_sig = None
        if adx.iloc[-1] > 25:
            dir_ = 'UPTREND' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'DOWNTREND'
            adx_sig = {'strength': 'STRONG', 'direction': dir_}
        
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        trend = "BULLISH" if price > sma200 else "BEARISH"
        
        verdict = "HOLD"
        if dm and "BUY" in dm['type']:
            verdict = "BUY (Signal)"
        elif dm and "SELL" in dm['type']:
            verdict = "SELL (Signal)"
        elif trend == "BULLISH" and shannon['near_term'] == "BULLISH":
            verdict = "BUY (Trend)"
        elif trend == "BEARISH" and shannon['near_term'] == "BEARISH":
            verdict = "SELL (Trend)"
        
        count = f"Buy {int(last['Buy_Setup'])}" if last['Buy_Setup'] > 0 else f"Sell {int(last['Sell_Setup'])}"
        
        return {'ticker': ticker, 'price': price, 'demark': dm, 'rsi': rsi_sig, 'squeeze': sq, 'fib': fib, 'shannon': shannon, 'macd': macd_sig, 'stochastic': sto_sig, 'adx': adx_sig, 'trend': trend, 'verdict': verdict, 'target': target, 'stop': stop, 'rsi_val': last['RSI'], 'count': count}
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return None

if __name__ == "__main__":
    print("="*60)
    print("INSTITUTIONAL TRADING ALERT")
    print("="*60)
    print("\n[PRE-FLIGHT]")
    if not os.environ.get('TELEGRAM_TOKEN') or not os.environ.get('TELEGRAM_CHAT_ID'):
        print("⚠️ Telegram missing")
    else:
        print("✓ Telegram OK")
    if not os.environ.get('FRED_API_KEY'):
        print("⚠️ FRED missing")
    else:
        print("✓ FRED OK")
    
    print("\n[1/3] MACRO")
    macro = get_macro()
    msg = "📊 MACRO\n\n"
    if macro:
        liq = macro['net_liq'].pct_change(63).iloc[-1]
        regime = "🟢 RISK ON" if liq > 0 else "🔴 RISK OFF"
        msg += f"{regime}\nLiq: {liq*100:.2f}%\n\n"
    else:
        msg += "Unavailable\n\n"
    send_telegram(msg)
    
    print("\n[2/3] PORTFOLIO")
    p_msg = "💼 PORTFOLIO\n\n"
    for t in CURRENT_PORTFOLIO:
        res = analyze_ticker(t, is_portfolio=True)
        if res:
            p_msg += f"*{t}*: {res['verdict']} @ {fmt_price(res['price'])}\n"
            p_msg += f"Target: {fmt_price(res['target'])} | Stop: {fmt_price(res['stop'])}\n"
            p_msg += f"Trend: {res['trend']} | RSI: {res['rsi_val']:.0f}\n"
            p_msg += f"DeMark: {res['count']}\n"
            if res['demark']:
                p_msg += f"🎯 {res['demark']['type']} ({'Perf' if res['demark'].get('perfected') else 'Unp'})\n"
            if res['squeeze']:
                p_msg += f"💥 {res['squeeze']['tf']} {res['squeeze']['bias']}\n"
            p_msg += "\n"
    send_telegram(p_msg)
    
    print("\n[3/3] SCANNING")
    universe = list(set(STRATEGIC_TICKERS + get_futures()))
    power = []
    perfected = []
    sq_list = []
    shannon_list = []
    
    def process(t):
        return analyze_ticker(t)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process, t): t for t in universe}
        for future in as_completed(futures):
            res = future.result()
            if not res:
                continue
            score = 0
            d = res['demark']
            if d and d.get('perfected'):
                score += 2
            if res['rsi']:
                score += 1
            if res['squeeze']:
                score += 1
            if res['shannon']['breakout']:
                score += 1
            if score >= 3:
                power.append(res)
            if d and d.get('perfected'):
                perfected.append(res)
            if res['squeeze']:
                sq_list.append(res)
            if res['shannon']['breakout']:
                shannon_list.append(res)
    
    a_msg = "🚨 ALERTS\n\n"
    if power:
        a_msg += "⭐ POWER\n"
        for s in sorted(power, key=lambda x: ('13' in x['demark']['type'] if x['demark'] else False), reverse=True)[:10]:
            a_msg += f"*{s['ticker']}*: {fmt_price(s['price'])}\n"
            a_msg += f"Target: {fmt_price(s['target'])}\n\n"
    if perfected:
        a_msg += "🎯 PERFECTED\n"
        for s in perfected[:10]:
            if s not in power:
                a_msg += f"{s['ticker']}: {s['demark']['type']}\n"
    if shannon_list:
        a_msg += "\n📈 BREAKOUTS\n"
        for s in shannon_list[:10]:
            if s not in power:
                a_msg += f"{s['ticker']}: Momentum\n"
    if sq_list:
        a_msg += "\n💥 SQUEEZES\n"
        for s in sq_list[:10]:
            if s not in power:
                a_msg += f"{s['ticker']}: {s['squeeze']['tf']}\n"
    
    send_telegram(a_msg)
    print("\n✓ Complete")
