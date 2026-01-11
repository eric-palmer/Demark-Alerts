# main.py - Enhanced Institutional Trading Scanner
import os
from data_fetcher import safe_download, get_macro, get_futures
from indicators import (calc_rsi, calc_squeeze, calc_fib, calc_demark, 
                       calc_shannon, calc_macd, calc_stoch, calc_adx)
from utils import send_telegram, fmt_price
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

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

def calculate_timing(df, timeframe='Daily'):
    """Calculate expected timing for signal based on historical volatility"""
    try:
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        avg_daily_move = df['Close'].pct_change().abs().rolling(20).mean().iloc[-1]
        
        if timeframe == 'Weekly':
            min_days = 5
            max_days = 30
        else:
            min_days = 1
            max_days = 15
        
        if avg_daily_move > 0.03:
            return f"{min_days}-{int(max_days/2)} days"
        elif avg_daily_move > 0.015:
            return f"{int(min_days*2)}-{max_days} days"
        else:
            return f"{int(max_days/2)}-{max_days*2} days"
    except:
        return "1-4 weeks" if timeframe == 'Weekly' else "3-10 days"

def calculate_risk_reward(entry, target, stop):
    """Calculate risk/reward ratio"""
    try:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk > 0:
            rr = reward / risk
            return f"{rr:.2f}:1"
        return "N/A"
    except:
        return "N/A"

def calculate_position_size_suggestion(price, stop, account_risk_pct=2):
    """Suggest position size based on 2% account risk"""
    try:
        risk_per_share = abs(price - stop)
        risk_budget = 10000 * (account_risk_pct / 100)
        shares = int(risk_budget / risk_per_share)
        return shares if shares > 0 else 1
    except:
        return "N/A"

def analyze_ticker(ticker, is_portfolio=False):
    """Comprehensive ticker analysis with institutional-grade metrics"""
    try:
        df = safe_download(ticker)
        if df is None or len(df) < 50:
            return None
        
        if not is_portfolio and (df['Volume'].iloc[-5:].sum() == 0 or df['Close'].iloc[-1] < 0.00000001):
            return None
        
        df_w = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
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
        
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(14).mean().iloc[-1]
        
        recent_high = df['High'].iloc[-20:].max()
        recent_low = df['Low'].iloc[-20:].min()
        
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-5:].mean()
        volume_surge = current_volume > avg_volume * 1.5
        
        price_vs_20sma = ((price - sma_20) / sma_20) * 100
        price_vs_50sma = ((price - sma_50) / sma_50) * 100
        
        dm = None
        dm_perf = False
        dm_timeframe = None
        
        if last['Buy_Countdown'] == 13:
            dm = {'type': 'BUY', 'signal': '13', 'tf': 'Daily'}
            dm_timeframe = 'Daily'
        elif last['Sell_Countdown'] == 13:
            dm = {'type': 'SELL', 'signal': '13', 'tf': 'Daily'}
            dm_timeframe = 'Daily'
        elif last['Buy_Setup'] == 9:
            dm = {'type': 'BUY', 'signal': '9', 'tf': 'Daily'}
            dm_timeframe = 'Daily'
        elif last['Sell_Setup'] == 9:
            dm = {'type': 'SELL', 'signal': '9', 'tf': 'Daily'}
            dm_timeframe = 'Daily'
        elif last_w['Buy_Countdown'] == 13:
            dm = {'type': 'BUY', 'signal': '13', 'tf': 'Weekly'}
            dm_timeframe = 'Weekly'
        elif last_w['Sell_Countdown'] == 13:
            dm = {'type': 'SELL', 'signal': '13', 'tf': 'Weekly'}
            dm_timeframe = 'Weekly'
        
        if dm:
            if dm['signal'] == '13':
                dm_perf = True
            elif dm['type'] == 'BUY' and len(df) >= 4:
                dm_perf = (df['Low'].iloc[-1] < df['Low'].iloc[-3] and 
                          df['Low'].iloc[-1] < df['Low'].iloc[-4])
            elif dm['type'] == 'SELL' and len(df) >= 4:
                dm_perf = (df['High'].iloc[-1] > df['High'].iloc[-3] and 
                          df['High'].iloc[-1] > df['High'].iloc[-4])
            dm['perfected'] = dm_perf
        
        rsi_sig = None
        rsi_val = last['RSI']
        if rsi_val < 30:
            rsi_sig = {'type': 'OVERSOLD', 'val': rsi_val, 'strength': 'STRONG' if rsi_val < 20 else 'MODERATE'}
        elif rsi_val > 70:
            rsi_sig = {'type': 'OVERBOUGHT', 'val': rsi_val, 'strength': 'STRONG' if rsi_val > 80 else 'MODERATE'}
        
        rsi_divergence = None
        if len(df) >= 20:
            if df['Close'].iloc[-1] < df['Close'].iloc[-10] and df['RSI'].iloc[-1] > df['RSI'].iloc[-10]:
                rsi_divergence = 'BULLISH'
            elif df['Close'].iloc[-1] > df['Close'].iloc[-10] and df['RSI'].iloc[-1] < df['RSI'].iloc[-10]:
                rsi_divergence = 'BEARISH'
        
        sq = None
        if d_sq:
            sq = {'tf': 'Daily', 'bias': d_sq['bias'], 'move': d_sq['move']}
        elif w_sq:
            sq = {'tf': 'Weekly', 'bias': w_sq['bias'], 'move': w_sq['move']}
        
        macd_sig = None
        macd_strength = None
        if len(macd_hist) >= 2:
            if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0:
                macd_sig = {'type': 'BULLISH CROSS'}
                macd_strength = 'STRONG' if abs(macd_hist.iloc[-1]) > abs(macd_hist.iloc[-10:].mean()) else 'WEAK'
            elif macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0:
                macd_sig = {'type': 'BEARISH CROSS'}
                macd_strength = 'STRONG' if abs(macd_hist.iloc[-1]) > abs(macd_hist.iloc[-10:].mean()) else 'WEAK'
            
            if macd_sig:
                macd_sig['strength'] = macd_strength
        
        sto_sig = None
        if len(sto_k) >= 2:
            if sto_k.iloc[-1] > sto_d.iloc[-1] and sto_k.iloc[-2] <= sto_d.iloc[-2] and sto_k.iloc[-1] < 20:
                sto_sig = {'type': 'BULLISH CROSS OVERSOLD', 'k': sto_k.iloc[-1], 'd': sto_d.iloc[-1]}
            elif sto_k.iloc[-1] < sto_d.iloc[-1] and sto_k.iloc[-2] >= sto_d.iloc[-2] and sto_k.iloc[-1] > 80:
                sto_sig = {'type': 'BEARISH CROSS OVERBOUGHT', 'k': sto_k.iloc[-1], 'd': sto_d.iloc[-1]}
        
        adx_sig = None
        trend_strength = None
        if adx.iloc[-1] > 25:
            direction = 'UPTREND' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'DOWNTREND'
            if adx.iloc[-1] > 50:
                trend_strength = 'VERY STRONG'
            elif adx.iloc[-1] > 40:
                trend_strength = 'STRONG'
            else:
                trend_strength = 'MODERATE'
            adx_sig = {'strength': trend_strength, 'direction': direction, 'adx': adx.iloc[-1]}
        
        trend = "BULLISH" if price > sma_200 else "BEARISH"
        trend_quality = "STRONG" if abs(price - sma_200) / sma_200 > 0.05 else "WEAK"
        
        if dm and dm['type'] == 'BUY':
            target_1 = price + (atr_14 * 1.5)
            target_2 = price + (atr_14 * 2.5)
            target_3 = min(recent_high, price + (atr_14 * 4))
            stop = max(recent_low, price - (atr_14 * 1.5))
            
            if abs(price - sma_20) / price < 0.02:
                stop = min(stop, sma_20 * 0.97)
            
        elif dm and dm['type'] == 'SELL':
            target_1 = price - (atr_14 * 1.5)
            target_2 = price - (atr_14 * 2.5)
            target_3 = max(recent_low, price - (atr_14 * 4))
            stop = min(recent_high, price + (atr_14 * 1.5))
            
            if abs(price - sma_20) / price < 0.02:
                stop = max(stop, sma_20 * 1.03)
        else:
            target_1 = price + (atr_14 * 2)
            target_2 = price + (atr_14 * 3)
            target_3 = recent_high
            stop = price - (atr_14 * 1.5)
        
        signal_score = 0
        bullish_count = 0
        bearish_count = 0
        
        if dm:
            if dm['type'] == 'BUY':
                signal_score += 3 if dm['perfected'] else 2
                bullish_count += 1
            else:
                signal_score += 3 if dm['perfected'] else 2
                bearish_count += 1
        
        if rsi_sig:
            signal_score += 1
            if rsi_sig['type'] == 'OVERSOLD':
                bullish_count += 1
            else:
                bearish_count += 1
        
        if sq:
            signal_score += 1
            if sq['bias'] == 'BULLISH':
                bullish_count += 1
            else:
                bearish_count += 1
        
        if shannon['breakout']:
            signal_score += 2
            bullish_count += 1
        
        if macd_sig:
            signal_score += 1
            if macd_sig['type'] == 'BULLISH CROSS':
                bullish_count += 1
            else:
                bearish_count += 1
        
        if sto_sig:
            signal_score += 1
            if 'BULLISH' in sto_sig['type']:
                bullish_count += 1
            else:
                bearish_count += 1
        
        if adx_sig:
            signal_score += 1
        
        verdict = "HOLD"
        confidence = "LOW"
        
        if bullish_count >= 3:
            verdict = "BUY"
            confidence = "HIGH" if bullish_count >= 4 else "MODERATE"
        elif bearish_count >= 3:
            verdict = "SELL"
            confidence = "HIGH" if bearish_count >= 4 else "MODERATE"
        elif dm and dm['perfected']:
            verdict = f"{dm['type']} (DeMark)"
            confidence = "MODERATE"
        elif trend == "BULLISH" and shannon['near_term'] == "BULLISH":
            verdict = "BUY (Trend)"
            confidence = "LOW"
        elif trend == "BEARISH" and shannon['near_term'] == "BEARISH":
            verdict = "SELL (Trend)"
            confidence = "LOW"
        
        timing = calculate_timing(df, dm_timeframe if dm else 'Daily')
        rr_ratio = calculate_risk_reward(price, target_2, stop)
        pos_size = calculate_position_size_suggestion(price, stop)
        
        count = f"Buy {int(last['Buy_Setup'])}" if last['Buy_Setup'] > 0 else f"Sell {int(last['Sell_Setup'])}"
        
        return {
            'ticker': ticker,
            'price': price,
            'target_1': target_1,
            'target_2': target_2,
            'target_3': target_3,
            'stop': stop,
            'timing': timing,
            'rr_ratio': rr_ratio,
            'pos_size': pos_size,
            'verdict': verdict,
            'confidence': confidence,
            'signal_score': signal_score,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'demark': dm,
            'rsi': rsi_sig,
            'rsi_val': rsi_val,
            'rsi_divergence': rsi_divergence,
            'squeeze': sq,
            'fib': fib,
            'shannon': shannon,
            'macd': macd_sig,
            'stochastic': sto_sig,
            'adx': adx_sig,
            'trend': trend,
            'trend_quality': trend_quality,
            'volume_surge': volume_surge,
            'price_vs_20sma': price_vs_20sma,
            'price_vs_50sma': price_vs_50sma,
            'count': count,
            'sma_20': sma_20,
            'sma_50': sma_50
        }
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return None

def format_signal_block(res, show_all=False):
    """Format a standardized signal block"""
    msg = f"\n*{res['ticker']}* @ {fmt_price(res['price'])}\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"*Action:* {res['verdict']} ({res['confidence']} confidence)\n"
    msg += f"*Targets:* {fmt_price(res['target_1'])} / {fmt_price(res['target_2'])} / {fmt_price(res['target_3'])}\n"
    msg += f"*Stop Loss:* {fmt_price(res['stop'])}\n"
    msg += f"*Risk/Reward:* {res['rr_ratio']}\n"
    msg += f"*Time Horizon:* {res['timing']}\n"
    msg += f"*Position Size:* ~{res['pos_size']} shares (2% risk)\n\n"
    
    msg += f"*Technical Summary:*\n"
    msg += f"• Trend: {res['trend']} ({res['trend_quality']})\n"
    msg += f"• RSI: {res['rsi_val']:.1f}"
    if res['rsi_divergence']:
        msg += f" ({res['rsi_divergence']} DIV)"
    msg += "\n"
    msg += f"• Price vs 20 SMA: {res['price_vs_20sma']:.1f}%\n"
    msg += f"• Signal Confluence: {res['bullish_signals']} Bull / {res['bearish_signals']} Bear\n"
    
    if show_all:
        msg += f"\n*Active Signals:*\n"
        if res['demark']:
            perf = "✓ Perfected" if res['demark']['perfected'] else "○ Imperfected"
            msg += f"• DeMark: {res['demark']['type']} {res['demark']['signal']} ({res['demark']['tf']}) {perf}\n"
        if res['squeeze']:
            msg += f"• Squeeze: {res['squeeze']['tf']} {res['squeeze']['bias']} (Move: {fmt_price(res['squeeze']['move'])})\n"
        if res['macd']:
            msg += f"• MACD: {res['macd']['type']} ({res['macd'].get('strength', 'N/A')})\n"
        if res['stochastic']:
            msg += f"• Stochastic: {res['stochastic']['type']}\n"
        if res['adx']:
            msg += f"• ADX: {res['adx']['strength']} {res['adx']['direction']} ({res['adx']['adx']:.1f})\n"
        if res['fib']:
            msg += f"• Fibonacci: {res['fib']['action']} at {res['fib']['level']}\n"
        if res['shannon']['breakout']:
            msg += f"• AlphaTrends: Momentum Breakout\n"
    
    return msg

if __name__ == "__main__":
    print("="*60)
    print("INSTITUTIONAL TRADING ALERT SYSTEM")
    print("="*60)
    
    print("\n[PRE-FLIGHT CHECKS]")
    telegram_ok = bool(os.environ.get('TELEGRAM_TOKEN') and os.environ.get('TELEGRAM_CHAT_ID'))
    fred_ok = bool(os.environ.get('FRED_API_KEY'))
    
    if not telegram_ok:
        print("⚠️  Telegram missing")
    else:
        print("✓ Telegram OK")
    
    if not fred_ok:
        print("⚠️  FRED missing")
    else:
        print("✓ FRED OK")
    
    print("\n[1/4] MACRO ANALYSIS")
    macro = None
    if fred_ok:
        macro = get_macro()
    
    msg = "📊 *GLOBAL MACRO SNAPSHOT*\n\n"
    
    if macro and 'net_liq' in macro:
        try:
            liq = macro['net_liq'].pct_change(63).iloc[-1]
            liq_6m = macro['net_liq'].pct_change(126).iloc[-1]
            regime = "🟢 RISK ON" if liq > 0 else "🔴 RISK OFF"
            
            msg += f"*Market Regime:* {regime}\n"
            msg += f"• Net Liquidity (3M): {liq*100:+.2f}%\n"
            msg += f"• Net Liquidity (6M): {liq_6m*100:+.2f}%\n"
            
            if 'term_premia' in macro:
                term_spread = macro['term_premia'].iloc[-1]
                yield_curve = "NORMAL" if term_spread > 0 else "INVERTED"
                msg += f"• Yield Curve: {yield_curve} ({term_spread:.2f}%)\n"
            
            if 'inflation' in macro:
                inflation = macro['inflation'].iloc[-1]
                msg += f"• 5Y Inflation Exp: {inflation:.2f}%\n"
            
            msg += f"\n💡 *Interpretation:* "
            if liq > 0.05:
                msg += "Strong liquidity expansion favors risk assets\n"
            elif liq < -0.05:
                msg += "Liquidity contraction - defensive posture\n"
            else:
                msg += "Neutral liquidity environment\n"
        except Exception as e:
            print(f"Macro calc error: {e}")
            msg += "Macro data partially available\n"
    else:
        msg += "Temporarily unavailable\n"
    
    send_telegram(msg)
    
    print("\n[2/4] PORTFOLIO ANALYSIS")
    p_msg = "💼 *PORTFOLIO DEEP DIVE*\n\n"
    portfolio_analyzed = 0
    
    for t in CURRENT_PORTFOLIO:
        print(f"  Analyzing {t}...")
        try:
            res = analyze_ticker(t, is_portfolio=True)
            if res:
                p_msg += format_signal_block(res, show_all=True)
                portfolio_analyzed += 1
                print(f"  ✓ {t} analyzed")
            else:
                print(f"  ✗ {t} failed - no data")
                p_msg += f"*{t}*: Data unavailable\n\n"
        except Exception as e:
            print(f"  ✗ {t} error: {e}")
            p_msg += f"*{t}*: Analysis error\n\n"
    
    print(f"  Portfolio: {portfolio_analyzed}/{len(CURRENT_PORTFOLIO)} analyzed")
    send_telegram(p_msg)
    
    print("\n[3/4] SCANNING MARKET")
    universe = list(set(STRATEGIC_TICKERS + get_futures()))
    print(f"  Universe: {len(universe)} tickers")
    
    power = []
    perfected_demark = []
    imperfected_demark = []
    squeeze_setups = []
    momentum_breakouts = []
    oversold_rsi = []
    overbought_rsi = []
    
    successful_scans = 0
    failed_scans = 0
    data_failures = []
    
    def process(t):
        return analyze_ticker(t)
    
    print("  Starting parallel scan...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process, t): t for t in universe}
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                res = future.result()
                
                if not res:
                    failed_scans += 1
                    data_failures.append(ticker)
                    continue
                
                successful_scans += 1
                
                if res['signal_score'] >= 4:
                    power.append(res)
                    print(f"  🌟 POWER: {ticker}")
                
                if res['demark']:
                    if res['demark']['perfected']:
                        perfected_demark.append(res)
                        print(f"  🎯 PERF DM: {ticker}")
                    else:
                        imperfected_demark.append(res)
                
                if res['squeeze']:
                    squeeze_setups.append(res)
                
                if res['shannon']['breakout']:
                    momentum_breakouts.append(res)
                
                if res['rsi'] and res['rsi']['type'] == 'OVERSOLD':
                    oversold_rsi.append(res)
                
                if res['rsi'] and res['rsi']['type'] == 'OVERBOUGHT':
                    overbought_rsi.append(res)
                
                total = successful_scans + failed_scans
                if total % 25 == 0:
                    print(f"  Progress: {total}/{len(universe)} ({successful_scans} OK, {failed_scans} fail)")
                    
            except Exception as e:
                failed_scans += 1
                print(f"  ✗ Error {ticker}: {e}")
    
    print(f"\n  Complete!")
    print(f"  Success: {successful_scans}/{len(universe)}")
    print(f"  Failed: {failed_scans}")
    print(f"  Power: {len(power)}")
    print(f"  Perfected DM: {len(perfected_demark)}")
    print(f"  Imperfected DM: {len(imperfected_demark)}")
    print(f"  Squeeze: {len(squeeze_setups)}")
    print(f"  Breakouts: {len(momentum_breakouts)}")
    
    if failed_scans > len(universe) * 0.5:
        print(f"\n  ⚠️  High failure rate!")
        print(f"  Sample: {', '.join(data_failures[:10])}")
    
    print("\n[4/4] GENERATING ALERTS")
    
    if power:
        print(f"  Sending {len(power)} power setups...")
        a_msg = "⭐ *POWER RANKINGS*\n"
        a_msg += "Multiple indicators in agreement\n\n"
        
        power_sorted = sorted(power, key=lambda x: x['signal_score'], reverse=True)
        
        for res in power_sorted[:8]:
            a_msg += format_signal_block(res, show_all=False)
        
        send_telegram(a_msg)
    else:
        print("  No power setups")
    
    if perfected_demark:
        print(f"  Sending {len(perfected_demark)} perfected DeMark...")
        dm_msg = "🎯 *PERFECTED DEMARK*\n\n"
        
        perf_sorted = sorted(perfected_demark, 
                           key=lambda x: (x['demark']['tf'] == 'Weekly', 
                                        x['demark']['signal'] == '13'), 
                           reverse=True)
        
        for res in perf_sorted[:10]:
            dm_msg += format_signal_block(res, show_all=False)
        
        send_telegram(dm_msg)
    else:
        print("  No perfected DeMark")
    
    if imperfected_demark:
        print(f"  Sending {len(imperfected_demark)} imperfected DeMark...")
        im_msg = "IMPERFECTED DEMARK\n\n"
        imperf_sorted = sorted(imperfected_demark,
                         key=lambda x: (x['demark']['tf'] == 'Weekly',
                                      x['demark']['signal'] == '13'),
                         reverse=True)
    
    for res in imperf_sorted[:10]:
        dm = res['demark']
        im_msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
        im_msg += f"• {dm['type']} {dm['signal']} ({dm['tf']})\n"
        im_msg += f"• Targets: {fmt_price(res['target_1'])} / {fmt_price(res['target_2'])}\n"
        im_msg += f"• Stop: {fmt_price(res['stop'])}\n"
        im_msg += f"• Timing: {res['timing']}\n\n"
    
    send_telegram(im_msg)

if squeeze_setups:
    print(f"  Sending {len(squeeze_setups)} squeezes...")
    sq_msg = "💥 *SQUEEZE SETUPS*\n\n"
    
    sq_sorted = sorted(squeeze_setups,
                      key=lambda x: x['squeeze']['tf'] == 'Weekly',
                      reverse=True)
    
    for res in sq_sorted[:10]:
        if res in power:
            continue
        sq = res['squeeze']
        sq_msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
        sq_msg += f"• {sq['tf']} | {sq['bias']}\n"
        sq_msg += f"• Move: ±{fmt_price(sq['move'])}\n"
        sq_msg += f"• Targets: {fmt_price(res['target_1'])} / {fmt_price(res['target_2'])}\n"
        sq_msg += f"• Stop: {fmt_price(res['stop'])}\n\n"
    
    send_telegram(sq_msg)

summary = f"\n📊 *SCAN SUMMARY*\n\n"
summary += f"Universe: {len(universe)} tickers\n"
summary += f"Analyzed: {successful_scans}\n"
summary += f"Failed: {failed_scans}\n\n"
summary += f"Power: {len(power)}\n"
summary += f"Perfected DM: {len(perfected_demark)}\n"
summary += f"Imperfected DM: {len(imperfected_demark)}\n"
summary += f"Squeeze: {len(squeeze_setups)}\n"
summary += f"Breakouts: {len(momentum_breakouts)}\n"
summary += f"Oversold: {len(oversold_rsi)}\n"
summary += f"Overbought: {len(overbought_rsi)}\n"

if failed_scans > 0:
    summary += f"\n⚠️ {failed_scans} tickers failed\n"

send_telegram(summary)

print("\n✓ Complete")
