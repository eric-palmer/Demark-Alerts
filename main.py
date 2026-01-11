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
    """Calculate expected timing for signal based on historical volatility and price action"""
    try:
        # Calculate ATR for volatility
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Calculate average daily move
        avg_daily_move = df['Close'].pct_change().abs().rolling(20).mean().iloc[-1]
        
        # Estimate time to target based on volatility
        if timeframe == 'Weekly':
            min_days = 5
            max_days = 30
        else:  # Daily
            min_days = 1
            max_days = 15
        
        # Adjust based on volatility - higher vol = faster targets
        if avg_daily_move > 0.03:  # >3% daily moves
            return f"{min_days}-{int(max_days/2)} days"
        elif avg_daily_move > 0.015:  # 1.5-3% moves
            return f"{int(min_days*2)}-{max_days} days"
        else:  # Low volatility
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
    """Suggest position size based on 2% account risk per trade"""
    try:
        risk_per_share = abs(price - stop)
        # For $100k account at 2% risk = $2000 risk budget
        # Position size = Risk Budget / Risk Per Share
        # We'll show shares for a $10k risk budget as example
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
        
        # Filter dead tickers
        if not is_portfolio and (df['Volume'].iloc[-5:].sum() == 0 or df['Close'].iloc[-1] < 0.00000001):
            return None
        
        # Weekly data for longer-term signals
        df_w = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        if len(df_w) < 20:
            return None
        
        # Calculate all indicators
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
        
        # Enhanced ATR-based targets with multiple timeframes
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        
        atr_14 = tr.rolling(14).mean().iloc[-1]
        atr_20 = tr.rolling(20).mean().iloc[-1]
        
        # Support/Resistance levels
        recent_high = df['High'].iloc[-20:].max()
        recent_low = df['Low'].iloc[-20:].min()
        
        # Calculate key moving averages for support/resistance
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        ema_9 = df['Close'].ewm(span=9).mean().iloc[-1]
        
        # Volume analysis
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-5:].mean()
        volume_surge = current_volume > avg_volume * 1.5
        
        # Volatility percentile (where are we in historical volatility?)
        vol_percentile = (df['Close'].pct_change().rolling(20).std().iloc[-1] / 
                         df['Close'].pct_change().rolling(100).std().mean())
        
        # Price position analysis
        price_vs_20sma = ((price - sma_20) / sma_20) * 100
        price_vs_50sma = ((price - sma_50) / sma_50) * 100
        
        # === DEMARK ANALYSIS ===
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
        
        # Enhanced perfection logic
        if dm:
            if dm['signal'] == '13':
                dm_perf = True  # All 13s are automatically perfected
            elif dm['type'] == 'BUY' and len(df) >= 4:
                # Perfected buy: new low below lows of bars 2 and 3 of setup
                dm_perf = (df['Low'].iloc[-1] < df['Low'].iloc[-3] and 
                          df['Low'].iloc[-1] < df['Low'].iloc[-4])
            elif dm['type'] == 'SELL' and len(df) >= 4:
                # Perfected sell: new high above highs of bars 2 and 3 of setup
                dm_perf = (df['High'].iloc[-1] > df['High'].iloc[-3] and 
                          df['High'].iloc[-1] > df['High'].iloc[-4])
            dm['perfected'] = dm_perf
        
        # === RSI ANALYSIS ===
        rsi_sig = None
        rsi_val = last['RSI']
        if rsi_val < 30:
            rsi_sig = {'type': 'OVERSOLD', 'val': rsi_val, 'strength': 'STRONG' if rsi_val < 20 else 'MODERATE'}
        elif rsi_val > 70:
            rsi_sig = {'type': 'OVERBOUGHT', 'val': rsi_val, 'strength': 'STRONG' if rsi_val > 80 else 'MODERATE'}
        
        # RSI divergence detection
        rsi_divergence = None
        if len(df) >= 20:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if df['Close'].iloc[-1] < df['Close'].iloc[-10] and df['RSI'].iloc[-1] > df['RSI'].iloc[-10]:
                rsi_divergence = 'BULLISH'
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif df['Close'].iloc[-1] > df['Close'].iloc[-10] and df['RSI'].iloc[-1] < df['RSI'].iloc[-10]:
                rsi_divergence = 'BEARISH'
        
        # === SQUEEZE ANALYSIS ===
        sq = None
        if d_sq:
            sq = {'tf': 'Daily', 'bias': d_sq['bias'], 'move': d_sq['move']}
        elif w_sq:
            sq = {'tf': 'Weekly', 'bias': w_sq['bias'], 'move': w_sq['move']}
        
        # === MACD ANALYSIS ===
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
        
        # === STOCHASTIC ANALYSIS ===
        sto_sig = None
        if len(sto_k) >= 2:
            if sto_k.iloc[-1] > sto_d.iloc[-1] and sto_k.iloc[-2] <= sto_d.iloc[-2] and sto_k.iloc[-1] < 20:
                sto_sig = {'type': 'BULLISH CROSS OVERSOLD', 'k': sto_k.iloc[-1], 'd': sto_d.iloc[-1]}
            elif sto_k.iloc[-1] < sto_d.iloc[-1] and sto_k.iloc[-2] >= sto_d.iloc[-2] and sto_k.iloc[-1] > 80:
                sto_sig = {'type': 'BEARISH CROSS OVERBOUGHT', 'k': sto_k.iloc[-1], 'd': sto_d.iloc[-1]}
        
        # === ADX TREND STRENGTH ===
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
        
        # === TREND CLASSIFICATION ===
        trend = "BULLISH" if price > sma_200 else "BEARISH"
        trend_quality = "STRONG" if abs(price - sma_200) / sma_200 > 0.05 else "WEAK"
        
        # === INTELLIGENT TARGET & STOP CALCULATION ===
        # Use confluence of technical levels
        if dm and dm['type'] == 'BUY':
            # For buy signals
            target_1 = price + (atr_14 * 1.5)  # Conservative
            target_2 = price + (atr_14 * 2.5)  # Moderate
            target_3 = min(recent_high, price + (atr_14 * 4))  # Aggressive (capped at recent high)
            
            # Stop loss below recent support or 1.5 ATR
            stop = max(recent_low, price - (atr_14 * 1.5))
            
            # Adjust stop if near key MA
            if abs(price - sma_20) / price < 0.02:
                stop = min(stop, sma_20 * 0.97)  # Just below 20 SMA
            
        elif dm and dm['type'] == 'SELL':
            # For sell signals
            target_1 = price - (atr_14 * 1.5)
            target_2 = price - (atr_14 * 2.5)
            target_3 = max(recent_low, price - (atr_14 * 4))
            
            stop = min(recent_high, price + (atr_14 * 1.5))
            
            if abs(price - sma_20) / price < 0.02:
                stop = max(stop, sma_20 * 1.03)
        else:
            # No clear signal - use standard levels
            target_1 = price + (atr_14 * 2)
            target_2 = price + (atr_14 * 3)
            target_3 = recent_high
            stop = price - (atr_14 * 1.5)
        
        # === SIGNAL CONFLUENCE SCORE ===
        # Count how many indicators agree
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
        
        # === VERDICT WITH CONFIDENCE ===
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
        
        # === TIMING CALCULATION ===
        timing = calculate_timing(df, dm_timeframe if dm else 'Daily')
        
        # === RISK METRICS ===
        rr_ratio = calculate_risk_reward(price, target_2, stop)
        pos_size = calculate_position_size_suggestion(price, stop)
        
        # Setup count display
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
            'sma_50': sma_50,
            'ema_9': ema_9
        }
    except Exception as e:
        print(f"Error {ticker}: {e}")
        return None

def format_signal_block(res, show_all=False):
    """Format a standardized signal block for any ticker"""
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
            msg += f"• Squeeze: {res['squeeze']['tf']} {res['squeeze']['bias']} (Target move: {fmt_price(res['squeeze']['move'])})\n"
        if res['macd']:
            msg += f"• MACD: {res['macd']['type']} ({res['macd'].get('strength', 'N/A')})\n"
        if res['stochastic']:
            msg += f"• Stochastic: {res['stochastic']['type']}\n"
        if res['adx']:
            msg += f"• ADX: {res['adx']['strength']} {res['adx']['direction']} (ADX: {res['adx']['adx']:.1f})\n"
        if res['fib']:
            msg += f"• Fibonacci: {res['fib']['action']} at {res['fib']['level']}\n"
        if res['shannon']['breakout']:
            msg += f"• AlphaTrends: Momentum Breakout (10/20 cross above 50)\n"
    
    return msg

if __name__ == "__main__":
    print("="*60)
    print("INSTITUTIONAL TRADING ALERT SYSTEM")
    print("="*60)
    
    # Pre-flight checks
    print("\n[PRE-FLIGHT CHECKS]")
    if not os.environ.get('TELEGRAM_TOKEN') or not os.environ.get('TELEGRAM_CHAT_ID'):
        print("⚠️  Telegram credentials missing")
    else:
        print("✓ Telegram configured")
    if not os.environ.get('FRED_API_KEY'):
        print("⚠️  FRED_API_KEY missing")
    else:
        print("✓ FRED configured")
    
    # === MACRO ANALYSIS ===
    print("\n[1/4] MACRO ANALYSIS")
    macro = get_macro()
    msg = "📊 *GLOBAL MACRO SNAPSHOT*\n\n"
    
    if macro:
        liq = macro['net_liq'].pct_change(63).iloc[-1]
        liq_6m = macro['net_liq'].pct_change(126).iloc[-1]
        regime = "🟢 RISK ON" if liq > 0 else "🔴 RISK OFF"
        
        term_spread = macro['term_premia'].iloc[-1]
        yield_curve = "NORMAL" if term_spread > 0 else "INVERTED"
        
        inflation = macro['inflation'].iloc[-1]
        
        msg += f"*Market Regime:* {regime}\n"
        msg += f"• Net Liquidity (3M): {liq*100:+.2f}%\n"
        msg += f"• Net Liquidity (6M): {liq_6m*100:+.2f}%\n"
        msg += f"• Yield Curve: {yield_curve} ({term_spread:.2f}%)\n"
        msg += f"• 5Y Inflation Exp: {inflation:.2f}%\n\n"
        
        if liq > 0.05:
            msg += "💡 *Interpretation:* Strong liquidity expansion favors risk assets\n"
        elif liq < -0.05:
            msg += "⚠️ *Interpretation:* Liquidity contraction - defensive posture recommended\n"
        else:
            msg += "📊 *Interpretation:* Neutral liquidity environment\n"
    else:
        msg += "Macro data temporarily unavailable\n"
    
    msg += "\n"
    send_telegram(msg)
    
    # === PORTFOLIO ANALYSIS ===
    print("\n[2/4] PORTFOLIO ANALYSIS")
    p_msg = "💼 *PORTFOLIO DEEP DIVE*\n\n"
    
    for t in CURRENT_PORTFOLIO:
        print(f"  Analyzing {t}...")
        res = analyze_ticker(t, is_portfolio=True)
        if res:
            p_msg += format_signal_block(res, show_all=True)
    
    send_telegram(p_msg)
    
    # === MARKET SCAN ===
    print("\n[3/4] SCANNING MARKET UNIVERSE")
    universe = list(set(STRATEGIC_TICKERS + get_futures()))
    
    power = []
    perfected_demark = []
    imperfected_demark = []
    squeeze_setups = []
    momentum_breakouts = []
    oversold_rsi = []
    overbought_rsi = []
    
    def process(t):
        return analyze_ticker(t)
    
    completed = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process, t): t for t in universe}
        for future in as_completed(futures):
            completed += 1
            if completed % 20 == 0:
                print(f"  Progress: {completed}/{len(universe)}")
            
            res = future.result()
            if not res:
                continue
            
            # Power rankings: 4+ signals in agreement
            if res['signal_score'] >= 4:
                power.append(res)
            
            # DeMark signals
            if res['demark']:
                if res['demark']['perfected']:
                    perfected_demark.append(res)
                else:
                    imperfected_demark.append(res)
            
            # Other signal categories
            if res['squeeze']:
                squeeze_setups.append(res)
            if res['shannon']['breakout']:
                momentum_breakouts.append(res)
            if res['rsi'] and res['rsi']['type'] == 'OVERSOLD':
                oversold_rsi.append(res)
            if res['rsi'] and res['rsi']['type'] == 'OVERBOUGHT':
                overbought_rsi.append(res)
    
    print(f"  Scan complete: {completed} tickers analyzed")
    
    # === ALERTS GENERATION ===
    print("\n[4/4] GENERATING ALERTS")
    
    # POWER RANKINGS
    if power:
        a_msg = "⭐ *POWER RANKINGS* (High Conviction Setups)\n"
        a_msg += "Multiple indicators in strong agreement\n\n"
        
        # Sort by signal score
        power_sorted = sorted(power, key=lambda x: x['signal_score'], reverse=True)
        
        for res in power_sorted[:8]:
            a_msg += format_signal_block(res, show_all=False)
        
        send_telegram(a_msg)
    
# PERFECTED DEMARK
    if perfected_demark:
        dm_msg = "🎯 *PERFECTED DEMARK SIGNALS*\n"
        dm_msg += "Highest probability reversal setups\n\n"
        
        # Sort by timeframe (Weekly first) and signal type (13 before 9)
        perf_sorted = sorted(perfected_demark, 
                           key=lambda x: (x['demark']['tf'] == 'Weekly', 
                                        x['demark']['signal'] == '13'), 
                           reverse=True)
        
        for res in perf_sorted[:10]:
            dm_msg += format_signal_block(res, show_all=False)
        
        send_telegram(dm_msg)
    
    # IMPERFECTED DEMARK
    if imperfected_demark:
        im_msg = "○ *IMPERFECTED DEMARK SIGNALS*\n"
        im_msg += "Monitor for perfection or cancellation\n\n"
        
        imperf_sorted = sorted(imperfected_demark,
                             key=lambda x: (x['demark']['tf'] == 'Weekly',
                                          x['demark']['signal'] == '13'),
                             reverse=True)
        
        for res in imperf_sorted[:10]:
            dm = res['demark']
            im_msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
            im_msg += f"• Signal: {dm['type']} {dm['signal']} ({dm['tf']})\n"
            im_msg += f"• Targets: {fmt_price(res['target_1'])} / {fmt_price(res['target_2'])}\n"
            im_msg += f"• Stop: {fmt_price(res['stop'])}\n"
            im_msg += f"• Timing: {res['timing']}\n"
            im_msg += f"• Needs: Price action perfection\n\n"
        
        send_telegram(im_msg)
    
    # SQUEEZE SETUPS
    if squeeze_setups:
        sq_msg = "💥 *VOLATILITY SQUEEZE SETUPS*\n"
        sq_msg += "Compression patterns ready to expand\n\n"
        
        # Sort by timeframe (Weekly first)
        sq_sorted = sorted(squeeze_setups,
                          key=lambda x: x['squeeze']['tf'] == 'Weekly',
                          reverse=True)
        
        for res in sq_sorted[:10]:
            if res in power:
                continue
            sq = res['squeeze']
            sq_msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
            sq_msg += f"• Timeframe: {sq['tf']}\n"
            sq_msg += f"• Bias: {sq['bias']}\n"
            sq_msg += f"• Expected Move: ±{fmt_price(sq['move'])}\n"
            sq_msg += f"• Targets: {fmt_price(res['target_1'])} / {fmt_price(res['target_2'])}\n"
            sq_msg += f"• Stop: {fmt_price(res['stop'])}\n"
            sq_msg += f"• Timing: {res['timing']}\n\n"
        
        send_telegram(sq_msg)
    
    # MOMENTUM BREAKOUTS
    if momentum_breakouts:
        mo_msg = "📈 *ALPHATRENDS BREAKOUTS*\n"
        mo_msg += "10/20 SMA cross with 50 SMA confirmation\n\n"
        
        for res in momentum_breakouts[:10]:
            if res in power:
                continue
            mo_msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
            mo_msg += f"• Setup: Bullish MA alignment\n"
            mo_msg += f"• Targets: {fmt_price(res['target_1'])} / {fmt_price(res['target_2'])}\n"
            mo_msg += f"• Stop: {fmt_price(res['stop'])} (below 20 SMA)\n"
            mo_msg += f"• Timing: {res['timing']}\n"
            mo_msg += f"• R/R: {res['rr_ratio']}\n\n"
        
        send_telegram(mo_msg)
    
    # OVERSOLD OPPORTUNITIES
    if oversold_rsi:
        os_msg = "🔵 *OVERSOLD CONDITIONS*\n"
        os_msg += "RSI < 30 - potential bounce candidates\n\n"
        
        # Sort by RSI (lowest first)
        os_sorted = sorted(oversold_rsi, key=lambda x: x['rsi_val'])
        
        for res in os_sorted[:8]:
            if res in power:
                continue
            os_msg += f"*{res['ticker']}* @ {fmt_price(res['price'])}\n"
            os_msg += f"• RSI: {res['rsi_val']:.1f} ({res['rsi']['strength']})\n"
            if res['rsi_divergence']:
                os_msg += f"• Divergence: {res['rsi_divergence']}\n"
            os_msg += f"• Targets: {fmt_price(res['target_1'])} / {fmt_price(res['target_2'])}\n"
            os_msg += f"• Stop: {fmt_price(res['stop'])}\n"
            os_msg += f"• Timing: {res['timing']}\n\n"
        
        send_telegram(os_msg)
    
    # Summary statistics
    summary = f"\n📊 *SCAN SUMMARY*\n\n"
    summary += f"Universe Scanned: {len(universe)} tickers\n"
    summary += f"Power Setups: {len(power)}\n"
    summary += f"Perfected DeMark: {len(perfected_demark)}\n"
    summary += f"Imperfected DeMark: {len(imperfected_demark)}\n"
    summary += f"Squeeze Setups: {len(squeeze_setups)}\n"
    summary += f"Momentum Breakouts: {len(momentum_breakouts)}\n"
    summary += f"Oversold (RSI<30): {len(oversold_rsi)}\n"
    summary += f"Overbought (RSI>70): {len(overbought_rsi)}\n"
    
    send_telegram(summary)
    
    print("\n✓ Scan Complete")
    print(f"  Power Setups: {len(power)}")
    print(f"  Perfected DeMark: {len(perfected_demark)}")
    print(f"  Total Signals: {len(power) + len(perfected_demark) + len(imperfected_demark)}")
