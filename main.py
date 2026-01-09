import yfinance as yf
import pandas as pd
import pandas_datareader.data as web 
import requests
import os
import time
import io
import datetime
import numpy as np

# --- CONFIGURATION (UPDATED WITH DJT) ---
STRATEGIC_TICKERS = [
    # -- Meme / PolitiFi --
    'DJT', 'PENGU-USD', 'FARTCOIN-USD', 'DOGE-USD',
    
    # -- Crypto: Coins --
    'BTC-USD', 'ETH-USD', 'SOL-USD',
    
    # -- Crypto: Miners & Infrastructure --
    'BTDR', 'MARA', 'RIOT', 'HUT', 'CLSK', 'IREN', 'CIFR', 'BTBT',
    'WYFI', 'CORZ', 'CRWV', 'APLD', 'NBIS', 'WULF', 'HIVE', 'BITF',
    'WGMI', 'MNRS', 'OWNB', 'BMNR', 'SBET', 'FWDI', 'BKKT',
    
    # -- Crypto: ETFs & Proxies --
    'IBIT', 'ETHA', 'BITQ', 'BSOL', 'GSOL', 'SOLT',
    'MSTR', 'COIN', 'HOOD', 'GLXY', 'STKE', 'DFDV', 'NODE', 'GEMI', 'BLSH',
    'CRCL',
    
    # -- Commodities (Futures & Proxies) --
    'GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'PL=F', 'PA=F', # Futures
    'GLD', 'SLV', 'PALL', 'PPLT', 'NIKL', 'LIT', 'ILIT', 'REMX',
    
    # -- Energy, Uranium & Grid --
    'VOLT', 'GRID', 'EQT', 'TAC', 'BE', 'OKLO', 'SMR', 'NEE', 
    'URA', 'SRUUF', 'CCJ', 'KAMJY', 'UNL',
    
    # -- Metal Miners --
    'XME', 'PICK', 'GDX', 'SILV', 'SLVP', 'COPX', 'LAC',
    
    # -- Tech, AI & Mag 7 --
    'NVDA', 'SMH', 'SMHX', 'TSM', 'AVGO', 'QCOM', 'MU', 'AMD', 'TER', 
    'NOW', 'AXON', 'SNOW', 'PLTR', 'GOOG', 'MSFT', 'META', 'AMZN', 'AAPL',
    'TSLA', 'NFLX', 'SPOT', 'SHOP', 'UBER', 'DASH', 'NET', 'DXCM', 'ETSY',
    'SQ', 'FIG', 'MAGS', 'MTUM', 'IVES',
    
    # -- Innovation / Fundstrat --
    'ARKK', 'ARKF', 'ARKG', 'GRNY', 'GRNI', 'GRNJ', 'XBI', 'XHB',
    
    # -- Sectors --
    'XLK', 'XLI', 'XLU', 'XLRE', 'XLB', 'XLV', 'XLF', 'XLE', 'XLP', 'XLY', 'XLC',
    
    # -- International --
    'BABA', 'JD', 'BIDU', 'PDD', 'XIACY', 'BYDDY', 'LKNCY', 'TCEHY',
    'MCHI', 'INDA', 'EWZ', 'EWJ', 'EWG', 'EWU', 'EWY', 'EWW', 'EWT', 'EWC', 'EEM',
    'AMX', 'PBR', 'VALE', 'NSRGY', 'DEO',
    
    # -- Financials / Bio / Other --
    'BLK', 'STT', 'ARES', 'SOFI', 'PYPL', 'IBKR', 'WU',
    'RXRX', 'SDGR', 'TEM', 'ABSI', 'DNA', 'TWST', 'GLW', 
    'KHC', 'LULU', 'YETI', 'DLR', 'EQIX', 'ORCL', 'LSF'
]

# --- HELPER: TELEGRAM SENDER ---
def send_telegram_alert(message, header=""):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    full_msg = f"{header}\n\n{message}" if header else message
    if len(full_msg) > 4000:
        for i in range(0, len(full_msg), 4000):
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": full_msg[i:i+4000], "parse_mode": "Markdown"})
    else:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                      json={"chat_id": chat_id, "text": full_msg, "parse_mode": "Markdown"})

# ==========================================================
#  ENGINE 1: MARKET RADAR (Regime Model)
# ==========================================================
def get_market_radar_regime():
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=365)
        # 1. Net Liquidity (Fed Assets - TGA - RRP)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        net_liq = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
        
        # 2. Growth (SPY)
        spy = yf.download('SPY', start=start, progress=False)['Close']
        if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
        
        # 3. Momentum (3-Month ROC)
        liq_momo = net_liq.pct_change(63).iloc[-1] * 100
        growth_momo = spy.pct_change(63).iloc[-1] * 100
        
        if liq_momo > 0 and growth_momo > 0:
            return f"🟢 **REGIME: REFLATION (Risk On)**\n   └ Liq: {liq_momo:.2f}% | Growth: {growth_momo:.2f}%"
        elif liq_momo > 0 and growth_momo < 0:
            return f"🔵 **REGIME: RECOVERY (Accumulate)**\n   └ Liq: {liq_momo:.2f}% | Growth: {growth_momo:.2f}%"
        elif liq_momo < 0 and growth_momo < 0:
            return f"🔴 **REGIME: DEFLATION (Slowdown)**\n   └ Liq: {liq_momo:.2f}% | Growth: {growth_momo:.2f}%"
        else:
            return f"🟠 **REGIME: STAGFLATION (Turbulence)**\n   └ Liq: {liq_momo:.2f}% | Growth: {growth_momo:.2f}%"
    except: return "⚠️ Market Radar Data Unavailable"

# ==========================================================
#  ENGINE 2: CAPITAL WARS (Michael Howell Liquidity Cycle)
# ==========================================================
def get_capital_wars_regime():
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=365)
        fred = web.DataReader(['WALCL', 'WTREGEN', 'RRPONTSYD'], 'fred', start, datetime.datetime.now())
        fred = fred.resample('D').ffill().dropna()
        
        fed_assets = fred['WALCL'] / 1000
        net_liq = fed_assets - fred['WTREGEN'] - fred['RRPONTSYD']
        
        # Trends
        liq_trend_short = net_liq.pct_change(20).iloc[-1]
        liq_trend_long = net_liq.pct_change(60).iloc[-1]
        fed_trend = fed_assets.pct_change(60).iloc[-1]
        
        phase = ""
        action = ""
        if liq_trend_long > 0:
            if liq_trend_short > 0:
                phase = "SPECULATION (Late Cycle)"
                action = "Assets: Hard Assets / Equity / Bitcoin"
            else:
                phase = "CALM (Mid Cycle)"
                action = "Assets: Quality Equity / Credit"
        else:
            if liq_trend_short > 0:
                phase = "REBOUND (Early Cycle)"
                action = "Assets: Gov Bonds / Early Tech"
            else:
                phase = "TURBULENCE (Crisis)"
                action = "Assets: Cash / Gold / Volatility"

        # Treasury QE Detector (Baton Pass)
        treasury_qe = (liq_trend_long > 0 and fed_trend <= 0)
            
        msg = f"🌊 **CYCLE PHASE:** {phase}\n   └ Action: {action}\n"
        if treasury_qe: msg += "   └ 🚨 **Signal:** 'Treasury QE' Detected (Baton Pass)"
        else: msg += "   └ Signal: Standard Fed Liquidity"
        return msg
    except: return "⚠️ Capital Wars Data Unavailable"

# ==========================================================
#  ENGINE 3: DEMARK STOCK SCANNER
# ==========================================================
def calculate_demark(df):
    df['Close_4'] = df['Close'].shift(4)
    df['Buy_Setup'] = 0; df['Sell_Setup'] = 0
    df['Buy_Countdown'] = 0; df['Sell_Countdown'] = 0
    df['Buy_13_Perfected'] = False; df['Sell_13_Perfected'] = False

    buy_seq = 0; sell_seq = 0
    buy_cd = 0; sell_cd = 0
    active_buy = False; active_sell = False
    buy_idxs = []; sell_idxs = []

    closes = df['Close'].values; closes_4 = df['Close_4'].values
    lows = df['Low'].values; highs = df['High'].values
    
    for i in range(4, len(df)):
        # Setup
        if closes[i] < closes_4[i]: buy_seq += 1
        else: buy_seq = 0
        df.iloc[i, df.columns.get_loc('Buy_Setup')] = buy_seq
        
        if closes[i] > closes_4[i]: sell_seq += 1
        else: sell_seq = 0
        df.iloc[i, df.columns.get_loc('Sell_Setup')] = sell_seq
        
        # Reset (Sequential Logic)
        if buy_seq == 9:
            active_buy = True; buy_cd = 0; buy_idxs = []
            active_sell = False
        if sell_seq == 9:
            active_sell = True; sell_cd = 0; sell_idxs = []
            active_buy = False

        # Countdown
        if active_buy:
            if closes[i] <= lows[i-2]:
                buy_cd += 1; buy_idxs.append(i)
                df.iloc[i, df.columns.get_loc('Buy_Countdown')] = buy_cd
                if buy_cd == 13:
                    if len(buy_idxs) >= 8 and lows[i] <= closes[buy_idxs[7]]:
                        df.iloc[i, df.columns.get_loc('Buy_13_Perfected')] = True
                    active_buy = False

        if active_sell:
            if closes[i] >= highs[i-2]:
                sell_cd += 1; sell_idxs.append(i)
                df.iloc[i, df.columns.get_loc('Sell_Countdown')] = sell_cd
                if sell_cd == 13:
                    if len(sell_idxs) >= 8 and highs[i] >= closes[sell_idxs[7]]:
                        df.iloc[i, df.columns.get_loc('Sell_13_Perfected')] = True
                    active_sell = False
    return df

def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex):
            try: df.columns = df.columns.get_level_values(0)
            except: pass
        if 'Close' not in df.columns: return None

        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df = calculate_demark(df)
        last = df.iloc[-1]
        price = last['Close']
        sma = last['SMA_200']
        
        signal = None
        
        # Priority: 13 -> 9
        if last['Buy_Countdown'] == 13:
            signal = {
                'ticker': ticker, 'type': 'BUY', 'algo': 'COUNTDOWN 13',
                'price': price, 'perfected': last['Buy_13_Perfected'],
                'action': 'ACCUMULATE (Bottom)', 'timing': 'Weeks',
                'stop': min(df['Low'].iloc[-13:]), 'target': price * 1.15,
                'trend': 'Bullish Dip' if price > sma else 'Counter-Trend'
            }
        elif last['Sell_Countdown'] == 13:
            signal = {
                'ticker': ticker, 'type': 'SELL', 'algo': 'COUNTDOWN 13',
                'price': price, 'perfected': last['Sell_13_Perfected'],
                'action': 'DISTRIBUTE (Top)', 'timing': 'Weeks',
                'stop': max(df['High'].iloc[-13:]), 'target': price * 0.85,
                'trend': 'Extension' if price > sma else 'Bearish'
            }
        elif last['Buy_Setup'] == 9:
            stop = min(df['Low'].iloc[-9:])
            risk = max(price - stop, price * 0.01)
            # Perfection Logic
            l9, l8, l7, l6 = df['Low'].iloc[-1], df['Low'].iloc[-2], df['Low'].iloc[-3], df['Low'].iloc[-4]
            perf = (l9 < l7 and l9 < l6) or (l8 < l7 and l8 < l6)
            
            signal = {
                'ticker': ticker, 'type': 'BUY', 'algo': 'SETUP 9',
                'price': price, 'perfected': perf,
                'action': 'SCALP (Bounce)', 'timing': '1-4 Days',
                'stop': stop, 'target': price + (risk * 2),
                'trend': 'Bullish Dip' if price > sma else 'Counter-Trend'
            }
        elif last['Sell_Setup'] == 9:
            stop = max(df['High'].iloc[-9:])
            risk = max(stop - price, price * 0.01)
            h9, h8, h7, h6 = df['High'].iloc[-1], df['High'].iloc[-2], df['High'].iloc[-3], df['High'].iloc[-4]
            perf = (h9 > h7 and h9 > h6) or (h8 > h7 and h8 > h6)
            
            signal = {
                'ticker': ticker, 'type': 'SELL', 'algo': 'SETUP 9',
                'price': price, 'perfected': perf,
                'action': 'SCALP (Pullback)', 'timing': '1-4 Days',
                'stop': stop, 'target': price - (risk * 2),
                'trend': 'Bearish' if price < sma else 'Extension'
            }
        return signal
    except: return None

# ==========================================================
#  MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print("1. Generating Macro Report...")
    radar = get_market_radar_regime()
    capital_wars = get_capital_wars_regime()
    
    macro_msg = f"🌍 **GLOBAL MACRO INSIGHTS** 🌍\n\n{radar}\n\n{capital_wars}\n───────────────"
    send_telegram_alert(macro_msg)
    
    print("2. Scanning Tickers...")
    # Ticker list is now strictly defined in STRATEGIC_TICKERS
    full_universe = list(set(STRATEGIC_TICKERS))
    
    signals = []
    for i, ticker in enumerate(full_universe):
        if i % 50 == 0: print(f"Processing {i}/{len(full_universe)}...")
        res = analyze_ticker(ticker)
        if res: signals.append(res)
        time.sleep(0.05)
        
    if signals:
        # Sort by Importance: 13s -> 9s
        signals.sort(key=lambda x: (x['algo'] == 'SETUP 9', x['type']))
        
        stock_msg = "🔔 **DEMARK SIGNALS (INSTITUTIONAL)** 🔔\n"
        for s in signals:
            icon = "🟢" if "BUY" in s['type'] else "🔴"
            perf = "⭐" if s['perfected'] else "⚠️"
            stock_msg += f"{icon} **{s['ticker']}** [{s['algo']}] {perf}\n"
            stock_msg += f"   ⚡ {s['action']} ({s['trend']})\n"
            stock_msg += f"   🎯 ${s['target']:.2f} | 🛑 ${s['stop']:.2f}\n"
            stock_msg += f"   ⏳ {s['timing']}\n"
            stock_msg += "───────────────\n"
            
        send_telegram_alert(stock_msg)
    else:
        print("No stock signals found.")
    
    print("Done.")
