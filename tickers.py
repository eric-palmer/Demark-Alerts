# tickers.py - Dynamic Universe Sourcing
import pandas as pd

def get_sp500():
    """Scrapes live S&P 500 list"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        return [t.replace('.', '-') for t in df['Symbol'].tolist()]
    except: 
        return ['SPY', 'MSFT', 'AAPL', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM']

def get_nasdaq100():
    """Scrapes live Nasdaq 100 list"""
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables = pd.read_html(url)
        # Wikipedia table index varies, search for correct table
        for t in tables:
            if 'Ticker' in t.columns: return [x.replace('.', '-') for x in t['Ticker'].tolist()]
            elif 'Symbol' in t.columns: return [x.replace('.', '-') for x in t['Symbol'].tolist()]
        return ['QQQ', 'NVDA', 'AAPL', 'MSFT']
    except: return ['QQQ', 'NVDA', 'AAPL', 'MSFT']

def get_top_crypto():
    """High-Liquidity Crypto (Tiingo Format)"""
    return [
        'btcusd', 'ethusd', 'solusd', 'bnbusd', 'xrpusd', 'adausd', 'dogeusd', 
        'avaxusd', 'dotusd', 'linkusd', 'maticusd', 'shibusd', 'ltcusd', 
        'bchusd', 'atomusd', 'uniusd', 'xmrusd', 'etcusd', 'xlmusd', 'filusd',
        'hbarusd', 'aptusd', 'ldousd', 'nearusd', 'qntusd', 'algousd', 'stxusd',
        'aaveusd', 'imxusd', 'ftmusd', 'sandusd', 'manausd', 'thetausd', 'vetusd'
    ]

def get_universe():
    print("   🌐 Fetching Dynamic Ticker Lists...")
    sp = get_sp500()
    ndx = get_nasdaq100()
    cry = get_top_crypto()
    # Merge and Deduplicate
    full = list(set(sp + ndx + cry))
    print(f"   ✅ Universe Loaded: {len(full)} Assets")
    return full
