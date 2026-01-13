# tickers.py - Dynamic Universe Sourcing
import pandas as pd

def get_sp500():
    """Scrapes Wikipedia for the latest S&P 500 list"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Fix dot format (BRK.B -> BRK-B) for Tiingo compatibility
        return [t.replace('.', '-') for t in tickers]
    except:
        print("⚠️ Failed to scrape S&P 500. Using backup.")
        return ['SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'JPM']

def get_nasdaq100():
    """Scrapes Wikipedia for the latest Nasdaq 100 list"""
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables = pd.read_html(url)
        # Wikipedia table index varies, usually 4th table
        for t in tables:
            if 'Ticker' in t.columns:
                return [x.replace('.', '-') for x in t['Ticker'].tolist()]
            elif 'Symbol' in t.columns:
                return [x.replace('.', '-') for x in t['Symbol'].tolist()]
        return ['QQQ', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA']
    except:
        return ['QQQ', 'NVDA', 'AAPL', 'MSFT']

def get_top_crypto():
    """
    Returns high-liquidity Tiingo-supported crypto symbols.
    Dynamic scraping is risky for crypto due to symbol mismatches.
    This list covers the 'Real' market.
    """
    return [
        'btcusd', 'ethusd', 'solusd', 'bnbusd', 'xrpusd', 'adausd', 'dogeusd', 
        'avaxusd', 'dotusd', 'linkusd', 'maticusd', 'shibusd', 'ltcusd', 
        'bchusd', 'atomusd', 'uniusd', 'xmrusd', 'etcusd', 'xlmusd', 'filusd',
        'hbarusd', 'aptusd', 'ldousd', 'nearusd', 'qntusd', 'algousd', 'stxusd',
        'aaveusd', 'imxusd', 'ftmusd', 'sandusd', 'manausd', 'thetausd', 'vetusd'
    ]

def get_universe():
    print("   🌐 Fetching Dynamic Ticker Lists...")
    sp500 = get_sp500()
    nasdaq = get_nasdaq100()
    crypto = get_top_crypto()
    
    # Merge and deduplicate
    full_list = list(set(sp500 + nasdaq + crypto))
    print(f"   ✅ Universe Loaded: {len(full_list)} Assets")
    return full_list
