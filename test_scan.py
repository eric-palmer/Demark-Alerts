# test_scan.py - Diagnostic scanner
from data_fetcher import safe_download
import pandas as pd

test_tickers = [
    'SLV', 'DJT',  # Your portfolio
    'AAPL', 'NVDA', 'TSLA',  # Major stocks
    'BTC-USD', 'ETH-USD',  # Crypto
    'ES=F', 'GC=F',  # Futures
    'SPY', 'QQQ'  # ETFs
]

print("="*60)
print("DATA FETCH TEST")
print("="*60)

success = []
failed = []

for ticker in test_tickers:
    print(f"\nTesting {ticker}...")
    df = safe_download(ticker)
    
    if df is not None and len(df) >= 50:
        print(f"  ✓ Success: {len(df)} days of data")
        print(f"  Latest: {df.index[-1]} @ ${df['Close'].iloc[-1]:.2f}")
        success.append(ticker)
    else:
        print(f"  ✗ Failed")
        failed.append(ticker)

print("\n" + "="*60)
print(f"RESULTS: {len(success)}/{len(test_tickers)} successful")
print(f"Success: {', '.join(success)}")
print(f"Failed: {', '.join(failed)}")
