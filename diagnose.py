# diagnose.py - Data X-Ray
import os
import pandas as pd
from tiingo import TiingoClient

# 1. Setup Client
api_key = os.environ.get('TIINGO_API_KEY')
if not api_key:
    print("❌ ERROR: TIINGO_API_KEY is missing from Secrets.")
    exit()

client = TiingoClient({'api_key': api_key, 'session': True})
print("✅ Client Initialized. Fetching Data...\n")

def inspect_ticker(ticker):
    print(f"--- INSPECTING {ticker} ---")
    try:
        # Fetch 20 days of data
        df = client.get_dataframe(ticker, startDate='2025-12-01')
        
        # 1. Print Raw Columns
        print(f"Raw Columns Received: {list(df.columns)}")
        
        # 2. Check Data Types
        print("\nData Types:")
        print(df.dtypes)
        
        # 3. Print First 5 Rows (To see the actual numbers)
        print("\nFirst 5 Rows of Data:")
        # Select key columns to display
        cols = [c for c in df.columns if 'open' in c.lower() or 'high' in c.lower() or 'low' in c.lower() or 'close' in c.lower()]
        print(df[cols].head())
        
        # 4. Volatility Check (The "Flat Candle" Test)
        # We need to know if High is different from Low
        # Handle different column names (adjHigh vs high)
        high_col = 'adjHigh' if 'adjHigh' in df.columns else 'high'
        low_col = 'adjLow' if 'adjLow' in df.columns else 'low'
        
        if high_col in df.columns and low_col in df.columns:
            diff = (df[high_col] - df[low_col]).mean()
            print(f"\nAverage Volatility (High - Low): {diff:.4f}")
            if diff == 0:
                print("⚠️ CRITICAL FAILURE: High equals Low. Indicators will crash.")
            else:
                print("✅ Volatility Detected. Data looks healthy.")
        else:
            print(f"❌ CRITICAL FAILURE: Could not find High/Low columns. Found: {list(df.columns)}")

    except Exception as e:
        print(f"❌ FETCH ERROR: {e}")
    print("\n" + "="*30 + "\n")

# Run X-Ray
inspect_ticker('SLV')
inspect_ticker('DJT')
