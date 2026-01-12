# debug_tiingo.py - The Data X-Ray
import os
import pandas as pd
from tiingo import TiingoClient
import datetime

# 1. Setup
api_key = os.environ.get('TIINGO_API_KEY')
if not api_key:
    print("❌ NO API KEY FOUND")
    exit()

client = TiingoClient({'api_key': api_key, 'session': True})
ticker = 'SLV' # The problem child

print(f"\n🔎 INSPECTING: {ticker} via Tiingo")
print("="*40)

try:
    # 2. Fetch RAW Data (No filters)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
    print(f"Requesting data from {start_date}...")
    
    df = client.get_dataframe(ticker, startDate=start_date)
    
    # 3. INSPECTION 1: What columns did we actually get?
    print(f"\n[1] RAW COLUMNS RECEIVED:")
    print(list(df.columns))
    
    # 4. INSPECTION 2: Data Types (Are they numbers or text?)
    print(f"\n[2] DATA TYPES:")
    print(df.dtypes)
    
    # 5. INSPECTION 3: The "Ghost Data" Check
    # Print the last 5 rows of High/Low to see if they are identical or NaN
    print(f"\n[3] RECENT RAW DATA (Last 5 Days):")
    # Try to find relevant columns dynamically
    cols_to_show = [c for c in df.columns if 'high' in c.lower() or 'low' in c.lower() or 'close' in c.lower()]
    print(df[cols_to_show].tail())
    
    # 6. INSPECTION 4: The Math Test
    # Try to calculate Range (High - Low) manually
    print(f"\n[4] MATH TEST:")
    try:
        # Guess column names based on raw output
        h_col = [c for c in df.columns if 'high' in c.lower()][0]
        l_col = [c for c in df.columns if 'low' in c.lower()][0]
        
        highs = pd.to_numeric(df[h_col], errors='coerce')
        lows = pd.to_numeric(df[l_col], errors='coerce')
        
        tr = highs - lows
        print(f"   Using columns: {h_col} - {l_col}")
        print(f"   Last 5 Ranges: \n{tr.tail().values}")
        print(f"   Average Range: {tr.mean()}")
        
        if tr.mean() == 0:
            print("   ⚠️ CRITICAL: High equals Low (Flat Data). Indicators will fail.")
        elif tr.isnull().all():
            print("   ⚠️ CRITICAL: All ranges are NaN. Data conversion failed.")
        else:
            print("   ✅ Math looks valid.")
            
    except Exception as e:
        print(f"   ❌ Math Crash: {e}")

except Exception as e:
    print(f"❌ FATAL ERROR: {e}")

print("="*40)
