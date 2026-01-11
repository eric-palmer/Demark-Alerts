def get_macro():
    """Fetch macro data with comprehensive error handling"""
    try:
        api_key = os.environ.get('FRED_API_KEY')
        
        if not api_key:
            print("ERROR: FRED_API_KEY environment variable is not set")
            return None
        
        print(f"Using FRED API key: {api_key[:5]}...{api_key[-3:]}")
        
        start = datetime.datetime.now() - datetime.timedelta(days=730)
        series = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'DGS10', 'DGS2', 'T5YIE']
        session = get_session()
        
        dfs = []
        failed_series = []
        
        for s in series:
            df = fetch_fred(s, start, api_key, session)
            if not df.empty:
                dfs.append(df)
            else:
                failed_series.append(s)
        
        if not dfs:
            print(f"ERROR: All FRED series failed")
            return None
        
        # Combine FRED data
        fred = pd.concat(dfs, axis=1)
        fred = fred.resample('D').ffill().dropna()
        
        result = {}
        
        # Calculate metrics (handle missing series gracefully)
        if 'WALCL' in fred.columns and 'WTREGEN' in fred.columns and 'RRPONTSYD' in fred.columns:
            result['net_liq'] = (fred['WALCL'] / 1000) - fred['WTREGEN'] - fred['RRPONTSYD']
            result['fed_assets'] = fred['WALCL'] / 1000
        
        if 'DGS10' in fred.columns and 'DGS2' in fred.columns:
            result['term_premia'] = fred['DGS10'] - fred['DGS2']
        
        if 'T5YIE' in fred.columns:
            result['inflation'] = fred['T5YIE']
        
        # Fetch SPY with multiple retries and longer delays
        print("  Fetching SPY data...")
        spy = None
        
        for attempt in range(5):  # Increased retries
            spy = safe_download('SPY')
            if spy is not None:
                print(f"  ✓ SPY data fetched on attempt {attempt + 1}")
                break
            else:
                print(f"  Attempt {attempt + 1}/5 failed for SPY, retrying...")
                time.sleep(5)  # Longer delay between retries
        
        if spy is None:
            print("  WARNING: SPY data unavailable after 5 attempts")
            print("  Continuing without SPY benchmark data")
            # Don't return None - we can still provide macro data without SPY
        else:
            result['spy'] = spy['Close']
        
        if not result:
            print("  ERROR: No usable macro data")
            return None
        
        print(f"  ✓ Macro data ready with {len(result)} metrics")
        return result
        
    except Exception as e:
        print(f"MACRO FETCH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
