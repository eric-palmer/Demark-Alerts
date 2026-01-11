if __name__ == "__main__":
    print("="*60)
    print("INSTITUTIONAL TRADING ALERT SYSTEM")
    print("="*60)
    
    # Pre-flight checks
    print("\n[PRE-FLIGHT CHECKS]")
    telegram_ok = bool(os.environ.get('TELEGRAM_TOKEN') and os.environ.get('TELEGRAM_CHAT_ID'))
    fred_ok = bool(os.environ.get('FRED_API_KEY'))
    
    if not telegram_ok:
        print("⚠️  Telegram credentials missing")
    else:
        print("✓ Telegram configured")
    
    if not fred_ok:
        print("⚠️  FRED_API_KEY missing")
    else:
        print("✓ FRED configured")
    
    # === MACRO ANALYSIS ===
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
            print(f"Macro calculation error: {e}")
            msg += "Macro data partially available\n"
    else:
        msg += "Temporarily unavailable - continuing with analysis\n"
    
    send_telegram(msg)
    
    # === PORTFOLIO ANALYSIS ===
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
    
    # === MARKET SCAN ===
    print("\n[3/4] SCANNING MARKET UNIVERSE")
    universe = list(set(STRATEGIC_TICKERS + get_futures()))
    print(f"  Universe size: {len(universe)} tickers")
    
    power = []
    perfected_demark = []
    imperfected_demark = []
    squeeze_setups = []
    momentum_breakouts = []
    oversold_rsi = []
    overbought_rsi = []
    
    # Track statistics
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
                
                # Categorize signals
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
                
                # Progress updates
                total_processed = successful_scans + failed_scans
                if total_processed % 25 == 0:
                    print(f"  Progress: {total_processed}/{len(universe)} ({successful_scans} success, {failed_scans} failed)")
                    
            except Exception as e:
                failed_scans += 1
                print(f"  ✗ Error processing {ticker}: {e}")
    
    print(f"\n  Scan complete!")
    print(f"  Successfully analyzed: {successful_scans}/{len(universe)}")
    print(f"  Failed: {failed_scans}")
    print(f"  Power setups: {len(power)}")
    print(f"  Perfected DeMark: {len(perfected_demark)}")
    print(f"  Imperfected DeMark: {len(imperfected_demark)}")
    print(f"  Squeeze setups: {len(squeeze_setups)}")
    print(f"  Momentum breakouts: {len(momentum_breakouts)}")
    
    if failed_scans > len(universe) * 0.5:
        print(f"\n  ⚠️  WARNING: High failure rate!")
        print(f"  Sample failures: {', '.join(data_failures[:10])}")
    
    # === ALERTS GENERATION ===
    print("\n[4/4] GENERATING ALERTS")
    
    # POWER RANKINGS
    if power:
        print(f"  Sending {len(power)} power setups...")
        a_msg = "⭐ *POWER RANKINGS* (High Conviction)\n"
        a_msg += "Multiple indicators in strong agreement\n\n"
        
        power_sorted = sorted(power, key=lambda x: x['signal_score'], reverse=True)
        
        for res in power_sorted[:8]:
            a_msg += format_signal_block(res, show_all=False)
        
        send_telegram(a_msg)
    else:
        print("  No power setups found")
    
    # PERFECTED DEMARK
    if perfected_demark:
        print(f"  Sending {len(perfected_demark)} perfected DeMark...")
        dm_msg = "🎯 *PERFECTED DEMARK SIGNALS*\n"
        dm_msg += "Highest probability reversal setups\n\n"
        
        perf_sorted = sorted(perfected_demark, 
                           key=lambda x: (x['demark']['tf'] == 'Weekly', 
                                        x['demark']['signal'] == '13'), 
                           reverse=True)
        
        for res in perf_sorted[:10]:
            dm_msg += format_signal_block(res, show_all=False)
        
        send_telegram(dm_msg)
    else:
        print("  No perfected DeMark signals")
    
    # IMPERFECTED DEMARK
    if imperfected_demark:
        print(f"  Sending {len(imperfected_demark)} imperfected DeMark...")
        im_msg = "○ *IMPERFECTED DEMARK SIGNALS*\n"
        im_msg += "Monitor for perfection\n\n"
        
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
            im_msg += f"• Timing: {res['timing']}\n\n"
        
        send_telegram(im_msg)
    
    # SQUEEZE SETUPS
    if squeeze_setups:
        print(f"  Sending {len(squeeze_setups)} squeeze setups...")
        sq_msg = "💥 *VOLATILITY SQUEEZE SETUPS*\n\n"
        
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
    
    # Summary
    summary = f"\n📊 *SCAN SUMMARY*\n\n"
    summary += f"Universe: {len(universe)} tickers\n"
    summary += f"Successfully Analyzed: {successful_scans}\n"
    summary += f"Failed: {failed_scans}\n\n"
    summary += f"Power Setups: {len(power)}\n"
    summary += f"Perfected DeMark: {len(perfected_demark)}\n"
    summary += f"Imperfected DeMark: {len(imperfected_demark)}\n"
    summary += f"Squeeze Setups: {len(squeeze_setups)}\n"
    summary += f"Momentum Breakouts: {len(momentum_breakouts)}\n"
    summary += f"Oversold (RSI<30): {len(oversold_rsi)}\n"
    summary += f"Overbought (RSI>70): {len(overbought_rsi)}\n"
    
    if failed_scans > 0:
        summary += f"\n⚠️ {failed_scans} tickers failed to fetch data\n"
    
    send_telegram(summary)
    
    print("\n✓ Scan Complete")
    print(f"  Signals sent via Telegram")
