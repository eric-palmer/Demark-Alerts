def check_connection():
    # It looks for the NAME defined in the YAML, not the Secret name
    key = os.environ.get('TIINGO_API_KEY')
    bot = os.environ.get('TELEGRAM_BOT_TOKEN') # The YAML maps this from TELEGRAM_TOKEN
    chat = os.environ.get('TELEGRAM_CHAT_ID')
    
    if not key or not bot or not chat:
        print("❌ CRITICAL: Missing Environment Variables.")
        print(f"   - TIINGO: {'OK' if key else 'MISSING'}")
        print(f"   - BOT_TOKEN: {'OK' if bot else 'MISSING'}")
        print(f"   - CHAT_ID: {'OK' if chat else 'MISSING'}")
        return False
    return True
