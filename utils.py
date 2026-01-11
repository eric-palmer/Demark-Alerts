# utils.py - Helper functions
import os
import requests

def fmt_price(price):
    """Format price nicely ($0.00)"""
    if price is None: 
        return "N/A"
    try:
        price = float(price)
        if price < 0.1:
            return f"${price:.4f}"
        if price < 1.0:
            return f"${price:.3f}"
        return f"${price:.2f}"
    except:
        return str(price)

def send_telegram(message):
    """Send message to Telegram with error handling"""
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    
    # If secrets are missing, just print to log instead of crashing
    if not token or not chat_id:
        print("⚠️ TELEGRAM SECRETS MISSING")
        print(f"[Would have sent]: {message}")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        # 10 second timeout so the script doesn't freeze
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"Telegram Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Telegram Connection Failed: {e}")

def get_session():
    """Legacy session helper"""
    return requests.Session()
