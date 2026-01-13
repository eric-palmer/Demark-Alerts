# utils.py - Institutional Alerts & Formatting
import os
import requests

def fmt_price(val):
    """Formats price nicely: $0.05, $12.50, $24,500"""
    if val is None: return "N/A"
    if val < 1: return f"${val:.4f}"
    if val < 1000: return f"${val:.2f}"
    return f"${val:,.0f}"

def send_telegram(message):
    """Sends message via Telegram Bot"""
    # USE THE STANDARDIZED NAMES
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not token or not chat_id:
        print("⚠️ TELEGRAM SECRETS MISSING in utils.py")
        print(f"[Would have sent]: {message}")
        return

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        resp = requests.post(url, json=payload, timeout=10)
        
        if resp.status_code != 200:
            print(f"❌ Telegram Send Failed: {resp.text}")
        else:
            print("✅ Telegram Message Sent.")
            
    except Exception as e:
        print(f"❌ Telegram Error: {e}")
