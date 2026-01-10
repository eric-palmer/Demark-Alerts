# utils.py - Utility functions
import requests
import os
import time
import pandas as pd

def send_telegram(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return False
    for i in range(0, len(message), 4000):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": message[i:i+4000], "parse_mode": "Markdown"},
                timeout=10
            )
            if not r.ok:
                print(f"Telegram error: {r.status_code}")
            time.sleep(1)
        except Exception as e:
            print(f"Telegram error: {e}")
            return False
    return True

def fmt_price(p):
    if p is None or pd.isna(p):
        return "N/A"
    p = float(p)
    if p < 0.01:
        return f"${p:.6f}"
    if p < 1:
        return f"${p:.4f}"
    return f"${p:.2f}"

def get_session():
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': '*/*'
    })
    return s
