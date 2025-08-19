import requests
import os
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN_WHISPER")
# print(BOT_TOKEN )
resp = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates")
# print(resp.json())
resp_json = resp.json()
print(resp_json['result'][0]['message']['from']['id']) 