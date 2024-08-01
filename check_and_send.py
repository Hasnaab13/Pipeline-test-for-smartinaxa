import os
import sys
import requests
from datetime import datetime

# Configuration
DOCUMENT_PATH = "qa_inaxa_groundtruth_version08-05-2024_1.xlsx"
TIMESTAMP_FILE = "last_update_timestamp.txt"
BOT_TOKEN = "7427451676:AAHRuoRuj8Wgr_zUUqFIsVmHvuXPmUbzNPM"
CHAT_ID = "5370229480"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"

def get_last_modification_time(file_path):
    return os.path.getmtime(file_path)

def get_stored_timestamp():
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE, 'r') as f:
            return float(f.read().strip())
    return 0

def store_current_timestamp(timestamp):
    with open(TIMESTAMP_FILE, 'w') as f:
        f.write(str(timestamp))

def send_to_telegram(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            TELEGRAM_API_URL,
            data={'chat_id': CHAT_ID},
            files={'document': f}
        )
    return response.json()

def main():
    if not os.path.exists(DOCUMENT_PATH):
        print(f"File {DOCUMENT_PATH} does not exist.")
        sys.exit(1)
    
    current_mod_time = get_last_modification_time(DOCUMENT_PATH)
    last_mod_time = get_stored_timestamp()

    if current_mod_time > last_mod_time:
        print("File has been updated.")
        response = send_to_telegram(DOCUMENT_PATH)
        print(f"Telegram response: {response}")
        store_current_timestamp(current_mod_time)
    else:
        print("File has not been updated.")
        sys.exit(0)

if __name__ == "__main__":
    main()
