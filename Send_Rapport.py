import requests

def send_document(bot_token, chat_id, document_path):
    url = f'https://api.telegram.org/bot{bot_token}/sendDocument'
    files = {'document': open(document_path, 'rb')}
    data = {'chat_id': chat_id}
    response = requests.post(url, files=files, data=data)
    return response.json()

if __name__ == "__main__":
    BOT_TOKEN = "7427451676:AAHRuoRuj8Wgr_zUUqFIsVmHvuXPmUbzNPM"
    CHAT_ID = "5370229480"
    DOCUMENT_PATH = "qa_inaxa_groundtruth_version08-05-2024_1.xlsx"
    
    response = send_document(BOT_TOKEN, CHAT_ID, DOCUMENT_PATH)
    print(response)
