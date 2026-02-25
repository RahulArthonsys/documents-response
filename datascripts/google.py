import datetime
import os

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
except ImportError:
    pass

from helper import get_refresh_token, get_collection, insert_logger_data

O_AUTH_CODE = os.environ.get('GOOGLE_O_AUTH_CODE', '')
REDIRECT_URL = 'https://anodelabs-access.web.app/auth'

CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')
CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')
PROJECT_ID = os.environ.get('GOOGLE_PROJECT_ID', '')
TOKEN_ENDPOINT = "https://www.googleapis.com/oauth2/v4/token"
GOOGLE_DEVICES_ENDPOINT = 'https://smartdevicemanagement.googleapis.com/v1/enterprises/'


def initialize():
    refresh_token()


def refresh_token():
    now = datetime.datetime.now()
    try:
        payload = {
            'refresh_token': get_refresh_token('google'),
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'grant_type': 'refresh_token'
        }

        response = requests.post(TOKEN_ENDPOINT, json=payload)
        if response.status_code == 200:
            access_token = response.json().get('access_token')
            list_devices(access_token, now)
        else:
            insert_logger_data('google', False, response.text, now)
    except Exception as e:
        insert_logger_data('google', False, str(e), now)


def list_devices(access_token, now):
    endpoint = f"{GOOGLE_DEVICES_ENDPOINT}{PROJECT_ID}/devices"
    headers = {"Authorization": f"Bearer {access_token}",
               "Content-Type": "application/json; charset=UTF-8"}
    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        print(response.text)
        record_id = dump_data_to_mongodb(response.json())
        insert_logger_data('google', True, f'Data Loaded to mongoDB ({record_id})', now)
    else:
        insert_logger_data('google', False, response.text, now)


def dump_data_to_mongodb(data):
    google_data = get_collection('google')
    record = {
        'data': data,
        'is_processed': False,
        'datetime': datetime.datetime.now()
    }
    record_id = google_data.insert_one(record).inserted_id
    print(record_id)
    return record_id


initialize()
