import datetime
import requests
from helper import get_collection, insert_logger_data, get_emporia_device_zip

WEATHER_APP_ID = 'appid=5a349144c9f927a1d47e81c2bbd8d090'
OPEN_WEATHER_ENDPOINT = f'https://api.openweathermap.org/data/2.5/forecast/?{WEATHER_APP_ID}&units=imperial&zip='


def dump_data_to_mongodb(data, zip_code):
    open_weather_data = get_collection('weather_forecast_openweather')
    record = {
        'data': data,
        'is_processed': False,
        'zip_code': zip_code,
        'datetime': datetime.datetime.now()
    }
    record_id = open_weather_data.insert_one(record).inserted_id
    return record_id


def get_weather_openweather(zip_code):
    now = datetime.datetime.now()
    try:
        endpoint = OPEN_WEATHER_ENDPOINT + zip_code
        headers = {"Content-Type": "application/json; charset=UTF-8"}
        response = requests.get(endpoint, headers=headers)

        if response.status_code == 200:
            record_id = dump_data_to_mongodb(response.json(), zip_code)
            insert_logger_data('weather forecast open weather', True, f'Data Loaded to mongoDB ({record_id})', now)
        else:
            insert_logger_data('weather forecast open weather', False, response.text, now)
    except Exception as e:
        insert_logger_data('weather forecast open weather', False, str(e), now)


for zip_code in get_emporia_device_zip():
    get_weather_openweather(zip_code)
