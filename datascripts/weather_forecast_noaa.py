import datetime

import eeweather
import requests
from helper import get_collection, insert_logger_data, get_emporia_device_zip

WEATHER_ENDPOINT = 'https://api.weather.gov/'

def dump_data_to_mongodb(data, zip_code):
    weather_forecast = get_collection('weather_forecast_noaa')
    record = {
        'data': data,
        'is_processed': False,
        'zip_code': zip_code,
        'datetime': datetime.datetime.now()
    }
    record_id = weather_forecast.insert_one(record).inserted_id
    return record_id


def get_weather_forecast_noaa(zip_code):
    now = datetime.datetime.now()
    try:
        lat, long = eeweather.zcta_to_lat_long(zip_code)
        endpoint = WEATHER_ENDPOINT + f'points/{lat},{long}'
        response = requests.get(endpoint)
        if response.status_code == 200:
            properties = response.json()['properties']
            grid_id = properties['gridId']
            grid_x = properties['gridX']
            grid_y = properties['gridY']
            forecast_endpoint = WEATHER_ENDPOINT + f'gridpoints/{grid_id}/{grid_x},{grid_y}/forecast/hourly'
            forecast_response = requests.get(forecast_endpoint)
            if forecast_response.status_code == 200:
                record_id = dump_data_to_mongodb(forecast_response.json(), zip_code)
                insert_logger_data('weather forecast noaa', True, f'Data Loaded to mongoDB ({record_id})', now)
            else:
                insert_logger_data('weather forecast noaa', False, f"zip={zip_code} {forecast_response.text}", now)
        else:
            insert_logger_data('weather forecast noaa', False, f"zip={zip_code} {response.text}", now)
    except Exception as e:
        insert_logger_data('weather forecast noaa', False, f"zip={zip_code} {str(e)}", now)


for zip_code in get_emporia_device_zip():
    get_weather_forecast_noaa(zip_code)





