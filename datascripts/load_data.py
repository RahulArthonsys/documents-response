from datetime import datetime

from helper import get_collection, insert_thermostat_data, insert_weather_data, insert_emporia_data, insert_logger_data, \
    insert_emporia_device_data, get_emporia_device, update_emporia_device_data, delete_emporia_device, \
    insert_weather_forecast_data, check_record_exist_or_not, insert_weather_forecast_openweather_data, \
    check_openweather_record_exist_or_not


def load_google_data(now):
    try:
        google_data = get_collection('google')
        count = google_data.count_documents({"is_processed": False})
        for gd in google_data.find({"is_processed": False}):
            for device in gd['data']['devices']:
                ambient_temp = device['traits']['sdm.devices.traits.Temperature']['ambientTemperatureCelsius']
                heat_temp = device['traits']['sdm.devices.traits.ThermostatTemperatureSetpoint'].get('heatCelsius')
                cool_temp = device['traits']['sdm.devices.traits.ThermostatTemperatureSetpoint'].get('coolCelsius')
                data_dict = {
                    'device_name': device['name'],
                    'humidity': device['traits']['sdm.devices.traits.Humidity']['ambientHumidityPercent'],
                    'ambient_temp': 9.0/5.0 * int(ambient_temp) + 32,
                    'heat_temp': (9.0/5.0 * int(heat_temp) + 32) if heat_temp else None,
                    'cool_temp': (9.0/5.0 * int(cool_temp) + 32) if cool_temp else None,
                    'temp_scale': device['traits']['sdm.devices.traits.Settings']['temperatureScale'],
                    'mode': device['traits']['sdm.devices.traits.ThermostatMode']['mode'],
                    'fan_mode': device['traits']['sdm.devices.traits.Fan']['timerMode'],
                    'device_type': device['type'],
                    'created_at': gd['datetime'],
                    'device_vendor': 'google',
                    'ref_id': str(gd['_id']),
                }
                insert_thermostat_data(**data_dict)
            google_data.update_one({"_id": gd['_id']}, {"$set": {"is_processed": True}})
        insert_logger_data('google', True, f'Data Loaded to postgres ({count})', now)
    except Exception as e:
        insert_logger_data('google', False, f'Load data to postgres failed ({str(e)})', now)


def load_honeywells_data(now):
    try:
        honeywells_data = get_collection('honeywells')
        count = honeywells_data.count_documents({"is_processed": False})
        for hd in honeywells_data.find({"is_processed": False}):
            for location in hd['data']:
                for device in location['devices']:
                    data_dict = {
                        'device_name': device['deviceSerialNo'],
                        'humidity': device['displayedOutdoorHumidity'],
                        'ambient_temp': device['thermostat']['outdoorTemperature'],
                        'heat_temp': device['thermostat']['changeableValues']['heatSetpoint'],
                        'cool_temp': device['thermostat']['changeableValues']['coolSetpoint'],
                        'temp_scale': device['thermostat']['units'].upper(),
                        'mode': device['thermostat']['changeableValues']['mode'].upper(),
                        'fan_mode': device['settings']['fan']['changeableValues']['mode'].upper(),
                        'device_type': device['deviceType'],
                        'created_at': hd['datetime'],
                        'device_vendor': 'honeywells',
                        'ref_id': str(hd['_id']),
                    }
                    insert_thermostat_data(**data_dict)
            honeywells_data.update_one({"_id": hd['_id']}, {"$set": {"is_processed": True}})
        insert_logger_data('honeywells', True, f'Data Loaded to postgres ({count})', now)
    except Exception as e:
        insert_logger_data('honeywells', False, f'Load data to postgres failed ({str(e)})', now)


def load_weather_data(now):
    try:
        weather_data = get_collection('weather')
        count = weather_data.count_documents({"is_processed": False})
        for wd in weather_data.find({"is_processed": False}):
            data_dict = {
                'humidity': wd['data']['main']['humidity'],
                'temperature': wd['data']['main']['temp'],
                'temperature_min': wd['data']['main']['temp_min'],
                'temperature_max': wd['data']['main']['temp_max'],
                'feels_like': wd['data']['main']['feels_like'],
                'pressure': wd['data']['main']['pressure'],
                'created_at': wd['datetime'],
                'zip_code': wd['zip_code'],
                'ref_id': str(wd['_id']),

            }
            insert_weather_data(**data_dict)
            weather_data.update_one({"_id": wd['_id']}, {"$set": {"is_processed": True}})
        insert_logger_data('weather', True, f'Data Loaded to postgres ({count})', now)
    except Exception as e:
        insert_logger_data('weather', False, f'Load data to postgres failed ({str(e)})', now)


def load_weather_forecast_noaa(now):
    try:
        weather_forecast_data = get_collection('weather_forecast_noaa')
        count = weather_forecast_data.count_documents({"is_processed": False})
        for wd in weather_forecast_data.find({"is_processed": False}):
            for period in wd['data']['properties']['periods']:
                # zip code end_date
                data_dict = {
                    'end_date': period['endTime'],
                    'temperature': period['temperature'],
                    'wind_speed': period['windSpeed'],
                    'zip_code': wd['zip_code'],
                    'ref_id': str(wd['_id']),
                    'created_at': period['endTime']
                }
                record_exists = check_record_exist_or_not(**data_dict)
                if not record_exists:
                    insert_weather_forecast_data(**data_dict)
                    weather_forecast_data.update_one({"_id": wd['_id']}, {"$set": {"is_processed": True}})
        insert_logger_data('weather forecast noaa', True, f'Data Loaded to postgres ({count})', now)
    except Exception as e:
        insert_logger_data('weather forecast noaa', False, f'Load data to postgres failed ({str(e)})', now)


def load_weather_forecast_openweather(now):
    try:
        weather_forecast_openweather_data = get_collection('weather_forecast_openweather')
        count = weather_forecast_openweather_data.count_documents({"is_processed": False})
        for wd in weather_forecast_openweather_data.find({"is_processed": False}):
            for list in wd['data']['list']:
                # zip code end_date
                data_dict = {
                    'humidity': list['main']['humidity'],
                    'temperature': list['main']['temp'],
                    'temperature_min': list['main']['temp_min'],
                    'temperature_max': list['main']['temp_max'],
                    'feels_like': list['main']['feels_like'],
                    'pressure': list['main']['pressure'],
                    'created_at': wd['datetime'],
                    'zip_code': wd['zip_code'],
                    'ref_id': str(wd['_id']),
                }
                record_exists = check_openweather_record_exist_or_not(**data_dict)
                if not record_exists:
                    insert_weather_forecast_openweather_data(**data_dict)
                    weather_forecast_openweather_data.update_one({"_id": wd['_id']}, {"$set": {"is_processed": True}})
        insert_logger_data('weather forecast open weather', True, f'Data Loaded to postgres ({count})', now)
    except Exception as e:
        insert_logger_data('weather forecast open weather', False, f'Load data to postgres failed ({str(e)})', now)


def load_emporia_data(now):
    try:
        emporia_data = get_collection('emporia')
        count = emporia_data.count_documents({"is_processed": False})
        new_device_ids = []
        now = datetime.now()
        device_ids = []
        for ed in emporia_data.find({"is_processed": False}):
            device_ids += get_emporia_device()
            for device in ed['data']['devices']:
                new_device_ids.append(device['manufacturerDeviceId'])
                if device['manufacturerDeviceId'] in device_ids:
                    data_dict = {
                        'device_name': device['deviceName'],
                        'device_id': device['manufacturerDeviceId'],
                    }
                    update_emporia_device_data(**data_dict)
                else:
                    data_dict = {
                        'device_name': device['deviceName'],
                        'device_id': device['manufacturerDeviceId'],
                        'created_at': now,
                    }
                    insert_emporia_device_data(**data_dict)
                if device['usage']:
                    uses_channel_list = device['usage']['deviceUsages'][0]['channelUsages']
                    uses = 0
                    for channel in uses_channel_list:
                        channel_uses = channel['usages']
                        if len(channel_uses) == 6:
                            channel_uses.pop(-1)
                        uses += sum([(x*60)/1000 for x in channel_uses if not isinstance(x, str)])
                        uses_kwh = uses/5
                    data_dict = {
                        'device_name': device['deviceName'],
                        'device_id': device['manufacturerDeviceId'],
                        'created_at': ed['datetime'],
                        'usage': round(uses_kwh, 2),
                        'ref_id': str(ed['_id']),
                    }
                    insert_emporia_data(**data_dict)
                    emporia_data.update_one({"_id": ed['_id']}, {"$set": {"is_processed": True}})
        insert_logger_data('emporia', True, f'Data Loaded to postgres ({count})', now)
        delete_device_ids = set(device_ids) - set(new_device_ids)
        for id in delete_device_ids:
            data_dict = {
                'device_id': id,
            }
            delete_emporia_device(**data_dict)
    except Exception as e:
        insert_logger_data('emporia', False, f'Load data to postgres failed ({str(e)})', now)


def load_data():
    now = datetime.now()
    # load_google_data(now)
    # load_honeywells_data(now)
    load_weather_data(now)
    load_weather_forecast_noaa(now)
    load_weather_forecast_openweather(now)
    load_emporia_data(now)


load_data()